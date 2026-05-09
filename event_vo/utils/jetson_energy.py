"""
Jetson on-board power / energy sampling via INA3221 sysfs nodes.

Tested on:
  - Jetson Orin Nano (vdd_in rail under /sys/bus/i2c/drivers/ina3221/...)
  - Jetson Nano (per-rail in*_input + curr*_input)

Usage
-----

    from event_vo.utils.jetson_energy import JetsonEnergySampler

    sampler = JetsonEnergySampler(period_s=0.1)
    if sampler.available():
        sampler.start()
        ...   # run the workload
        stats = sampler.stop()
        print(f"avg power = {stats.avg_power_w:.2f} W")
        print(f"energy    = {stats.energy_j:.2f} J")
    else:
        print("no INA3221 power sensors visible (not running on Jetson?)")

Implementation note
-------------------
The discovery / integration logic is taken from the reference
`jetson_energy.py` script. We add a small background thread on top
so the sampler runs without the workload having to call
`sample_once()` manually.
"""

from __future__ import annotations

import glob
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read().strip()
        return int(s)
    except Exception:
        return None


@dataclass(frozen=True)
class _PowerSensor:
    kind: str
    power_path: Optional[str] = None
    voltage_path: Optional[str] = None
    current_path: Optional[str] = None
    label: Optional[str] = None

    def display_path(self) -> str:
        if self.kind == "power_path" and self.power_path is not None:
            return self.power_path
        if self.voltage_path is not None and self.current_path is not None:
            return f"{self.voltage_path}|{self.current_path}"
        return "<unknown>"


def _discover_vdd_in_sensor() -> Optional[_PowerSensor]:
    for hwmon_dir in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"):
        for i in range(1, 5):
            label_path = os.path.join(hwmon_dir, f"in{i}_label")
            voltage_path = os.path.join(hwmon_dir, f"in{i}_input")
            current_path = os.path.join(hwmon_dir, f"curr{i}_input")
            if not (os.path.exists(label_path) and os.path.exists(voltage_path)
                    and os.path.exists(current_path)):
                continue
            try:
                label = open(label_path, "r", encoding="utf-8").read().strip()
            except Exception:
                continue
            if label == "VDD_IN":
                return _PowerSensor(
                    kind="hwmon_vi",
                    voltage_path=voltage_path,
                    current_path=current_path,
                    label=label,
                )
    return None


def _discover_direct_power_sensors() -> List[_PowerSensor]:
    """Older Jetson kernels expose direct power readings as
    ``in_power*_input`` under the INA3221 IIO nodes.
    """
    candidates: List[str] = []
    patterns = [
        "/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power*_input",
        "/sys/bus/i2c/drivers/ina3221x/*/iio_device/in_power*_input",
        "/sys/bus/i2c/drivers/ina3221x/*/in_power*_input",
    ]
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    return [
        _PowerSensor(kind="power_path", power_path=path)
        for path in sorted(set(candidates))
    ]


def _discover_hwmon_power_sensors() -> List[_PowerSensor]:
    """Per-rail INA3221 readings via hwmon (Jetson Nano style).

    inN_input   -> bus voltage in millivolts
    currN_input -> current in milliamps
    Power = (mV * mA) / 1e6 W
    """
    sensors: List[_PowerSensor] = []
    patterns = [
        "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/curr*_input",
        "/sys/bus/i2c/drivers/ina3221/*/curr*_input",
    ]
    for pattern in patterns:
        for current_path in sorted(glob.glob(pattern)):
            base, filename = os.path.split(current_path)
            if not filename.startswith("curr") or not filename.endswith("_input"):
                continue
            channel = filename[len("curr"):-len("_input")]
            voltage_path = os.path.join(base, f"in{channel}_input")
            if not os.path.exists(voltage_path):
                continue
            label_path = os.path.join(base, f"in{channel}_label")
            label = None
            if os.path.exists(label_path):
                try:
                    label = open(label_path, "r", encoding="utf-8").read().strip()
                except Exception:
                    label = None
            sensors.append(
                _PowerSensor(
                    kind="hwmon_vi",
                    voltage_path=voltage_path,
                    current_path=current_path,
                    label=label,
                )
            )

    deduped: List[_PowerSensor] = []
    seen = set()
    for sensor in sensors:
        key = (sensor.voltage_path, sensor.current_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sensor)
    return deduped


def _discover_power_sensors() -> List[_PowerSensor]:
    """Pick the best available rail set:

    1. The total VDD_IN rail (Orin family).
    2. Direct power readings (older IIO kernels).
    3. Per-rail V/I readings via hwmon (older Nano kernels).
    """
    vdd_in = _discover_vdd_in_sensor()
    if vdd_in is not None:
        return [vdd_in]

    direct = _discover_direct_power_sensors()
    if direct:
        return direct

    hwmon = _discover_hwmon_power_sensors()
    if hwmon:
        return hwmon

    return []


def _power_w_from_sensors(sensors: List[_PowerSensor]) -> Optional[float]:
    total_w = 0.0
    found = False

    for sensor in sensors:
        if sensor.kind == "power_path" and sensor.power_path is not None:
            raw_power = _read_int(sensor.power_path)
            if raw_power is None:
                continue
            # Some kernels report uW, some mW. Heuristic: large -> uW.
            if raw_power >= 100000:
                total_w += raw_power / 1e6
            else:
                total_w += raw_power / 1e3
            found = True
            continue

        if sensor.voltage_path is None or sensor.current_path is None:
            continue
        mv = _read_int(sensor.voltage_path)
        ma = _read_int(sensor.current_path)
        if mv is None or ma is None:
            continue
        total_w += (mv * ma) / 1e6
        found = True

    if not found:
        return None
    return total_w


def _sensor_exists(sensor: _PowerSensor) -> bool:
    if sensor.kind == "power_path":
        return bool(sensor.power_path) and os.path.exists(sensor.power_path)
    return (bool(sensor.voltage_path and sensor.current_path)
            and os.path.exists(sensor.voltage_path)
            and os.path.exists(sensor.current_path))


@dataclass
class EnergyStats:
    avg_power_w: Optional[float]
    peak_power_w: Optional[float]
    energy_j: Optional[float]
    samples: int
    duration_s: float
    sample_period_s: float


class JetsonEnergySampler:
    """Background-thread power sampler.

    Energy = sum(P * dt) over samples, with dt taken from real timestamps.
    """

    def __init__(self, period_s: float = 0.1):
        self.period_s = float(max(0.02, period_s))
        self._sensors = _discover_power_sensors()
        self._p_samples: List[float] = []
        self._t_samples: List[float] = []
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t_start: Optional[float] = None
        self._t_stop: Optional[float] = None

    def available(self) -> bool:
        return bool(self._sensors) and _sensor_exists(self._sensors[0])

    def sensor_paths(self) -> List[str]:
        return [s.display_path() for s in self._sensors]

    def sensor_labels(self) -> List[str]:
        return [s.label or "<no-label>" for s in self._sensors]

    def _loop(self) -> None:
        next_t = time.perf_counter()
        while not self._stop_evt.is_set():
            p = _power_w_from_sensors(self._sensors)
            now = time.time()
            if p is not None:
                self._p_samples.append(float(p))
                self._t_samples.append(now)
            next_t += self.period_s
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                # Bounded wait so that stop() returns promptly.
                self._stop_evt.wait(sleep_for)
            else:
                # Fell behind schedule; reset to avoid drift accumulation.
                next_t = time.perf_counter()

    def start(self) -> None:
        if not self.available():
            return
        self._p_samples.clear()
        self._t_samples.clear()
        self._stop_evt.clear()
        self._t_start = time.time()
        self._thread = threading.Thread(target=self._loop,
                                        name="JetsonEnergySampler",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> EnergyStats:
        self._t_stop = time.time()
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=2.0 + self.period_s * 2)
            self._thread = None

        ps = self._p_samples
        ts = self._t_samples
        if not ps:
            return EnergyStats(
                avg_power_w=None, peak_power_w=None, energy_j=None,
                samples=0,
                duration_s=(self._t_stop - self._t_start) if self._t_start else 0.0,
                sample_period_s=self.period_s,
            )

        # Trapezoidal-ish integration using actual sample timestamps.
        energy = 0.0
        for i in range(1, len(ps)):
            dt = ts[i] - ts[i - 1]
            if dt <= 0:
                dt = self.period_s
            energy += 0.5 * (ps[i] + ps[i - 1]) * dt
        # First sample contributes the gap from t_start to ts[0].
        if self._t_start is not None and ts:
            head_dt = max(0.0, ts[0] - self._t_start)
            energy += ps[0] * head_dt

        return EnergyStats(
            avg_power_w=sum(ps) / len(ps),
            peak_power_w=max(ps),
            energy_j=energy,
            samples=len(ps),
            duration_s=(self._t_stop - self._t_start) if self._t_start else 0.0,
            sample_period_s=self.period_s,
        )
