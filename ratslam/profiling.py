import csv
import os
import time
import tracemalloc

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


class FrameProfiler(object):
    """Per-frame latency profiler tailored for embedded benchmarking
    (e.g. Jetson Nano).

    Records per-frame timings for each pipeline stage plus a total,
    computes percentile statistics (min / p50 / avg / p95 / p99 / max),
    and exports both a human-readable report and a per-frame CSV
    suitable for offline plotting.
    """

    MODULES = ("module_a", "module_b", "module_c", "module_d", "total")
    MODULE_LABELS = {
        "module_a": "Time surface + undistort",
        "module_b": "VO core (track + pose)",
        "module_c": "Scene descriptor",
        "module_d": "RatSLAM backend",
        "total":    "End-to-end per frame",
    }

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.track_memory = enabled
        self.frames = []
        self._ticks = {}
        self._current = {}
        self._wall_start = None
        self._wall_end = None
        self._proc = None
        self.energy_stats = None
        self.energy_sensors = None
        if self.enabled and _HAS_PSUTIL:
            try:
                self._proc = psutil.Process(os.getpid())
                self._proc.cpu_percent(None)  # prime CPU% counter
            except Exception:
                self._proc = None
        if self.track_memory:
            tracemalloc.start()

    def attach_energy(self, energy_stats, sensor_paths=None):
        """Attach energy statistics gathered by an external sampler so they
        end up in both ``write_report`` and ``print_summary``.
        """
        self.energy_stats = energy_stats
        self.energy_sensors = list(sensor_paths) if sensor_paths else None

    def start_frame(self):
        if not self.enabled:
            return
        if self._wall_start is None:
            self._wall_start = time.perf_counter()
        self._current = {k: 0.0 for k in self.MODULES}
        self._ticks["frame"] = time.perf_counter()

    def tic(self, key):
        if not self.enabled:
            return
        self._ticks[key] = time.perf_counter()

    def toc(self, key):
        if not self.enabled:
            return
        self._current[key] += time.perf_counter() - self._ticks.get(key, time.perf_counter())

    def end_frame(self):
        if not self.enabled:
            return
        self._current["total"] = time.perf_counter() - self._ticks.get("frame", time.perf_counter())
        self.frames.append(self._current.copy())
        self._wall_end = time.perf_counter()

    def _series(self, key):
        return [f[key] for f in self.frames]

    @staticmethod
    def _percentile(sorted_vals, p):
        if not sorted_vals:
            return 0.0
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        k = (len(sorted_vals) - 1) * p
        f = int(k)
        c = min(f + 1, len(sorted_vals) - 1)
        if f == c:
            return sorted_vals[f]
        return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

    def _stats(self, key):
        s = sorted(self._series(key))
        if not s:
            return None
        return {
            "min": s[0],
            "p50": self._percentile(s, 0.50),
            "avg": sum(s) / len(s),
            "p95": self._percentile(s, 0.95),
            "p99": self._percentile(s, 0.99),
            "max": s[-1],
        }

    def _wall_seconds(self):
        if self._wall_start is None or self._wall_end is None:
            return 0.0
        return max(0.0, self._wall_end - self._wall_start)

    def write_csv(self, csv_path):
        """Dump per-frame latencies (seconds) to CSV for offline analysis."""
        if not self.enabled or not self.frames:
            return
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx"] + list(self.MODULES))
            for i, fr in enumerate(self.frames):
                w.writerow([i] + [f"{fr[k]:.9f}" for k in self.MODULES])

    def _format_report_lines(self, n_view_cells, n_experiences, n_loop_closures):
        if not self.enabled:
            return ["Profiling disabled."]
        if not self.frames:
            return ["Profiling enabled but no frames were processed."]

        wall = self._wall_seconds()
        n = len(self.frames)
        total_stats = self._stats("total")
        avg_total_ms = total_stats["avg"] * 1000.0
        running_fps = 1.0 / max(total_stats["avg"], 1e-9)
        wall_fps = n / wall if wall > 1e-9 else 0.0

        lines = []
        lines.append("Event NeuroSLAM Profiling Report")
        lines.append(f"Frames profiled    : {n}")
        lines.append(f"Wall-clock runtime : {wall:.3f} s")
        lines.append(f"Throughput (wall)  : {wall_fps:.2f} fps")
        lines.append(f"Throughput (avg t) : {running_fps:.2f} fps")
        lines.append("")
        lines.append("Per-module latency (ms):")
        header = f"  {'module':<28s}{'min':>8s}{'p50':>8s}{'avg':>8s}{'p95':>8s}{'p99':>8s}{'max':>8s}"
        lines.append(header)
        for key in self.MODULES:
            st = self._stats(key)
            if st is None:
                continue
            label = self.MODULE_LABELS.get(key, key)
            lines.append(
                f"  {label:<28s}"
                f"{st['min']*1000:>8.2f}"
                f"{st['p50']*1000:>8.2f}"
                f"{st['avg']*1000:>8.2f}"
                f"{st['p95']*1000:>8.2f}"
                f"{st['p99']*1000:>8.2f}"
                f"{st['max']*1000:>8.2f}"
            )
        lines.append("")
        lines.append(f"Avg total latency  : {avg_total_ms:.3f} ms ({running_fps:.2f} fps equivalent)")

        if self.track_memory:
            try:
                cur_mem, peak_mem = tracemalloc.get_traced_memory()
                lines.append(f"Python heap peak   : {peak_mem / (1024 * 1024):.2f} MB "
                             f"(current {cur_mem / (1024 * 1024):.2f} MB)")
            except Exception:
                pass

        if self._proc is not None:
            try:
                rss_mb = self._proc.memory_info().rss / (1024 * 1024)
                cpu_pct = self._proc.cpu_percent(None)
                lines.append(f"Process RSS        : {rss_mb:.2f} MB")
                lines.append(f"Process CPU usage  : {cpu_pct:.1f} % (since profiling start)")
            except Exception:
                pass

        if self.energy_stats is not None:
            lines.append("")
            lines.append("Power / energy (Jetson INA3221):")
            es = self.energy_stats
            if self.energy_sensors:
                for path in self.energy_sensors:
                    lines.append(f"  rail            : {path}")
            lines.append(f"  sample period   : {es.sample_period_s * 1000:.1f} ms")
            lines.append(f"  samples         : {es.samples}")
            lines.append(f"  measured for    : {es.duration_s:.3f} s")
            if es.avg_power_w is not None:
                lines.append(f"  average power   : {es.avg_power_w:.3f} W")
                lines.append(f"  peak power      : {es.peak_power_w:.3f} W")
                lines.append(f"  total energy    : {es.energy_j:.3f} J")
                if n > 0:
                    energy_per_frame = es.energy_j / n
                    lines.append(f"  energy / frame  : {energy_per_frame * 1000:.3f} mJ")
            else:
                lines.append("  (sampler did not capture any readings)")

        lines.append("")
        lines.append(f"View cells created : {n_view_cells}")
        lines.append(f"Experience nodes   : {n_experiences}")
        lines.append(f"Loop closures      : {n_loop_closures}")
        return lines

    def write_report(self, path, n_view_cells=0, n_experiences=0, n_loop_closures=0):
        lines = self._format_report_lines(n_view_cells, n_experiences, n_loop_closures)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        # Per-frame CSV for offline analysis (same directory as report).
        if self.enabled and self.frames:
            csv_path = os.path.splitext(path)[0] + "_per_frame.csv"
            self.write_csv(csv_path)

    def print_summary(self, n_view_cells=0, n_experiences=0, n_loop_closures=0):
        """Print the profiling summary to stdout (in addition to the file)."""
        if not self.enabled:
            return
        for line in self._format_report_lines(n_view_cells, n_experiences, n_loop_closures):
            print(line)
