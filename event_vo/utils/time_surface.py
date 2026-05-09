"""
Time Surface Generator.

Converts asynchronous event streams into fixed-timestep frame-like
images using exponential decay. Each pixel stores the recency of
its last event, producing smooth edge-map images suitable for
corner detection and optical flow.
"""

import numpy as np
from typing import Generator, Tuple


class TimeSurfaceGenerator:
    """
    Generates time-surface frames at fixed temporal intervals.

    For each timestep dt, the time surface is:
        T(x, y) = exp(-(t_now - t_last(x, y)) / tau)
    where t_last is the timestamp of the most recent event at (x, y).
    Pixels with no recent event decay to zero.
    """

    def __init__(self, width: int, height: int, dt_us: float, tau_us: float):
        self.width = width
        self.height = height
        self.dt_us = dt_us
        self.tau_us = tau_us

    def generate(self, events: np.ndarray) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Yield (time_surface_uint8, timestamp_s) at each fixed timestep.
        """
        t_start = events[0, 0]
        t_end = events[-1, 0]

        t_last = np.full((self.height, self.width), -1e12, dtype=np.float64)

        idx = 0
        n_events = len(events)

        t_frame = t_start + self.dt_us
        while t_frame <= t_end:
            while idx < n_events and events[idx, 0] <= t_frame:
                x = int(events[idx, 1])
                y = int(events[idx, 2])
                t = events[idx, 0]
                if 0 <= x < self.width and 0 <= y < self.height:
                    t_last[y, x] = t
                idx += 1

            dt = t_frame - t_last
            surface = np.exp(-dt / self.tau_us)
            surface_u8 = np.clip(surface * 255, 0, 255).astype(np.uint8)

            yield surface_u8, t_frame / 1e6

            t_frame += self.dt_us


class TimeSurfaceGeneratorFast:
    """
    Vectorized version — processes events in batch per frame.
    Significantly faster for large event files.
    """

    def __init__(self, width: int, height: int, dt_us: float, tau_us: float):
        self.width = width
        self.height = height
        self.dt_us = dt_us
        self.tau_us = tau_us

    def generate(self, events: np.ndarray) -> Generator[Tuple[np.ndarray, float], None, None]:
        t_start = events[0, 0]
        t_end = events[-1, 0]
        timestamps = events[:, 0]

        t_last = np.full((self.height, self.width), -1e12, dtype=np.float64)
        idx = 0
        n = len(events)

        t_frame = t_start + self.dt_us
        while t_frame <= t_end:
            end_idx = np.searchsorted(timestamps, t_frame, side="right")

            if end_idx > idx:
                batch = events[idx:end_idx]
                xs = batch[:, 1].astype(np.int32)
                ys = batch[:, 2].astype(np.int32)
                ts = batch[:, 0]

                valid = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
                xs, ys, ts = xs[valid], ys[valid], ts[valid]

                t_last[ys, xs] = ts

                idx = end_idx

            surface = np.exp(-(t_frame - t_last) / self.tau_us)
            surface_u8 = np.clip(surface * 255, 0, 255).astype(np.uint8)

            yield surface_u8, t_frame / 1e6

            t_frame += self.dt_us
