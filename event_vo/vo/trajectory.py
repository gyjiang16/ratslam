"""
Trajectory Accumulator.

Integrates per-frame (R, t) pose estimates into a global trajectory.
Outputs (x, y, z, yaw) at each timestep for visualization and
downstream use in RatSLAM.
"""

import numpy as np
from collections import deque
from typing import List, Tuple
from .pose_estimator import PoseEstimate


class ManhattanFilter:
    """
    Hybrid Manhattan-world heading filter with two snap pathways.

    Pathway 1 – EMA heading (works for all turns with detectable heading):
        Cumulative heading is smoothed with an EMA.  When the EMA
        deviates from the current cardinal by > snap_threshold,
        the cardinal snaps to the nearest 90-deg multiple.

    Pathway 2 – Sustained low similarity (catches dramatic turns where
        heading estimation fails):
        When frame similarity stays below *sim_very_low* for at least
        *min_turn_duration* consecutive frames, the cardinal is
        incremented by ±90 deg using the accumulated heading sign.
        A cooldown prevents multiple triggers during a single turn.

    Combined, the two pathways robustly detect both moderate turns
    (where the scene changes little but heading is measurable) and
    dramatic turns (where the scene changes a lot but heading is
    unreliable).
    """

    STRAIGHT = 0
    TURNING = 1

    def __init__(self, turn_threshold_deg: float = 45.0,
                 decay: float = 0.98,
                 sim_very_low: float = 0.35,
                 min_turn_duration: int = 12,
                 min_heading_for_duration: float = 3.0,
                 cooldown_frames: int = 60,
                 warmup_frames: int = 50):
        self.threshold = turn_threshold_deg
        self.decay = decay
        self.sim_very_low = sim_very_low
        self.min_turn_dur = min_turn_duration
        self.min_heading_dur = min_heading_for_duration
        self.cooldown_len = cooldown_frames
        self.warmup_len = warmup_frames

        self.heading_accum = 0.0
        self.cardinal = 0.0

        self._sustained_low = 0
        self._dur_accum = 0.0
        self._cooldown = 0
        self._frame_count = 0

        self.state = self.STRAIGHT
        self.position = np.zeros(3, dtype=np.float64)
        self.positions: List[np.ndarray] = [np.zeros(3)]

    def update(self, similarity: float, heading_deg: float, speed: float):
        self._frame_count += 1

        # --- Pathway 1: Leaky integrator of heading ---
        # Old contributions decay exponentially, preventing long-term drift.
        # A burst of heading during a turn accumulates quickly and triggers
        # a snap; steady-state noise from straight walking saturates at
        # noise / (1 - decay) which stays well below the snap threshold.
        self.heading_accum = self.heading_accum * self.decay + heading_deg

        # --- Pathway 2: Sustained very-low-similarity counter ---
        if similarity < self.sim_very_low:
            self._sustained_low += 1
            self._dur_accum += heading_deg
        else:
            if similarity > self.sim_very_low + 0.10:
                self._sustained_low = 0
                self._dur_accum = 0.0

        # --- Snap logic (skip warmup, respect cooldown) ---
        if self._cooldown > 0:
            self._cooldown -= 1
        elif self._frame_count > self.warmup_len:
            leaky_snap = abs(self.heading_accum) > self.threshold
            dur_snap = (self._sustained_low >= self.min_turn_dur
                        and abs(self._dur_accum) > self.min_heading_dur)

            if leaky_snap or dur_snap:
                direction = 1.0 if self.heading_accum > 0 else -1.0
                self.cardinal += direction * 90.0
                self.heading_accum = 0.0
                self._cooldown = self.cooldown_len
                self._sustained_low = 0
                self._dur_accum = 0.0

        is_turning = similarity < self.sim_very_low
        self.state = self.TURNING if is_turning else self.STRAIGHT

        h = np.radians(self.cardinal)
        self.position = self.position + speed * np.array(
            [-np.sin(h), 0.0, np.cos(h)])
        self.positions.append(self.position.copy())

    def get_position_3x1(self) -> np.ndarray:
        return self.position.reshape(3, 1)


class TrajectoryAccumulator:
    """
    Maintains a running global pose by chaining inter-frame transforms.
    
    Global pose at time k (W2C convention):
        R_global(k) = R(k) @ R_global(k-1)
        t_global(k) = t_global(k-1) - scale * R_global(k).T @ t(k)

    recoverPose returns t pointing from cam2 toward cam1 in cam2's frame,
    so the world-frame displacement is -R_wc^T @ t.
    """

    def __init__(self):
        self.R_global = np.eye(3, dtype=np.float64)
        self.t_global = np.zeros((3, 1), dtype=np.float64)

        self.positions: List[np.ndarray] = [self.t_global.copy()]
        self.orientations: List[np.ndarray] = [self.R_global.copy()]
        self.timestamps: List[float] = []
        self.headings: List[float] = [0.0]
        self.speeds: List[float] = [0.0]
        self.cumulative_heading: float = 0.0

    def update(self, pose: PoseEstimate, timestamp: float, scale: float = 1.0):
        """
        Integrate a new pose estimate into the global trajectory.
        
        Args:
            pose: PoseEstimate from the pose estimator.
            timestamp: Frame timestamp in seconds.
            scale: Translation scale factor.
        """
        if pose.success:
            self.R_global = pose.R @ self.R_global
            self.t_global = self.t_global - scale * self.R_global.T @ pose.t
            self.cumulative_heading += pose.heading_deg

        self.positions.append(self.t_global.copy())
        self.orientations.append(self.R_global.copy())
        self.timestamps.append(timestamp)
        self.headings.append(self.cumulative_heading)
        self.speeds.append(pose.translation_norm if pose.success else 0.0)

    def get_xy_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x_array, z_array) for top-down 2D trajectory plot."""
        positions = np.array([p.ravel() for p in self.positions])
        return positions[:, 0], positions[:, 2]

    def get_full_trajectory(self) -> np.ndarray:
        """Return (N, 3) array of [x, y, z] positions."""
        return np.array([p.ravel() for p in self.positions])

    def get_manhattan_trajectory(self, turn_threshold_deg: float = 45.0,
                                    smooth_window: int = 31
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct trajectory with Manhattan-world constraint.

        Heading is smoothed and then quantized to the nearest 90-degree grid.
        Straight segments are linearized so that jitter is removed entirely.

        Args:
            turn_threshold_deg: how far the smoothed heading must deviate
                from the current cardinal direction before snapping to the
                next 90-degree multiple.  Larger = less sensitive.
            smooth_window: moving-average window (odd number) applied to the
                raw cumulative heading before snapping.

        Returns:
            (positions_Nx3, filtered_headings_N)
        """
        raw = np.array(self.headings, dtype=np.float64)
        speeds = np.array(self.speeds, dtype=np.float64)
        n = len(raw)

        if smooth_window > 1:
            hw = smooth_window // 2
            padded = np.pad(raw, hw, mode="edge")
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(padded, kernel, mode="valid")[:n]
        else:
            smoothed = raw.copy()

        filtered = np.zeros(n)
        cardinal = round(smoothed[0] / 90.0) * 90.0
        for i in range(n):
            if abs(smoothed[i] - cardinal) > turn_threshold_deg:
                cardinal = round(smoothed[i] / 90.0) * 90.0
            filtered[i] = cardinal

        positions = np.zeros((n, 3))
        for i in range(1, n):
            h = np.radians(filtered[i])
            s = speeds[i]
            positions[i] = positions[i - 1] + np.array(
                [-s * np.sin(h), 0.0, s * np.cos(h)]
            )

        heading_diff = np.diff(filtered)
        change_idx = np.where(heading_diff != 0)[0] + 1
        boundaries = np.concatenate([[0], change_idx, [n]])

        result = positions.copy()
        for k in range(len(boundaries) - 1):
            start, end = int(boundaries[k]), int(boundaries[k + 1])
            seg_len = end - start
            if seg_len > 2:
                for dim in (0, 2):
                    result[start:end, dim] = np.linspace(
                        positions[start, dim],
                        positions[end - 1, dim],
                        seg_len,
                    )

        return result, filtered

    def get_odometry_for_ratslam(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (speeds, headings) arrays suitable for RatSLAM pose cell input.
        speeds:   translational velocity per timestep
        headings: heading change in degrees per timestep
        """
        return np.array(self.speeds), np.array(self.headings)
