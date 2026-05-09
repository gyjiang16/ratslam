"""
Event-Based Visual Odometry Pipeline.

Orchestrates: event loading → time surface → feature tracking → pose → trajectory.
"""

import numpy as np
import cv2
from typing import Optional, Callable

from ..config import VOConfig
from ..utils.event_loader import load_events, get_time_range
from ..utils.time_surface import TimeSurfaceGeneratorFast
from .feature_tracker import FeatureTracker
from .pose_estimator import PoseEstimator
from .trajectory import TrajectoryAccumulator, ManhattanFilter


class EventVO:
    """
    Complete event-based visual odometry pipeline.

    Usage:
        vo = EventVO(config)
        trajectory = vo.run("events.csv")
    """

    def __init__(self, config: Optional[VOConfig] = None,
                 manhattan: bool = False,
                 turn_threshold_deg: float = 45.0):
        self.cfg = config or VOConfig()
        self.manhattan = manhattan
        self.turn_threshold_deg = turn_threshold_deg
        self.manhattan_filter: Optional[ManhattanFilter] = None
        self._setup()

    def _setup(self):
        """Initialize all sub-modules from config."""
        cam = self.cfg.camera
        ts = self.cfg.time_surface
        feat = self.cfg.feature
        pose = self.cfg.pose

        self.ts_generator = TimeSurfaceGeneratorFast(
            width=cam.width,
            height=cam.height,
            dt_us=ts.dt_ms * 1000,
            tau_us=ts.tau_ms * 1000
        )

        self.tracker = FeatureTracker(
            max_corners=feat.max_corners,
            quality_level=feat.quality_level,
            min_distance=feat.min_distance,
            block_size=feat.block_size,
            lk_win_size=feat.lk_win_size,
            lk_max_level=feat.lk_max_level,
            fb_threshold=feat.fb_threshold,
            min_features=feat.min_features_for_pose,
            redetect_ratio=feat.redetect_ratio
        )

        self.pose_estimator = PoseEstimator(
            K=cam.K,
            ransac_threshold=pose.ransac_threshold,
            ransac_confidence=pose.ransac_confidence,
            min_inliers=pose.min_inliers,
            default_scale=pose.default_scale
        )

        self.trajectory = TrajectoryAccumulator()

        # Precompute undistortion maps if distortion coefficients provided
        self._undistort_map = None
        if cam.dist is not None:
            self._undistort_map = cv2.initUndistortRectifyMap(
                cam.K, cam.dist, None, cam.K,
                (cam.width, cam.height), cv2.CV_32FC1
            )

    def run(self, event_file: str,
            max_frames: Optional[int] = None,
            callback: Optional[Callable] = None,
            verbose: bool = True) -> TrajectoryAccumulator:
        """
        Run the full VO pipeline on an event file.

        Args:
            event_file: Path to CSV/NPZ/AEDAT4 file.
            max_frames: Process at most this many frames (None = all).
            callback: Optional function called per frame with
                      (frame_idx, time_surface, tracked_features, pose).
            verbose: Print progress.

        Returns:
            TrajectoryAccumulator with the estimated trajectory.
        """
        if verbose:
            print(f"Loading events from {event_file}...")

        events = load_events(event_file)
        t_start, t_end, duration = get_time_range(events)

        if verbose:
            print(f"  Events: {len(events):,}")
            print(f"  Duration: {duration / 1e6:.2f} s")
            print(f"  Time range: [{t_start:.0f}, {t_end:.0f}] us")
            print(f"  Frame dt: {self.cfg.time_surface.dt_ms} ms")
            expected_frames = int(duration / (self.cfg.time_surface.dt_ms * 1000))
            print(f"  Expected frames: ~{expected_frames}")
            print()

        frame_idx = 0
        n_success = 0
        n_fail = 0
        fx = self.cfg.camera.fx
        scale = self.cfg.pose.default_scale

        if self.manhattan:
            self.manhattan_filter = ManhattanFilter(
                turn_threshold_deg=self.turn_threshold_deg)

        prev_surface_u8 = None
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 6))

        for surface, timestamp in self.ts_generator.generate(events):
            if max_frames is not None and frame_idx >= max_frames:
                break

            if self._undistort_map is not None:
                surface = cv2.remap(surface, self._undistort_map[0],
                                    self._undistort_map[1], cv2.INTER_LINEAR)

            surface = cv2.GaussianBlur(surface, (3, 3), 1.0)

            # ---- Feature tracking + Essential Matrix (raw trajectory) ----
            tracked = self.tracker.process_frame(surface)
            pose = None
            frame_speed = scale

            if tracked is not None:
                pose = self.pose_estimator.estimate(
                    tracked.pts_prev, tracked.pts_curr)
                self.trajectory.update(pose, timestamp, scale=scale)
                if pose.success:
                    n_success += 1
                else:
                    n_fail += 1
            else:
                if frame_idx > 0:
                    from .pose_estimator import PoseEstimate
                    self.trajectory.update(
                        PoseEstimate(np.eye(3), np.zeros((3, 1)),
                                     0, 0, 0, 0, False),
                        timestamp)
                    n_fail += 1
                else:
                    frame_speed = 0.0

            # ---- Frame similarity (Pearson r on CLAHE-enhanced surfaces) ----
            similarity = 1.0
            if prev_surface_u8 is not None:
                prev_enh = clahe.apply(prev_surface_u8)
                curr_enh = clahe.apply(surface)
                pf = prev_enh.ravel().astype(np.float64)
                cf = curr_enh.ravel().astype(np.float64)
                ps, cs = pf.std(), cf.std()
                if ps > 1.0 and cs > 1.0:
                    similarity = float(np.corrcoef(pf, cf)[0, 1])
                    if np.isnan(similarity):
                        similarity = 1.0

            # ---- Heading estimate ----
            heading_deg = 0.0
            mf = self.tracker.last_mean_flow
            if pose is not None and pose.success:
                heading_deg = pose.heading_deg
            elif mf is not None and abs(mf[0]) > 0.3:
                heading_deg = np.degrees(-mf[0] / fx)
            

            prev_surface_u8 = surface.copy()

            if self.manhattan_filter is not None:
                self.manhattan_filter.update(similarity, heading_deg,
                                             frame_speed)

            if verbose and frame_idx > 0 and frame_idx % 50 == 0:
                if self.manhattan_filter is not None:
                    mf_obj = self.manhattan_filter
                    mp = mf_obj.position
                    state_str = ("TURN" if mf_obj.state == ManhattanFilter.TURNING
                                 else "STR ")
                    print(f"  Frame {frame_idx:4d} | t={timestamp:.3f}s | "
                          f"{state_str} sim={similarity:.2f} "
                          f"cardinal={mf_obj.cardinal:+.0f}deg | "
                          f"pos=({mp[0]:+.1f}, {mp[2]:+.1f})")
                else:
                    pos = self.trajectory.t_global.ravel()
                    n_tr = tracked.num_tracked if tracked else 0
                    n_in = pose.num_inliers if pose else 0
                    print(f"  Frame {frame_idx:4d} | t={timestamp:.3f}s | "
                          f"tracked={n_tr:3d} | inliers={n_in:3d} | "
                          f"heading={self.trajectory.cumulative_heading:+.1f}deg | "
                          f"pos=({pos[0]:+.2f}, {pos[2]:+.2f})")

            if callback is not None:
                if self.manhattan_filter is not None:
                    cb_pos = self.manhattan_filter.get_position_3x1()
                else:
                    cb_pos = (self.trajectory.positions[-1]
                              if self.trajectory.positions else None)
                callback(frame_idx, surface, tracked, pose, cb_pos)

            frame_idx += 1

        if verbose:
            print(f"\nDone. {frame_idx} frames processed.")
            print(f"  Successful poses: {n_success}")
            print(f"  Failed poses:     {n_fail}")
            if self.manhattan_filter is not None:
                mp = self.manhattan_filter.position
                print(f"  Manhattan pos:    ({mp[0]:.1f}, {mp[1]:.1f}, {mp[2]:.1f})")
                print(f"  Cardinal heading: {self.manhattan_filter.cardinal:.0f}deg")
            else:
                traj = self.trajectory.get_full_trajectory()
                print(f"  Final position:   "
                      f"({traj[-1,0]:.3f}, {traj[-1,1]:.3f}, {traj[-1,2]:.3f})")
                print(f"  Total heading:    "
                      f"{self.trajectory.cumulative_heading:.1f}deg")

        return self.trajectory
