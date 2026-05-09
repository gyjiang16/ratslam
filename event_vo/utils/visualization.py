"""
Visualization utilities for trajectory, time surfaces, and feature tracks.
"""

import numpy as np
import cv2
import os
from typing import Optional, List


def plot_trajectory_2d(trajectory, output_path: str = "trajectory_2d.png",
                       title: str = "Event VO Trajectory (Top-Down)",
                       positions: Optional[np.ndarray] = None):
    """
    Plot the 2D top-down trajectory (x-z plane).
    If *positions* (N,3) is given it overrides the trajectory object data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if positions is not None:
        x, z = positions[:, 0], positions[:, 2]
    else:
        x, z = trajectory.get_xy_trajectory()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(x, z, "b-", linewidth=1.0, alpha=0.8, label="Estimated")
    ax.plot(x[0], z[0], "go", markersize=10, label="Start")
    ax.plot(x[-1], z[-1], "rs", markersize=10, label="End")
    ax.set_xlabel("X (arbitrary units)")
    ax.set_ylabel("Z (arbitrary units)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Trajectory saved to {output_path}")


def plot_heading_profile(trajectory, output_path: str = "heading_profile.png"):
    """Plot cumulative heading over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ts = trajectory.timestamps
    headings = trajectory.headings[1:]  # skip initial zero
    speeds = trajectory.speeds[1:]

    axes[0].plot(ts, headings, "r-", linewidth=0.8)
    axes[0].set_ylabel("Cumulative heading (°)")
    axes[0].set_title("Heading and speed over time")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ts, speeds, "b-", linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel("Translation magnitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Heading profile saved to {output_path}")


class RealtimeVisualizer:
    """
    Real-time display of event time-surface frames with feature tracking
    annotations and a live 2D trajectory plot, rendered side by side.
    """

    TRAJ_SIZE = 480
    BG_COLOR = (30, 30, 30)
    GRID_COLOR = (60, 60, 60)
    PATH_COLOR = (255, 180, 0)
    START_COLOR = (0, 200, 0)
    CURR_COLOR = (0, 0, 255)
    ARROW_COLOR = (0, 255, 255)
    TEXT_COLOR = (200, 200, 200)

    def __init__(self, cam_width: int = 640, cam_height: int = 480,
                 window_name: str = "Event VO - Realtime"):
        self.cam_w = cam_width
        self.cam_h = cam_height
        self.traj_h = cam_height
        self.traj_w = self.TRAJ_SIZE
        self.win = window_name
        self.positions: List[np.ndarray] = []
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, cam_width + self.traj_w, cam_height)

    def _draw_event_frame(self, surface: np.ndarray,
                          tracked=None, pose=None) -> np.ndarray:
        vis = cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)

        if tracked is not None:
            for i in range(tracked.num_tracked):
                p0 = tuple(tracked.pts_prev[i].astype(int))
                p1 = tuple(tracked.pts_curr[i].astype(int))
                cv2.arrowedLine(vis, p0, p1, (0, 255, 0), 1, tipLength=0.3)
                cv2.circle(vis, p1, 3, (0, 200, 255), -1)

        inlier_str = ""
        if pose is not None:
            inlier_str = f"  inliers={pose.num_inliers}"
            status = "OK" if pose.success else "FAIL"
            color = (0, 255, 0) if pose.success else (0, 0, 255)
            cv2.putText(vis, status, (self.cam_w - 60, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        n_tracked = tracked.num_tracked if tracked else 0
        info = f"tracked={n_tracked}{inlier_str}"
        cv2.putText(vis, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)
        return vis

    def _draw_trajectory(self) -> np.ndarray:
        canvas = np.full((self.traj_h, self.traj_w, 3),
                         self.BG_COLOR, dtype=np.uint8)

        if len(self.positions) < 2:
            cv2.putText(canvas, "Waiting...", (10, self.traj_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 1)
            return canvas

        pts = np.array(self.positions)
        xs, zs = pts[:, 0], pts[:, 2]

        margin = 40
        draw_w = self.traj_w - 2 * margin
        draw_h = self.traj_h - 2 * margin

        x_range = xs.max() - xs.min()
        z_range = zs.max() - zs.min()
        data_range = max(x_range, z_range, 1e-6) * 1.2
        cx = (xs.max() + xs.min()) / 2.0
        cz = (zs.max() + zs.min()) / 2.0
        scale = min(draw_w, draw_h) / data_range

        def to_px(x, z):
            px = int(margin + draw_w / 2 + (x - cx) * scale)
            py = int(margin + draw_h / 2 - (z - cz) * scale)
            return (px, py)

        # Grid
        for gy in range(margin, self.traj_h - margin, 40):
            cv2.line(canvas, (margin, gy), (self.traj_w - margin, gy),
                     self.GRID_COLOR, 1)
        for gx in range(margin, self.traj_w - margin, 40):
            cv2.line(canvas, (gx, margin), (gx, self.traj_h - margin),
                     self.GRID_COLOR, 1)

        # Path
        for i in range(1, len(xs)):
            p0 = to_px(xs[i - 1], zs[i - 1])
            p1 = to_px(xs[i], zs[i])
            cv2.line(canvas, p0, p1, self.PATH_COLOR, 2)

        # Start / current markers
        start = to_px(xs[0], zs[0])
        curr = to_px(xs[-1], zs[-1])
        cv2.circle(canvas, start, 6, self.START_COLOR, -1)
        cv2.circle(canvas, curr, 6, self.CURR_COLOR, -1)

        # Heading arrow at current position
        if len(pts) >= 2:
            dx = xs[-1] - xs[-2]
            dz = zs[-1] - zs[-2]
            length = np.hypot(dx, dz)
            if length > 1e-8:
                dx, dz = dx / length, dz / length
                arrow_len = 20
                tip = (int(curr[0] + dx * arrow_len),
                       int(curr[1] - dz * arrow_len))
                cv2.arrowedLine(canvas, curr, tip,
                                self.ARROW_COLOR, 2, tipLength=0.4)

        # Labels
        cv2.putText(canvas, "Trajectory (top-down)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)
        pos = pts[-1]
        cv2.putText(canvas, f"x={pos[0]:+.1f} z={pos[2]:+.1f}",
                    (10, self.traj_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_COLOR, 1)
        return canvas

    def update(self, frame_idx: int, surface: np.ndarray,
               tracked=None, pose=None, position=None):
        if position is not None:
            self.positions.append(position.ravel().copy())

        event_vis = self._draw_event_frame(surface, tracked, pose)
        traj_vis = self._draw_trajectory()

        if event_vis.shape[0] != traj_vis.shape[0]:
            traj_vis = cv2.resize(traj_vis,
                                  (traj_vis.shape[1], event_vis.shape[0]))

        combined = cv2.hconcat([event_vis, traj_vis])
        cv2.putText(combined, f"Frame {frame_idx}",
                    (10, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1)
        cv2.imshow(self.win, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        if key == ord(' '):
            while True:
                k2 = cv2.waitKey(50) & 0xFF
                if k2 == ord(' ') or k2 == ord('q'):
                    break
        return True

    def close(self):
        cv2.destroyWindow(self.win)


def save_time_surface_video(event_file: str, output_path: str,
                            width: int = 640, height: int = 480,
                            dt_ms: float = 33.0, tau_ms: float = 20.0,
                            max_frames: int = 300, fps: int = 15):
    """Save first N time-surface frames as a video for debugging."""
    from ..utils import load_events, TimeSurfaceGeneratorFast

    events = load_events(event_file)
    gen = TimeSurfaceGeneratorFast(width, height, dt_ms * 1000, tau_ms * 1000)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)

    count = 0
    for surface, ts in gen.generate(events):
        writer.write(surface)
        count += 1
        if count >= max_frames:
            break

    writer.release()
    print(f"Time surface video saved to {output_path} ({count} frames)")
