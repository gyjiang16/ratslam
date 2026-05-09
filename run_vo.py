#!/usr/bin/env python3
"""
Event-Based Visual Odometry — Main Entry Point.

Usage:
    python run_vo.py --input events.csv
    python run_vo.py --input events.csv --dt 50 --tau 30 --max-frames 500
    python run_vo.py --input events.npz --width 640 --height 480

The pipeline:
    CSV/NPZ/AEDAT4 → Time Surface (fixed timestep) → FAST corners
    → Lucas-Kanade tracking → Essential matrix → R, t → Trajectory
"""

import argparse
import os
import sys

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from event_vo.config import VOConfig, CameraConfig, TimeSurfaceConfig, FeatureConfig, PoseConfig
from event_vo.vo import EventVO
from event_vo.utils.visualization import (
    plot_trajectory_2d, plot_heading_profile, RealtimeVisualizer
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Event-Based Sparse Feature Tracking Visual Odometry"
    )
    # Input
    parser.add_argument("--input", "-i", required=True,
                        help="Path to event file (CSV, NPZ, or AEDAT4)")
    parser.add_argument("--output-dir", "-o", default="output",
                        help="Directory for output files")

    # Time surface
    parser.add_argument("--dt", type=float, default=33.0,
                        help="Frame timestep in ms (default: 33 → ~30 Hz)")
    parser.add_argument("--tau", type=float, default=20.0,
                        help="Time surface decay constant in ms")

    # Feature tracking
    parser.add_argument("--max-corners", type=int, default=200)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Process at most N frames (None = all)")

    # Pose
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Translation scale factor")

    # Flags
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip trajectory plotting")
    parser.add_argument("--realtime", action="store_true",
                        help="Show real-time event frame, tracking, and trajectory")
    parser.add_argument("--manhattan", action="store_true",
                        help="Apply Manhattan-world heading filter (90-deg turn snapping)")
    parser.add_argument("--turn-threshold", type=float, default=45.0,
                        help="Heading deviation (deg) before snapping to next 90-deg cardinal (default: 45)")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # Use fixed DVXplorer Mini camera configuration
    cam_cfg = CameraConfig()

    # Build config
    config = VOConfig(
        camera=cam_cfg,
        time_surface=TimeSurfaceConfig(
            dt_ms=args.dt, tau_ms=args.tau
        ),
        feature=FeatureConfig(
            max_corners=args.max_corners
        ),
        pose=PoseConfig(
            default_scale=args.scale
        )
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Real-time visualizer (optional)
    rt_vis = None
    if args.realtime:
        rt_vis = RealtimeVisualizer(cam_width=cam_cfg.width,
                                    cam_height=cam_cfg.height)

    def _realtime_callback(frame_idx, surface, tracked, pose, position):
        if rt_vis is not None:
            keep_going = rt_vis.update(frame_idx, surface, tracked,
                                       pose, position)
            if not keep_going:
                raise KeyboardInterrupt("User pressed 'q'")

    # Run VO
    vo = EventVO(config,
                 manhattan=args.manhattan,
                 turn_threshold_deg=args.turn_threshold)
    try:
        trajectory = vo.run(
            args.input,
            max_frames=args.max_frames,
            callback=_realtime_callback if rt_vis else None,
            verbose=not args.quiet
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
        trajectory = vo.trajectory
    finally:
        if rt_vis is not None:
            rt_vis.close()

    # Manhattan trajectory data (computed in real-time during VO)
    manhattan_pos = None
    if args.manhattan and vo.manhattan_filter is not None:
        manhattan_pos = np.array(vo.manhattan_filter.positions)

    # Save trajectory as numpy
    traj_path = os.path.join(args.output_dir, "trajectory.npy")
    out_traj = manhattan_pos if manhattan_pos is not None else trajectory.get_full_trajectory()
    np.save(traj_path, out_traj)
    print(f"Trajectory saved to {traj_path}")

    # Save odometry output (for RatSLAM)
    speeds, headings = trajectory.get_odometry_for_ratslam()
    odom_path = os.path.join(args.output_dir, "odometry.npz")
    np.savez(odom_path, speeds=speeds, headings=headings,
             timestamps=np.array(trajectory.timestamps))
    print(f"Odometry (v, theta) saved to {odom_path}")

    # Plot
    if not args.no_plot:
        plot_trajectory_2d(
            trajectory,
            output_path=os.path.join(args.output_dir, "trajectory_2d.png"),
            positions=manhattan_pos,
            title="Event VO Trajectory (Manhattan)" if manhattan_pos is not None
                  else "Event VO Trajectory (Top-Down)"
        )
        plot_heading_profile(
            trajectory,
            output_path=os.path.join(args.output_dir, "heading_profile.png")
        )


# Allow numpy import at module level for np.save
import numpy as np

if __name__ == "__main__":
    main()
