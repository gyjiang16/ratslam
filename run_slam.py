#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from event_vo.config import VOConfig, CameraConfig, TimeSurfaceConfig, FeatureConfig, PoseConfig
from event_vo.utils.visualization import plot_trajectory_2d, plot_heading_profile
from event_vo.utils.jetson_energy import JetsonEnergySampler
from ratslam import EventNeuroSLAM


class SlamRealtimeVisualizer:
    def __init__(self, cam_width=640, cam_height=480, window_name="Event NeuroSLAM - Realtime"):
        self.cam_w = cam_width
        self.cam_h = cam_height
        self.map_w = 480
        self.win = window_name
        self.vo_positions = []
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, cam_width + self.map_w * 2, cam_height)

    def _draw_event(self, surface, tracked):
        vis = cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)
        if tracked is not None:
            for i in range(tracked.num_tracked):
                p0 = tuple(tracked.pts_prev[i].astype(int))
                p1 = tuple(tracked.pts_curr[i].astype(int))
                cv2.arrowedLine(vis, p0, p1, (0, 255, 0), 1, tipLength=0.25)
        return vis

    def _draw_polyline_map(self, points_2d, title, color):
        canvas = np.zeros((self.cam_h, self.map_w, 3), dtype=np.uint8) + 25
        if points_2d is None or len(points_2d) < 2:
            cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            return canvas

        pts = np.asarray(points_2d, dtype=np.float64)
        xs, ys = pts[:, 0], pts[:, 1]
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        span = max(max_x - min_x, max_y - min_y, 1e-6)
        scale = min(self.map_w - 40, self.cam_h - 40) / (span * 1.1)
        cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

        def to_px(x, y):
            px = int(self.map_w / 2 + (x - cx) * scale)
            py = int(self.cam_h / 2 - (y - cy) * scale)
            return px, py

        for i in range(1, len(pts)):
            cv2.line(canvas, to_px(xs[i - 1], ys[i - 1]), to_px(xs[i], ys[i]), color, 2)
        cv2.circle(canvas, to_px(xs[0], ys[0]), 5, (0, 255, 0), -1)
        cv2.circle(canvas, to_px(xs[-1], ys[-1]), 5, (0, 0, 255), -1)
        cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        return canvas

    def update(self, frame_idx, surface, tracked, vo_pos_3d, exp_map):
        if vo_pos_3d is not None:
            p = vo_pos_3d.ravel()
            self.vo_positions.append([float(p[0]), float(p[2])])
        exp_pos = exp_map.get_positions()
        event_panel = self._draw_event(surface, tracked)
        vo_panel = self._draw_polyline_map(self.vo_positions, "Raw VO (x-z)", (128, 128, 128))
        exp_panel = self._draw_polyline_map(exp_pos, "Experience Map (x-y)", (255, 180, 0))
        vis = cv2.hconcat([event_panel, vo_panel, exp_panel])
        cv2.putText(vis, f"Frame {frame_idx}", (10, self.cam_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.imshow(self.win, vis)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def close(self):
        cv2.destroyWindow(self.win)


def parse_args():
    parser = argparse.ArgumentParser(description="Event-only VO + RatSLAM backend")
    parser.add_argument("--input", "-i", required=True, help="Path to event file (CSV, NPZ, or AEDAT4)")
    parser.add_argument("--output-dir", "-o", default="output", help="Directory for output files")
    parser.add_argument("--dt", type=float, default=33.0, help="Frame timestep in ms")
    parser.add_argument("--tau", type=float, default=20.0, help="Time surface decay constant in ms")
    parser.add_argument("--max-corners", type=int, default=200)
    parser.add_argument("--max-frames", type=int, default=None, help="Process at most N frames")
    parser.add_argument("--scale", type=float, default=1.0, help="Translation scale factor")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting outputs")
    parser.add_argument("--realtime", action="store_true", help="Show real-time display")
    parser.add_argument("--turn-threshold", type=float, default=45.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--vt-threshold", type=float, default=0.65, help="FPE similarity threshold [0,1]")
    parser.add_argument("--exp-threshold", type=float, default=1.5, help="EXP_DELTA_PC_THRESHOLD")
    parser.add_argument("--downsample-factor", type=int, default=8, help="FPE downsample factor")
    parser.add_argument(
        "--min-loop-interval-frames",
        type=int,
        default=30,
        help="Minimum frame interval between two loop closures",
    )
    parser.add_argument(
        "--rotation-loop",
        dest="rotation_loop",
        action="store_true",
        default=True,
        help="Enable forced loop closure when total rotation reaches one full turn (default: on)",
    )
    parser.add_argument(
        "--no-rotation-loop",
        dest="rotation_loop",
        action="store_false",
        help="Disable the rotation-based forced loop closure mechanism",
    )
    parser.add_argument(
        "--rotation-loop-deg",
        type=float,
        default=320.0,
        help="Cumulative signed rotation (deg) required to trigger a forced closure",
    )
    parser.add_argument(
        "--rotation-loop-distance",
        type=float,
        default=50.0,
        help="Maximum distance from the first experience that still allows a forced closure",
    )
    parser.add_argument(
        "--rotation-loop-manhattan",
        dest="rotation_loop_manhattan",
        action="store_true",
        default=True,
        help="On rotation closure, rebalance segments along cardinal axes "
             "to keep right angles (default: on)",
    )
    parser.add_argument(
        "--no-rotation-loop-manhattan",
        dest="rotation_loop_manhattan",
        action="store_false",
        help="Use standard graph relaxation after rotation closure instead "
             "of Manhattan rebalancing (may produce curved corners)",
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling report")
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark preset for embedded boards (Jetson Orin Nano): enable "
             "--profile + energy sampling, disable --realtime / plotting / "
             "verbose; prints latency percentile and energy summary to stdout",
    )
    parser.add_argument(
        "--energy",
        action="store_true",
        help="Sample on-board power rails (Jetson INA3221) during the run "
             "and report average power and total energy",
    )
    parser.add_argument(
        "--energy-period-ms",
        type=float,
        default=100.0,
        help="Power-sampler period in milliseconds (default: 100)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bench:
        args.profile = True
        args.realtime = False
        args.no_plot = True
        args.quiet = True
        args.energy = True
    os.makedirs(args.output_dir, exist_ok=True)

    config = VOConfig(
        camera=CameraConfig(),
        time_surface=TimeSurfaceConfig(dt_ms=args.dt, tau_ms=args.tau),
        feature=FeatureConfig(max_corners=args.max_corners),
        pose=PoseConfig(default_scale=args.scale),
    )

    slam = EventNeuroSLAM(
        config=config,
        manhattan=True,
        turn_threshold_deg=args.turn_threshold,
        vt_threshold=args.vt_threshold,
        exp_threshold=args.exp_threshold,
        fpe_downsample_factor=args.downsample_factor,
        min_loop_interval_frames=args.min_loop_interval_frames,
        rotation_loop_enabled=args.rotation_loop,
        rotation_loop_threshold_deg=args.rotation_loop_deg,
        rotation_loop_distance_max=args.rotation_loop_distance,
        rotation_loop_manhattan=args.rotation_loop_manhattan,
        profile=args.profile,
        verbose=not args.quiet,
    )

    rt = SlamRealtimeVisualizer() if args.realtime else None

    def cb(frame_idx, surface, tracked, pose, position, exp_map):
        _ = pose
        if rt is not None:
            keep = rt.update(frame_idx, surface, tracked, position, exp_map)
            if not keep:
                raise KeyboardInterrupt("User stopped")

    energy_sampler = None
    if args.energy:
        energy_sampler = JetsonEnergySampler(period_s=args.energy_period_ms / 1000.0)
        if energy_sampler.available():
            if not args.quiet:
                print(f"[ENERGY] sampling INA3221 rails at {args.energy_period_ms:.0f} ms:")
                for path in energy_sampler.sensor_paths():
                    print(f"  - {path}")
            energy_sampler.start()
        else:
            if not args.quiet:
                print("[ENERGY] no Jetson INA3221 power sensors found - skipping energy sampling")
            energy_sampler = None

    try:
        result = slam.run(
            args.input,
            max_frames=args.max_frames,
            callback=cb,
            output_dir=args.output_dir,
        )
    except KeyboardInterrupt:
        result = {"trajectory": slam.vo.trajectory, "experience_map": slam.experience_map}
    finally:
        if rt is not None:
            rt.close()
        if energy_sampler is not None:
            stats = energy_sampler.stop()
            slam.profiler.attach_energy(stats, sensor_paths=energy_sampler.sensor_paths())

    traj = result["trajectory"]
    np.save(os.path.join(args.output_dir, "trajectory.npy"), traj.get_full_trajectory())
    speeds, headings = traj.get_odometry_for_ratslam()
    np.savez(
        os.path.join(args.output_dir, "odometry.npz"),
        speeds=speeds,
        headings=headings,
        timestamps=np.array(traj.timestamps),
    )

    if not args.no_plot:
        plot_trajectory_2d(traj, output_path=os.path.join(args.output_dir, "trajectory_2d.png"))
        plot_heading_profile(traj, output_path=os.path.join(args.output_dir, "heading_profile.png"))

    print(f"Saved outputs to {args.output_dir}")
    print(f"View cells: {slam.view_cells.size}")
    print(f"Experience nodes: {slam.experience_map.size}")
    print(f"Loop closures: {slam.experience_map.loop_closures}")
    print(f"  - FPE/pose-cell based: {slam.experience_map.loop_closures - slam.experience_map.rotation_loop_closures}")
    print(f"  - Rotation forced    : {slam.experience_map.rotation_loop_closures}")
    print(f"Total cumulative rotation: {np.degrees(slam.experience_map.total_rotation_rad):+.1f} deg")
    if args.profile:
        report_path = os.path.join(args.output_dir, 'profiling_report.txt')
        print()
        slam.profiler.print_summary(
            n_view_cells=slam.view_cells.size,
            n_experiences=slam.experience_map.size,
            n_loop_closures=slam.experience_map.loop_closures,
        )
        print()
        print(f"Profiling report: {report_path}")
        print(f"Per-frame CSV   : {os.path.splitext(report_path)[0]}_per_frame.csv")
    elif energy_sampler is not None and slam.profiler.energy_stats is not None:
        es = slam.profiler.energy_stats
        print()
        print("Power / energy (Jetson INA3221):")
        print(f"  samples         : {es.samples}")
        print(f"  measured for    : {es.duration_s:.3f} s")
        if es.avg_power_w is not None:
            print(f"  average power   : {es.avg_power_w:.3f} W")
            print(f"  peak power      : {es.peak_power_w:.3f} W")
            print(f"  total energy    : {es.energy_j:.3f} J")


if __name__ == "__main__":
    main()
