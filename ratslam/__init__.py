import os
import numpy as np
import cv2

from event_vo.config import VOConfig
from event_vo.vo.pipeline import EventVO
from event_vo.vo.trajectory import ManhattanFilter
from event_vo.utils.event_loader import load_events

from ratslam.view_cells import ViewCells
from ratslam.fpe_encoder import FPEEncoder
from ratslam.pose_cells import PoseCells
from ratslam.experience_map import ExperienceMap
from ratslam.profiling import FrameProfiler


class EventNeuroSLAM(object):
    """Event-only VO + RatSLAM backend integration."""

    def __init__(self, config=None, manhattan=False, turn_threshold_deg=45.0,
                 vt_threshold=0.65, exp_threshold=1.5, fpe_downsample_factor=8,
                 min_loop_interval_frames=30, profile=False, verbose=True,
                 rotation_loop_enabled=True,
                 rotation_loop_threshold_deg=320.0,
                 rotation_loop_distance_max=50.0,
                 rotation_loop_manhattan=True):
        self.config = config or VOConfig()
        self.verbose = verbose
        self.profile = profile

        self.vo = EventVO(self.config, manhattan=manhattan, turn_threshold_deg=turn_threshold_deg)
        self.fpe_encoder = FPEEncoder(
            frame_height=self.config.camera.height,
            frame_width=self.config.camera.width,
            downsample_factor=fpe_downsample_factor,
        )
        self.view_cells = ViewCells(
            encoder=self.fpe_encoder,
            vt_match_threshold=vt_threshold,
            heading_tolerance_rad=np.pi / 2.0,
            theta_resol=2,
        )
        self.pose_cells = PoseCells(verbose=verbose)
        self.experience_map = ExperienceMap(
            exp_delta_pc_threshold=exp_threshold,
            min_loop_interval_frames=min_loop_interval_frames,
            verbose=verbose,
            rotation_loop_enabled=rotation_loop_enabled,
            rotation_loop_threshold_deg=rotation_loop_threshold_deg,
            rotation_loop_distance_max=rotation_loop_distance_max,
            rotation_loop_manhattan=rotation_loop_manhattan,
        )
        self.profiler = FrameProfiler(enabled=profile)

    def run(self, event_file, max_frames=None, callback=None, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        events = load_events(event_file)
        if self.verbose:
            print(f"Loading events from {event_file}...")
            print(f"  Events: {len(events):,}")

        frame_idx = 0
        fx = self.config.camera.fx
        scale = self.config.pose.default_scale
        prev_surface_u8 = None
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 6))
        vo = self.vo
        if vo.manhattan and vo.manhattan_filter is None:
            vo.manhattan_filter = ManhattanFilter(turn_threshold_deg=vo.turn_threshold_deg)

        for surface, timestamp in vo.ts_generator.generate(events):
            if max_frames is not None and frame_idx >= max_frames:
                break
            self.profiler.start_frame()

            self.profiler.tic("module_a")
            if vo._undistort_map is not None:
                surface = cv2.remap(surface, vo._undistort_map[0], vo._undistort_map[1], cv2.INTER_LINEAR)
            surface = cv2.GaussianBlur(surface, (3, 3), 1.0)
            self.profiler.toc("module_a")

            self.profiler.tic("module_b")
            tracked = vo.tracker.process_frame(surface)
            pose = None
            frame_speed = 0.0
            vrot = 0.0
            if tracked is not None:
                pose = vo.pose_estimator.estimate(tracked.pts_prev, tracked.pts_curr)
                vo.trajectory.update(pose, timestamp, scale=scale)
                if pose.success:
                    frame_speed = float(pose.translation_norm)
                    vrot = np.deg2rad(pose.heading_deg)
                else:
                    frame_speed = 0.0
            else:
                frame_speed = 0.0
            self.profiler.toc("module_b")

            self.profiler.tic("module_c")
            heading_deg = 0.0
            mf = vo.tracker.last_mean_flow
            if pose is not None and pose.success:
                heading_deg = pose.heading_deg
            elif mf is not None and abs(mf[0]) > 0.3:
                heading_deg = np.degrees(-mf[0] / fx)
                vrot = np.deg2rad(heading_deg)

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

            if vo.manhattan_filter is not None:
                prev_cardinal = float(vo.manhattan_filter.cardinal)
                vo.manhattan_filter.update(similarity, heading_deg, frame_speed)
                curr_cardinal = float(vo.manhattan_filter.cardinal)
                heading_for_vt = np.deg2rad(curr_cardinal)
                d_cardinal = ((curr_cardinal - prev_cardinal + 180.0) % 360.0) - 180.0
                vtrans_slam = frame_speed
                vrot_slam = np.deg2rad(d_cardinal)
            else:
                heading_for_vt = np.deg2rad(vo.trajectory.cumulative_heading)
                vtrans_slam = frame_speed
                vrot_slam = vrot
            prev_surface_u8 = surface.copy()
            self.profiler.toc("module_c")

            self.profiler.tic("module_d")
            x_pc, y_pc, th_pc = self.pose_cells.active
            view_cell = self.view_cells(surface, x_pc, y_pc, th_pc, current_heading_rad=heading_for_vt)
            x_pc, y_pc, th_pc = self.pose_cells(view_cell, vtrans_slam, vrot_slam)
            self.experience_map(view_cell, vtrans_slam, vrot_slam, x_pc, y_pc, th_pc, frame_idx=frame_idx)
            self.profiler.toc("module_d")
            self.profiler.end_frame()

            if callback is not None:
                callback(frame_idx, surface, tracked, pose, vo.trajectory.positions[-1], self.experience_map)

            frame_idx += 1

        self._save_outputs(output_dir)
        self.profiler.write_report(
            os.path.join(output_dir, "profiling_report.txt"),
            n_view_cells=self.view_cells.size,
            n_experiences=self.experience_map.size,
            n_loop_closures=self.experience_map.loop_closures,
        )
        return {
            "trajectory": vo.trajectory,
            "experience_map": self.experience_map,
            "view_cells": self.view_cells,
            "num_frames": frame_idx,
        }

    def _save_outputs(self, output_dir):
        exp_positions = self.experience_map.get_positions()
        exp_edges = self.experience_map.get_edges()
        loop_edges = self.experience_map.get_loop_closure_edges()
        np.save(os.path.join(output_dir, "experience_map_nodes.npy"), exp_positions)
        np.save(os.path.join(output_dir, "experience_map_edges.npy"), exp_edges)
        np.save(os.path.join(output_dir, "experience_map_loop_edges.npy"), loop_edges)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))
            chain_label_used = False
            for e in exp_edges:
                ax.plot(
                    [e[0], e[2]],
                    [e[1], e[3]],
                    color="gray",
                    linewidth=0.7,
                    alpha=0.7,
                    label=None if chain_label_used else "Chain edge",
                )
                chain_label_used = True
            loop_label_used = False
            for e in loop_edges:
                ax.plot(
                    [e[0], e[2]],
                    [e[1], e[3]],
                    color="red",
                    linewidth=1.4,
                    alpha=0.9,
                    label=None if loop_label_used else "Loop closure",
                )
                loop_label_used = True
            if len(exp_positions) > 0:
                ax.scatter(
                    exp_positions[:, 0],
                    exp_positions[:, 1],
                    s=12,
                    c="tab:blue",
                    label="Experience nodes",
                )
                ax.scatter(
                    exp_positions[0, 0],
                    exp_positions[0, 1],
                    s=80,
                    c="green",
                    marker="o",
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=5,
                    label="Start (first_exp)",
                )
                ax.scatter(
                    exp_positions[-1, 0],
                    exp_positions[-1, 1],
                    s=80,
                    c="red",
                    marker="x",
                    linewidths=2.0,
                    zorder=5,
                    label="End (last_exp)",
                )
            ax.set_title("Experience Map (2D)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.axis("equal")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "experience_map_2d.png"), dpi=150)
            plt.close(fig)
        except Exception:
            pass
