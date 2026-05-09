import numpy as np

from ratslam._globals import (
    EXP_CORRECTION,
    EXP_DELTA_PC_THRESHOLD,
    EXP_LOOPS,
    PC_DIM_TH,
    PC_DIM_XY,
    clip_rad_180,
    min_delta,
    signed_delta_rad,
)


class Experience(object):
    def __init__(self, x_pc, y_pc, th_pc, x_m, y_m, facing_rad, view_cell):
        self.x_pc = x_pc
        self.y_pc = y_pc
        self.th_pc = th_pc
        self.x_m = x_m
        self.y_m = y_m
        self.facing_rad = facing_rad
        self.view_cell = view_cell
        self.links = []

    def link_to(self, target, accum_delta_x, accum_delta_y, accum_delta_facing):
        d = np.sqrt(accum_delta_x ** 2 + accum_delta_y ** 2)
        heading_rad = signed_delta_rad(self.facing_rad, np.arctan2(accum_delta_y, accum_delta_x))
        facing_rad = signed_delta_rad(self.facing_rad, accum_delta_facing)
        link = ExperienceLink(self, target, facing_rad, d, heading_rad)
        self.links.append(link)


class ExperienceLink(object):
    def __init__(self, parent, target, facing_rad, d, heading_rad):
        _ = parent
        self.target = target
        self.facing_rad = facing_rad
        self.d = d
        self.heading_rad = heading_rad


class ExperienceMap(object):
    """Experience map module with loop-closure and graph relaxation."""

    def __init__(
        self,
        exp_delta_pc_threshold=EXP_DELTA_PC_THRESHOLD,
        min_loop_interval_frames=30,
        verbose=False,
        rotation_loop_enabled=True,
        rotation_loop_threshold_deg=320.0,
        rotation_loop_distance_max=50.0,
        rotation_loop_min_interval_rad=None,
        rotation_loop_manhattan=True,
    ):
        self.size = 0
        self.exps = []
        self.current_exp = None
        self.current_view_cell = None
        self.accum_delta_x = 0.0
        self.accum_delta_y = 0.0
        self.accum_delta_facing = np.pi / 2
        self.history = []
        self.verbose = verbose
        self.exp_delta_pc_threshold = float(exp_delta_pc_threshold)
        self.min_loop_interval_frames = int(min_loop_interval_frames)
        self.last_loop_closure_frame = -10**9
        self.loop_closures = 0
        self.loop_closure_edges = []

        # Rotation-based forced loop closure (back to the first experience
        # when total signed rotation has accumulated a full turn AND the
        # current node is geometrically close enough to the origin).
        self.rotation_loop_enabled = bool(rotation_loop_enabled)
        self.rotation_loop_threshold_rad = np.deg2rad(float(rotation_loop_threshold_deg))
        self.rotation_loop_distance_max = float(rotation_loop_distance_max)
        if rotation_loop_min_interval_rad is None:
            self.rotation_loop_min_interval_rad = np.deg2rad(300.0)
        else:
            self.rotation_loop_min_interval_rad = float(rotation_loop_min_interval_rad)
        self.rotation_loop_manhattan = bool(rotation_loop_manhattan)
        self.first_exp = None
        self.total_rotation_rad = 0.0
        self.last_rotation_loop_total = 0.0
        self.rotation_loop_closures = 0

    def _create_exp(self, x_pc, y_pc, th_pc, view_cell):
        self.size += 1
        x_m = self.accum_delta_x
        y_m = self.accum_delta_y
        facing_rad = clip_rad_180(self.accum_delta_facing)
        if self.current_exp is not None:
            x_m += self.current_exp.x_m
            y_m += self.current_exp.y_m
        exp = Experience(x_pc, y_pc, th_pc, x_m, y_m, facing_rad, view_cell)
        if self.current_exp is not None:
            self.current_exp.link_to(exp, self.accum_delta_x, self.accum_delta_y, self.accum_delta_facing)
        self.exps.append(exp)
        view_cell.exps.append(exp)
        if self.first_exp is None:
            self.first_exp = exp
        return exp

    def __call__(self, view_cell, vtrans, vrot, x_pc, y_pc, th_pc, frame_idx=None):
        vtrans = 0.666 * np.floor(float(vtrans) / 0.666)
        self.accum_delta_facing = clip_rad_180(self.accum_delta_facing + float(vrot))
        self.accum_delta_x += vtrans * np.cos(self.accum_delta_facing)
        self.accum_delta_y += vtrans * np.sin(self.accum_delta_facing)

        # Track signed cumulative rotation (used by the rotation-based
        # forced loop closure). Continues across loop closures so that
        # every full turn after the last closure is counted independently.
        self.total_rotation_rad += float(vrot)

        if self.current_exp is None:
            delta_pc = 0.0
        else:
            delta_pc = np.sqrt(
                min_delta(self.current_exp.x_pc, x_pc, PC_DIM_XY) ** 2
                + min_delta(self.current_exp.y_pc, y_pc, PC_DIM_XY) ** 2
                + min_delta(self.current_exp.th_pc, th_pc, PC_DIM_TH) ** 2
            )

        adjust_map = False
        if len(view_cell.exps) == 0 or delta_pc > self.exp_delta_pc_threshold:
            exp = self._create_exp(x_pc, y_pc, th_pc, view_cell)
            self.current_exp = exp
            self.accum_delta_x = 0.0
            self.accum_delta_y = 0.0
            self.accum_delta_facing = self.current_exp.facing_rad
        elif view_cell != self.current_exp.view_cell:
            adjust_map = True
            matched_exp = None

            delta_pcs = []
            n_candidate_matches = 0
            for e in view_cell.exps:
                dpc = np.sqrt(
                    min_delta(e.x_pc, x_pc, PC_DIM_XY) ** 2
                    + min_delta(e.y_pc, y_pc, PC_DIM_XY) ** 2
                    + min_delta(e.th_pc, th_pc, PC_DIM_TH) ** 2
                )
                delta_pcs.append(dpc)
                if dpc < self.exp_delta_pc_threshold:
                    n_candidate_matches += 1

            if self.verbose and len(view_cell.exps) > 0:
                best_dpc = float(np.min(delta_pcs)) if delta_pcs else float("nan")
                print(
                    f"[FPE-CLOSURE] frame={frame_idx} vc={view_cell.id} "
                    f"reuses view-cell with {len(view_cell.exps)} prior exp(s); "
                    f"best dPC={best_dpc:.3f} thr={self.exp_delta_pc_threshold:.3f} "
                    f"candidates={n_candidate_matches}"
                )

            if n_candidate_matches > 1:
                if self.verbose:
                    print(f"[FPE-CLOSURE] frame={frame_idx} hash collision -> closure rejected")
            else:
                min_delta_id = int(np.argmin(delta_pcs))
                min_delta_val = float(delta_pcs[min_delta_id])
                if min_delta_val < self.exp_delta_pc_threshold:
                    matched_exp = view_cell.exps[min_delta_id]
                    link_exists = any(linked_exp == matched_exp for linked_exp in [l.target for l in self.current_exp.links])
                    if not link_exists:
                        frame_now = int(frame_idx) if frame_idx is not None else 10**9
                        allow_loop = (frame_now - self.last_loop_closure_frame) >= self.min_loop_interval_frames
                        if not allow_loop:
                            if self.verbose:
                                print(
                                    f"[FPE-CLOSURE] frame={frame_idx} candidate found "
                                    f"(dPC={min_delta_val:.3f}) but blocked by min-interval "
                                    f"({frame_now - self.last_loop_closure_frame} < {self.min_loop_interval_frames})"
                                )
                            matched_exp = None
                    if matched_exp is not None and not link_exists:
                        if self.verbose:
                            print(
                                f"[FPE-CLOSURE] frame={frame_idx} ACCEPTED loop closure "
                                f"current_exp={self.exps.index(self.current_exp)} -> "
                                f"matched_exp={self.exps.index(matched_exp)} "
                                f"dPC={min_delta_val:.3f}"
                            )
                        self.loop_closures += 1
                        if frame_idx is not None:
                            self.last_loop_closure_frame = int(frame_idx)
                        self.current_exp.link_to(
                            matched_exp, self.accum_delta_x, self.accum_delta_y, self.accum_delta_facing
                        )
                        self.loop_closure_edges.append((self.current_exp, matched_exp))
                if matched_exp is None:
                    matched_exp = self._create_exp(x_pc, y_pc, th_pc, view_cell)

                self.current_exp = matched_exp
                self.accum_delta_x = 0.0
                self.accum_delta_y = 0.0
                self.accum_delta_facing = self.current_exp.facing_rad

        # Rotation-based forced loop closure: when the cumulative signed
        # rotation since the last forced closure has reached ~360 deg AND
        # the current experience is geometrically near the first experience,
        # link back to the origin. For Manhattan-style environments we then
        # rebalance the segment lengths along each cardinal axis so the
        # trajectory closes with right angles and no curves; otherwise we
        # fall back to standard graph relaxation.
        rot_closure_kind = self._maybe_close_loop_by_rotation(frame_idx)
        if rot_closure_kind == "manhattan":
            self._manhattan_balance_close()
            self.history.append(self.current_exp)
            return True
        if rot_closure_kind == "relax":
            adjust_map = True

        self.history.append(self.current_exp)
        if not adjust_map:
            return False

        for _i in range(EXP_LOOPS):
            for e0 in self.exps:
                for l in e0.links:
                    e1 = l.target
                    cf = EXP_CORRECTION
                    lx = e0.x_m + l.d * np.cos(e0.facing_rad + l.heading_rad)
                    ly = e0.y_m + l.d * np.sin(e0.facing_rad + l.heading_rad)

                    e0.x_m = e0.x_m + (e1.x_m - lx) * cf
                    e0.y_m = e0.y_m + (e1.y_m - ly) * cf
                    e1.x_m = e1.x_m - (e1.x_m - lx) * cf
                    e1.y_m = e1.y_m - (e1.y_m - ly) * cf

                    df = signed_delta_rad(e0.facing_rad + l.facing_rad, e1.facing_rad)
                    e0.facing_rad = clip_rad_180(e0.facing_rad + df * cf)
                    e1.facing_rad = clip_rad_180(e1.facing_rad - df * cf)
        return True

    def _maybe_close_loop_by_rotation(self, frame_idx):
        """Trigger a loop closure to the first experience when a full
        turn has accumulated and we are still close to the origin.

        Returns one of:
          - None        : no closure was created
          - "manhattan" : closure created, caller should run Manhattan
                          rebalance (right-angle preserving)
          - "relax"     : closure created, caller should run standard
                          graph relaxation
        """
        if not self.rotation_loop_enabled:
            return None
        if self.first_exp is None or self.current_exp is None:
            return None
        if self.current_exp is self.first_exp:
            return None

        rot_since_last = self.total_rotation_rad - self.last_rotation_loop_total
        if abs(rot_since_last) < self.rotation_loop_threshold_rad:
            return None

        dx = self.first_exp.x_m - self.current_exp.x_m
        dy = self.first_exp.y_m - self.current_exp.y_m
        distance = float(np.sqrt(dx * dx + dy * dy))

        if distance > self.rotation_loop_distance_max:
            if self.verbose:
                print(
                    f"[ROT-CLOSURE] frame={frame_idx} rotation={np.degrees(rot_since_last):+.1f} "
                    f"deg reached threshold but distance to first_exp={distance:.2f} > "
                    f"max={self.rotation_loop_distance_max:.2f} -> rejected"
                )
            # Reset the rotation counter so we do not spam the rejection
            # message every frame; require another full turn before retry.
            self.last_rotation_loop_total += np.sign(rot_since_last) * self.rotation_loop_min_interval_rad
            return None

        already_linked = any(
            link.target is self.first_exp for link in self.current_exp.links
        )
        if already_linked:
            self.last_rotation_loop_total = self.total_rotation_rad
            return None

        frame_now = int(frame_idx) if frame_idx is not None else 10**9
        if (frame_now - self.last_loop_closure_frame) < self.min_loop_interval_frames:
            return None

        if self.verbose:
            print(
                f"[ROT-CLOSURE] frame={frame_idx} ACCEPTED forced closure: "
                f"rotation={np.degrees(rot_since_last):+.1f} deg, "
                f"distance_to_first={distance:.2f}, "
                f"current_exp={self.exps.index(self.current_exp)} -> first_exp=0, "
                f"mode={'manhattan' if self.rotation_loop_manhattan else 'relax'}"
            )

        # The whole point of this closure is to assert that the robot is
        # back at the origin. Pass zero relative displacement so geometry
        # encodes 'current_exp == first_exp'.
        self.current_exp.link_to(
            self.first_exp,
            0.0,
            0.0,
            self.first_exp.facing_rad - self.current_exp.facing_rad,
        )
        self.loop_closure_edges.append((self.current_exp, self.first_exp))
        self.loop_closures += 1
        self.rotation_loop_closures += 1
        self.last_loop_closure_frame = frame_now
        self.last_rotation_loop_total = self.total_rotation_rad
        return "manhattan" if self.rotation_loop_manhattan else "relax"

    def _manhattan_balance_close(self):
        """Rebalance experience positions along cardinal axes so the
        trajectory forms a closed Manhattan loop with right angles and
        no curves. Called once after a rotation-based forced closure has
        been registered, in lieu of standard graph relaxation.

        For each consecutive pair of experiences we snap the displacement
        to the cardinal axis closest to the segment's facing_rad, then
        scale all positive vs negative legs along each axis so that the
        total signed displacement returns to zero (i.e. the path closes
        back to the first experience).
        """
        n = len(self.exps)
        if n < 2 or self.first_exp is None:
            return

        # Step 1: derive cardinal-aligned per-step displacements.
        deltas = []
        for i in range(1, n):
            e_prev = self.exps[i - 1]
            e_curr = self.exps[i]
            f = e_curr.facing_rad
            cos_f, sin_f = float(np.cos(f)), float(np.sin(f))
            if abs(cos_f) >= abs(sin_f):
                ux = 1.0 if cos_f >= 0 else -1.0
                uy = 0.0
            else:
                ux = 0.0
                uy = 1.0 if sin_f >= 0 else -1.0
            dx = e_curr.x_m - e_prev.x_m
            dy = e_curr.y_m - e_prev.y_m
            length = ux * dx + uy * dy
            if length < 0:
                length = -length
                ux = -ux
                uy = -uy
            deltas.append((ux, uy, length))

        # Step 2: per-axis totals split by sign.
        pos_x = sum(l for ux, _uy, l in deltas if ux > 0.5)
        neg_x = sum(l for ux, _uy, l in deltas if ux < -0.5)
        pos_y = sum(l for _ux, uy, l in deltas if uy > 0.5)
        neg_y = sum(l for _ux, uy, l in deltas if uy < -0.5)

        # Step 3: balance scale factors. Target: net = 0 on each axis.
        if pos_x > 1e-6 and neg_x > 1e-6:
            avg_x = 0.5 * (pos_x + neg_x)
            scale_pos_x = avg_x / pos_x
            scale_neg_x = avg_x / neg_x
        else:
            scale_pos_x = scale_neg_x = 1.0
        if pos_y > 1e-6 and neg_y > 1e-6:
            avg_y = 0.5 * (pos_y + neg_y)
            scale_pos_y = avg_y / pos_y
            scale_neg_y = avg_y / neg_y
        else:
            scale_pos_y = scale_neg_y = 1.0

        if self.verbose:
            print(
                f"[ROT-CLOSURE] Manhattan rebalance: "
                f"X(+={pos_x:.2f}, -={neg_x:.2f} -> scale +={scale_pos_x:.3f}, -={scale_neg_x:.3f}); "
                f"Y(+={pos_y:.2f}, -={neg_y:.2f} -> scale +={scale_pos_y:.3f}, -={scale_neg_y:.3f})"
            )

        # Step 4: rewrite all experience positions cumulatively from
        # first_exp using the rescaled cardinal deltas.
        x = float(self.first_exp.x_m)
        y = float(self.first_exp.y_m)
        for i, (ux, uy, length) in enumerate(deltas):
            if ux > 0.5:
                length *= scale_pos_x
            elif ux < -0.5:
                length *= scale_neg_x
            if uy > 0.5:
                length *= scale_pos_y
            elif uy < -0.5:
                length *= scale_neg_y
            x += ux * length
            y += uy * length
            self.exps[i + 1].x_m = x
            self.exps[i + 1].y_m = y

        # Step 5: refresh chain link distances/headings to match new
        # positions so future relaxation passes do not undo the balance.
        for src in self.exps:
            for link in src.links:
                tgt = link.target
                ddx = tgt.x_m - src.x_m
                ddy = tgt.y_m - src.y_m
                link.d = float(np.sqrt(ddx * ddx + ddy * ddy))
                link.heading_rad = signed_delta_rad(
                    src.facing_rad, np.arctan2(ddy, ddx)
                )

    def get_positions(self):
        if not self.exps:
            return np.zeros((0, 2), dtype=np.float64)
        return np.array([[e.x_m, e.y_m] for e in self.exps], dtype=np.float64)

    def get_edges(self):
        edges = []
        for src in self.exps:
            for link in src.links:
                tgt = link.target
                edges.append((src.x_m, src.y_m, tgt.x_m, tgt.y_m))
        return np.array(edges, dtype=np.float64) if edges else np.zeros((0, 4), dtype=np.float64)

    def get_loop_closure_edges(self):
        edges = []
        for src, tgt in self.loop_closure_edges:
            edges.append((src.x_m, src.y_m, tgt.x_m, tgt.y_m))
        return np.array(edges, dtype=np.float64) if edges else np.zeros((0, 4), dtype=np.float64)
