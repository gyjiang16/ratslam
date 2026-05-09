import numpy as np

from ratslam._globals import (
    PC_DIM_XY,
    PC_DIM_TH,
    PC_E_XY_WRAP,
    PC_E_TH_WRAP,
    PC_W_E_DIM,
    PC_W_EXCITE,
    PC_AVG_XY_WRAP,
    POSECELL_VTRANS_SCALING,
)


class PoseCells(object):
    """2D pose-cell CAN with heading tracked by vrot accumulator."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.theta_resol = 2
        self.cells = np.zeros([PC_DIM_XY, PC_DIM_XY], dtype=np.float64)
        self.active = (PC_DIM_XY // 2, PC_DIM_XY // 2, self.theta_resol)
        self.cells[self.active[0], self.active[1]] = 1.0

        self.vtrans_acc = 0.0
        self.vrot_acc = 2 * (np.pi / 2)
        self.th_layer = np.zeros([2], dtype=np.float64)

    def posecell_quantization(self):
        self.cells = (1.0 / 16.0) * np.floor(self.cells / (1.0 / 16.0))

    def compute_activity_matrix(self, xywrap, thwrap, wdim, pcw):
        _ = thwrap
        pca_new = np.zeros([PC_DIM_XY, PC_DIM_XY], dtype=np.float64)
        indices = np.nonzero(self.cells)
        for i, j in zip(*indices):
            pca_new[np.ix_(xywrap[i:i + wdim], xywrap[j:j + wdim])] += self.cells[i, j] * pcw
        return pca_new

    def get_pc_max(self, xywrap):
        _ = xywrap
        pc_max_cells = (1.0 / 16.0) * np.floor(self.cells / (1.0 / 16.0))
        x, y = np.unravel_index(np.argmax(pc_max_cells), self.cells.shape)
        th = int(np.round(self.vrot_acc / (np.pi / self.theta_resol)))
        if th == 2 * self.theta_resol:
            th = 0
        return (int(x), int(y), int(th))

    def __call__(self, view_cell, vtrans, vrot):
        vtrans = float(vtrans) * POSECELL_VTRANS_SCALING
        vrot = float(vrot)

        self.vtrans_acc += vtrans
        self.vrot_acc += vrot

        if self.vtrans_acc > 1:
            vtrans = 1.0
            self.vtrans_acc -= 1.0
        else:
            vtrans = 0.0

        if self.vrot_acc >= 2 * np.pi:
            self.vrot_acc -= 2 * np.pi
        elif self.vrot_acc < 0:
            self.vrot_acc += 2 * np.pi

        if not view_cell.first:
            act_x = int(view_cell.x_pc) % PC_DIM_XY
            act_y = int(view_cell.y_pc) % PC_DIM_XY
            view_heading = float(view_cell.th_pc) * (np.pi / self.theta_resol)
            if self.vrot_acc - view_heading > np.pi:
                self.vrot_acc = 0.5 * self.vrot_acc + 0.5 * (view_heading + 2 * np.pi)
            elif self.vrot_acc - view_heading < -np.pi:
                self.vrot_acc = 0.5 * self.vrot_acc + 0.5 * (view_heading - 2 * np.pi)
            else:
                self.vrot_acc = 0.5 * self.vrot_acc + 0.5 * view_heading

            self.cells[self.cells < 0.2] = 0
            self.cells[self.cells >= 0.2] -= 0.2
            self.cells[act_x, act_y] = 1.0

        self.posecell_quantization()
        self.cells = self.compute_activity_matrix(PC_E_XY_WRAP, PC_E_TH_WRAP, PC_W_E_DIM, PC_W_EXCITE)
        self.cells[self.cells < 0.012 * 4] = 0
        self.cells[self.cells >= 0.012 * 4] -= 0.012 * 4
        self.cells[self.cells >= 1.0 / 32.0] += 0.35

        if vtrans == 1.0:
            for _dir_pc in range(PC_DIM_TH):
                q_heading = np.round(self.vrot_acc / (np.pi / self.theta_resol)) * (np.pi / self.theta_resol)
                self.th_layer[0] += np.cos(q_heading)
                self.th_layer[1] += np.sin(q_heading)

                if self.th_layer[0] >= 1:
                    self.cells[:, :] = np.roll(self.cells[:, :], 1, 1)
                    self.th_layer[0] -= 1
                elif self.th_layer[0] <= -1:
                    self.cells[:, :] = np.roll(self.cells[:, :], -1, 1)
                    self.th_layer[0] += 1

                if self.th_layer[1] >= 1:
                    self.cells[:, :] = np.roll(self.cells[:, :], 1, 0)
                    self.th_layer[1] -= 1
                elif self.th_layer[1] <= -1:
                    self.cells[:, :] = np.roll(self.cells[:, :], -1, 0)
                    self.th_layer[1] += 1

        self.active = self.get_pc_max(PC_AVG_XY_WRAP)
        return self.active
