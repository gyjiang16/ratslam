# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2013 Renato de Pontes Pereira, renato.ppontes at gmail dot com
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import numpy as np
from ratslam._globals import VT_ACTIVE_DECAY, VT_GLOBAL_DECAY, VT_MATCH_THRESHOLD
from ratslam.fpe_encoder import FPEEncoder

class ViewCell(object):
    '''A single view cell.

    A ViewCell object is used to store the information of a single view cell.
    '''
    _ID = 0

    def __init__(self, template, x_pc, y_pc, th_pc):
        '''Initialize a ViewCell.

        :param template: a 1D numpy array with the cell template.
        :param x_pc: the x position relative to the pose cell.
        :param y_pc: the y position relative to the pose cell.
        :param th_pc: the th position relative to the pose cell.
        '''
        self.id = ViewCell._ID
        self.template = template
        self.x_pc = x_pc
        self.y_pc = y_pc
        self.th_pc = th_pc
        self.decay = VT_ACTIVE_DECAY
        self.first = True
        self.exps = []

        ViewCell._ID += 1

class ViewCells(object):
    """View Cell module using FPE scene descriptors."""

    def __init__(self, encoder, vt_match_threshold=VT_MATCH_THRESHOLD,
                 heading_tolerance_rad=np.pi / 2.0, theta_resol=2):
        self.size = 0
        self.cells = []
        self.prev_cell = None
        self.memory_access = 0
        self.memory_access_max = 0
        self.previous_matched = 0
        self.previous_matched_prev = 0
        self.encoder = encoder
        self.match_threshold = float(vt_match_threshold)
        self.heading_tolerance_rad = float(heading_tolerance_rad)
        self.theta_resol = int(theta_resol)

    def _create_template(self, time_surface):
        return self.encoder.encode(time_surface)

    def _score(self, template):
        """Compute FPE similarity with all stored view cells."""
        scores = []
        for cell in self.cells:
            cell.decay -= VT_GLOBAL_DECAY
            if cell.decay < 0:
                cell.decay = 0
            s = FPEEncoder.similarity(template, cell.template)
            scores.append(float(s))
        return scores

    def _heading_ok(self, cell, current_heading_rad):
        if current_heading_rad is None:
            return True
        cell_heading = float(cell.th_pc) * (np.pi / float(self.theta_resol))
        heading_diff = abs(current_heading_rad - cell_heading)
        if heading_diff > np.pi:
            heading_diff = 2 * np.pi - heading_diff
        return heading_diff <= self.heading_tolerance_rad

    def create_cell(self, template, x_pc, y_pc, th_pc):
        cell = ViewCell(template, x_pc, y_pc, th_pc)
        self.cells.append(cell)
        self.size += 1
        return cell

    def __call__(self, time_surface, x_pc, y_pc, th_pc, current_heading_rad=None):
        """Execute one visual template iteration."""
        template = self._create_template(time_surface)

        if not self.size:
            cell = self.create_cell(template, x_pc, y_pc, th_pc)
            self.prev_cell = cell
            return cell

        scores = np.asarray(self._score(template), dtype=np.float64)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < self.match_threshold:
            cell = self.create_cell(template, x_pc, y_pc, th_pc)
            self.prev_cell = cell
            return cell

        cell = self.cells[best_idx]
        if not self._heading_ok(cell, current_heading_rad):
            cell = self.create_cell(template, x_pc, y_pc, th_pc)
            self.prev_cell = cell
            return cell

        cell.decay += VT_ACTIVE_DECAY

        if self.prev_cell != cell:
            cell.first = False

        self.prev_cell = cell
        return cell
