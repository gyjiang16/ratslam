"""
Sparse Feature Tracker for time-surface frames.

Detects FAST corners and tracks them across consecutive frames
using pyramidal Lucas-Kanade optical flow with forward-backward
consistency validation.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrackedFeatures:
    """Result of tracking between two frames."""
    pts_prev: np.ndarray     # (M, 2) matched points in previous frame
    pts_curr: np.ndarray     # (M, 2) matched points in current frame
    num_tracked: int         # number of successfully tracked features
    num_detected: int        # number of features in current detection pool


class FeatureTracker:
    """
    Detects and tracks sparse features across event-camera time-surface frames.
    
    Pipeline per frame:
      1. If no features or too few remain → detect new corners
      2. Track existing features from prev → curr via LK
      3. Validate via forward-backward consistency check
      4. Return matched point pairs
    """

    def __init__(self,
                 max_corners: int = 200,
                 quality_level: float = 0.05,
                 min_distance: int = 10,
                 block_size: int = 5,
                 lk_win_size: Tuple[int, int] = (21, 21),
                 lk_max_level: int = 3,
                 fb_threshold: float = 1.0,
                 min_features: int = 15,
                 redetect_ratio: float = 0.4):

        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.lk_win_size = lk_win_size
        self.lk_max_level = lk_max_level
        self.fb_threshold = fb_threshold
        self.min_features = min_features
        self.redetect_ratio = redetect_ratio

        # LK parameters
        self.lk_params = dict(
            winSize=self.lk_win_size,
            maxLevel=self.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # State
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self.last_mean_flow: Optional[np.ndarray] = None

    def detect_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect corner features on a single grayscale frame.
        Returns (N, 1, 2) array of corner positions.
        """
        pts = cv2.goodFeaturesToTrack(
            frame,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size
        )
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return pts

    def process_frame(self, frame: np.ndarray) -> Optional[TrackedFeatures]:
        """
        Process one time-surface frame. Returns TrackedFeatures if tracking
        succeeded, None for the first frame or if tracking failed.
        
        Args:
            frame: uint8 grayscale time-surface image.
        """
        # First frame — just detect and store
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            self._prev_pts = self.detect_features(frame)
            self.last_mean_flow = None
            return None

        # Check if we need to re-detect
        if (self._prev_pts is None or
            len(self._prev_pts) < self.min_features or
            len(self._prev_pts) < int(self.max_corners * self.redetect_ratio)):
            self._prev_pts = self.detect_features(self._prev_frame)
            if len(self._prev_pts) < self.min_features:
                self._prev_frame = frame.copy()
                self.last_mean_flow = None
                return None

        # Forward tracking: prev → curr
        pts_curr, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_frame, frame, self._prev_pts, None, **self.lk_params
        )

        if pts_curr is None:
            self._prev_frame = frame.copy()
            self._prev_pts = self.detect_features(frame)
            self.last_mean_flow = None
            return None

        # Median flow from ALL forward-tracked points (available even when
        # the full pose pipeline fails later).  Used as heading fallback.
        fwd_ok = status_fwd.ravel().astype(bool)
        if np.sum(fwd_ok) >= 5:
            all_flow = (pts_curr.reshape(-1, 2)[fwd_ok]
                        - self._prev_pts.reshape(-1, 2)[fwd_ok])
            self.last_mean_flow = np.median(all_flow, axis=0)
        else:
            self.last_mean_flow = None

        # Backward tracking: curr → prev (for consistency check)
        pts_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            frame, self._prev_frame, pts_curr, None, **self.lk_params
        )

        # Forward-backward consistency
        status_fwd = status_fwd.ravel().astype(bool)
        status_bwd = status_bwd.ravel().astype(bool)

        fb_error = np.linalg.norm(
            self._prev_pts.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1
        )
        fb_valid = fb_error < self.fb_threshold

        good = status_fwd & status_bwd & fb_valid

        if np.sum(good) < self.min_features:
            self._prev_frame = frame.copy()
            self._prev_pts = self.detect_features(frame)
            return None

        prev_good = self._prev_pts.reshape(-1, 2)[good]
        curr_good = pts_curr.reshape(-1, 2)[good]

        result = TrackedFeatures(
            pts_prev=prev_good,
            pts_curr=curr_good,
            num_tracked=int(np.sum(good)),
            num_detected=len(self._prev_pts)
        )

        # Update state: keep only successfully tracked points
        self._prev_frame = frame.copy()
        self._prev_pts = curr_good.reshape(-1, 1, 2).astype(np.float32)

        return result

    def force_redetect(self):
        """Force feature re-detection on next frame (e.g., at turn boundaries)."""
        self._prev_pts = None
