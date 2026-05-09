"""
Pose Estimator via Essential Matrix Decomposition.

Given matched 2D point correspondences between two frames,
recovers the inter-frame rotation R and translation direction t
using the five-point algorithm with RANSAC.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PoseEstimate:
    """Result of pose estimation between two frames."""
    R: np.ndarray           # (3, 3) rotation matrix
    t: np.ndarray           # (3, 1) unit translation vector
    num_inliers: int        # RANSAC inliers
    inlier_ratio: float     # fraction of inliers
    heading_deg: float      # yaw angle change in degrees
    translation_norm: float # magnitude of translation (pre-scale)
    success: bool


class PoseEstimator:
    """
    Estimates inter-frame camera pose from sparse 2D-2D correspondences.

    Pipeline:
      1. Normalize points using camera intrinsics K
      2. Compute essential matrix E via five-point + RANSAC
      3. Decompose E → (R, t) via cheirality check
      4. Extract heading change θ and translation speed v
    """

    def __init__(self, K: np.ndarray,
                 ransac_threshold: float = 1.0,
                 ransac_confidence: float = 0.999,
                 min_inliers: int = 10,
                 default_scale: float = 1.0):
        self.K = K
        self.ransac_threshold = ransac_threshold
        self.ransac_confidence = ransac_confidence
        self.min_inliers = min_inliers
        self.default_scale = default_scale

    def estimate(self, pts_prev: np.ndarray, pts_curr: np.ndarray) -> PoseEstimate:
        """
        Estimate pose from 2D-2D correspondences.
        
        Args:
            pts_prev: (N, 2) points in previous frame.
            pts_curr: (N, 2) points in current frame.
            
        Returns:
            PoseEstimate with R, t, and diagnostics.
        """
        n = len(pts_prev)
        if n < 5:
            return self._fail()

        # Compute essential matrix
        E, mask = cv2.findEssentialMat(
            pts_prev, pts_curr, self.K,
            method=cv2.RANSAC,
            prob=self.ransac_confidence,
            threshold=self.ransac_threshold
        )

        if E is None or mask is None:
            return self._fail()

        inlier_mask = mask.ravel().astype(bool)
        num_inliers = int(np.sum(inlier_mask))

        if num_inliers < self.min_inliers:
            return self._fail()

        # Recover pose (R, t) from essential matrix
        # recoverPose also performs the cheirality check internally
        retval, R, t, pose_mask = cv2.recoverPose(
            E, pts_prev, pts_curr, self.K, mask=mask.copy()
        )

        if retval < self.min_inliers:
            return self._fail()

        # Extract heading (yaw around Y-axis) from rotation matrix
        # For forward-looking camera: R_y(θ) has R[0,2]=sinθ, R[0,0]=cosθ
        heading_rad = np.arctan2(R[0, 2], R[0, 0])
        heading_deg = np.degrees(heading_rad)

        # Translation magnitude (unit vector, needs external scale)
        t_norm = np.linalg.norm(t)

        return PoseEstimate(
            R=R,
            t=t,
            num_inliers=num_inliers,
            inlier_ratio=num_inliers / n,
            heading_deg=heading_deg,
            translation_norm=t_norm * self.default_scale,
            success=True
        )

    def _fail(self) -> PoseEstimate:
        return PoseEstimate(
            R=np.eye(3),
            t=np.zeros((3, 1)),
            num_inliers=0,
            inlier_ratio=0.0,
            heading_deg=0.0,
            translation_norm=0.0,
            success=False
        )
