"""
Configuration for Event-Based Sparse Feature Tracking Visual Odometry.

All tunable parameters in one place.
Defaults are set for the iniVation DVXplorer Mini (640x480).
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class CameraConfig:
    """
    Camera intrinsic parameters.

    DVXplorer Mini: 640x480 resolution, 9um pixel pitch.
    Sensor size: 5.76 mm x 4.32 mm (1/2.5" format).

    If you have calibrated your camera (e.g. via DV software or OpenCV),
    set fx, fy, cx, cy, and distortion coefficients directly.

    If you only know the lens focal length in mm, use from_lens_mm().
    """
    width: int = 640
    height: int = 480

    # --- Intrinsics (in pixels) ---
    # Default: approximate for DVXplorer Mini with ~4.5mm lens
    # REPLACE with your actual calibration values for best accuracy!
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 240.0

    # --- Distortion (OpenCV radial-tangential model) ---
    # [k1, k2, p1, p2, k3] — set from DV calibration output
    dist_coeffs: Optional[Tuple[float, ...]] = None

    @property
    def K(self) -> np.ndarray:
        """3x3 camera intrinsic matrix."""
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)

    @property
    def dist(self) -> Optional[np.ndarray]:
        """Distortion coefficients as numpy array, or None."""
        if self.dist_coeffs is None:
            return None
        return np.array(self.dist_coeffs, dtype=np.float64)

    @classmethod
    def from_lens_mm(cls, focal_length_mm: float,
                     width: int = 640, height: int = 480,
                     pixel_pitch_um: float = 9.0) -> "CameraConfig":
        """
        Create config from lens focal length (mm) and sensor specs.

        DVXplorer Mini pixel pitch = 9.0 um.
        f_pixels = f_mm / pixel_pitch_mm = f_mm / 0.009

        Example:
            CameraConfig.from_lens_mm(4.5)  # ~500 px focal length
            CameraConfig.from_lens_mm(6.0)  # ~667 px focal length
        """
        pixel_pitch_mm = pixel_pitch_um / 1000.0
        f_pixels = focal_length_mm / pixel_pitch_mm
        return cls(
            width=width,
            height=height,
            fx=f_pixels,
            fy=f_pixels,
            cx=width / 2.0,
            cy=height / 2.0,
        )

    @classmethod
    def from_calibration_file(cls, filepath: str,
                              width: int = 640,
                              height: int = 480) -> "CameraConfig":
        """
        Load intrinsics from a YAML/JSON calibration file.

        Supports formats from:
          - DV software (iniVation)
          - OpenCV calibration
          - Kalibr

        Expected YAML structure (OpenCV style):
            camera_matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            distortion_coefficients: [k1, k2, p1, p2, k3]
        """
        import json
        from pathlib import Path

        path = Path(filepath)

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(path) as f:
                    data = yaml.safe_load(f)
            except ImportError:
                # Fallback: simple line parsing for OpenCV YAML
                data = _parse_opencv_yaml_simple(path)
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported calibration file format: {path.suffix}")

        # Extract intrinsics
        if "camera_matrix" in data:
            cm = data["camera_matrix"]
            if isinstance(cm, dict) and "data" in cm:
                cm = cm["data"]  # OpenCV YAML format
            fx, fy, cx, cy = cm[0], cm[4], cm[2], cm[5]
        elif "fx" in data:
            fx, fy = data["fx"], data["fy"]
            cx, cy = data.get("cx", width / 2), data.get("cy", height / 2)
        else:
            raise ValueError("Cannot find camera_matrix or fx/fy in calibration file")

        # Extract distortion
        dist = None
        if "distortion_coefficients" in data:
            dc = data["distortion_coefficients"]
            if isinstance(dc, dict) and "data" in dc:
                dc = dc["data"]
            dist = tuple(dc)
        elif "dist_coeffs" in data:
            dist = tuple(data["dist_coeffs"])

        return cls(
            width=width, height=height,
            fx=float(fx), fy=float(fy),
            cx=float(cx), cy=float(cy),
            dist_coeffs=dist,
        )


def _parse_opencv_yaml_simple(path) -> dict:
    """Minimal parser for OpenCV YAML calibration files (no PyYAML needed)."""
    import re
    result = {}
    with open(path) as f:
        text = f.read()

    # Find camera_matrix data
    m = re.search(r'camera_matrix:.*?data:\s*\[([\d.,\s\-e+]+)\]', text, re.DOTALL)
    if m:
        result["camera_matrix"] = [float(x) for x in m.group(1).split(",")]

    # Find distortion_coefficients data
    m = re.search(r'distortion_coefficients:.*?data:\s*\[([\d.,\s\-e+]+)\]', text, re.DOTALL)
    if m:
        result["distortion_coefficients"] = [float(x) for x in m.group(1).split(",")]

    return result


@dataclass
class TimeSurfaceConfig:
    """
    Parameters for time-surface generation.

    dt_ms: Frame interval. 33ms → ~30Hz. For slow indoor motion, try 50ms.
    tau_ms: Decay constant. Larger → smoother but blurrier edges.
           For DVXplorer Mini at walking speed, 15-25ms works well.
    """
    dt_ms: float = 50.0
    tau_ms: float = 20.0


@dataclass
class FeatureConfig:
    """Feature detection and tracking parameters."""
    max_corners: int = 200
    quality_level: float = 0.05
    min_distance: int = 10
    block_size: int = 5
    lk_win_size: Tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    fb_threshold: float = 1.0
    min_features_for_pose: int = 15
    redetect_ratio: float = 0.4


@dataclass
class PoseConfig:
    """Essential matrix estimation parameters."""
    ransac_threshold: float = 1.0
    ransac_confidence: float = 0.999
    min_inliers: int = 10
    default_scale: float = 1.0


@dataclass
class VOConfig:
    """Top-level configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    time_surface: TimeSurfaceConfig = field(default_factory=TimeSurfaceConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
