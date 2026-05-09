"""
Event-Based Visual Odometry via Sparse Feature Tracking.

A lightweight, event-camera-only VO pipeline designed for
indoor navigation on resource-constrained platforms (Jetson Nano).
"""

from .config import VOConfig
from .vo import EventVO
