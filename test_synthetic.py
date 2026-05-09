#!/usr/bin/env python3
"""
Test the Event VO pipeline with synthetic event data.

Generates a CSV file simulating a camera moving forward and turning,
then runs the full pipeline to verify everything works.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_synthetic_events(output_path: str, n_events: int = 200000,
                               width: int = 640, height: int = 480,
                               duration_us: int = 5_000_000):
    """
    Generate synthetic events simulating a textured indoor scene
    with a camera moving forward and then turning.
    
    Creates edges from simulated wall features, door frames, etc.
    """
    rng = np.random.default_rng(42)

    # Generate feature points on walls (edges that generate events)
    n_features = 80
    feature_x = rng.integers(20, width - 20, size=n_features)
    feature_y = rng.integers(20, height - 20, size=n_features)

    events = []
    t = 0
    dt_per_event = duration_us / n_events

    for i in range(n_events):
        t = int(i * dt_per_event)

        # Simulate motion: features shift slightly each event
        # Phase 1 (first 60%): forward motion → radial expansion from center
        # Phase 2 (last 40%): turning → horizontal shift
        phase = t / duration_us

        if phase < 0.6:
            # Radial expansion from image center
            fidx = rng.integers(0, n_features)
            fx, fy = feature_x[fidx], feature_y[fidx]
            # Add radial displacement from center
            dx = (fx - width // 2) * 0.002 * rng.normal(1, 0.3)
            dy = (fy - height // 2) * 0.002 * rng.normal(1, 0.3)
            x = int(np.clip(fx + dx * (t / 1000), 0, width - 1))
            y = int(np.clip(fy + dy * (t / 1000), 0, height - 1))
        else:
            # Horizontal sweep (turning)
            fidx = rng.integers(0, n_features)
            fx, fy = feature_x[fidx], feature_y[fidx]
            shift = int((phase - 0.6) * 200)
            x = int(np.clip(fx - shift + rng.integers(-2, 3), 0, width - 1))
            y = int(np.clip(fy + rng.integers(-1, 2), 0, height - 1))

        polarity = rng.choice([-1, 1])

        # Add some noise events
        if rng.random() < 0.1:
            x = rng.integers(0, width)
            y = rng.integers(0, height)

        events.append([t, x, y, polarity])

    events = np.array(events, dtype=np.int64)
    # Sort by timestamp
    events = events[events[:, 0].argsort()]

    # Save as CSV
    header = "t,x,y,p"
    np.savetxt(output_path, events, delimiter=",", header=header,
               fmt="%d", comments="")

    print(f"Generated {len(events)} synthetic events → {output_path}")
    print(f"  Duration: {duration_us / 1e6:.1f} s")
    print(f"  Resolution: {width}x{height}")
    return output_path


def test_pipeline():
    """Run the full pipeline on synthetic data."""
    from event_vo.config import VOConfig, CameraConfig, TimeSurfaceConfig
    from event_vo.vo import EventVO
    from event_vo.utils.visualization import plot_trajectory_2d, plot_heading_profile

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate synthetic events
    csv_path = os.path.join(output_dir, "test_events.csv")
    generate_synthetic_events(csv_path)

    # Step 2: Configure and run VO
    config = VOConfig(
        camera=CameraConfig(width=640, height=480),
        time_surface=TimeSurfaceConfig(dt_ms=33.0, tau_ms=20.0),
    )

    vo = EventVO(config)
    trajectory = vo.run(csv_path, verbose=True)

    # Step 3: Save results
    traj_data = trajectory.get_full_trajectory()
    np.save(os.path.join(output_dir, "trajectory.npy"), traj_data)

    speeds, headings = trajectory.get_odometry_for_ratslam()
    np.savez(os.path.join(output_dir, "odometry.npz"),
             speeds=speeds, headings=headings,
             timestamps=np.array(trajectory.timestamps))

    # Step 4: Plot
    plot_trajectory_2d(trajectory,
                       output_path=os.path.join(output_dir, "trajectory_2d.png"))
    plot_heading_profile(trajectory,
                         output_path=os.path.join(output_dir, "heading_profile.png"))

    print("\n=== Test Complete ===")
    print(f"  Frames processed: {len(trajectory.timestamps)}")
    print(f"  Final position: {traj_data[-1]}")
    print(f"  Total heading: {trajectory.cumulative_heading:.1f}°")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    test_pipeline()
