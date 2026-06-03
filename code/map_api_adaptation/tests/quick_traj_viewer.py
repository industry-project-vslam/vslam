"""
Quick trajectory visualizer for online_tracker.
Shows camera positions and trajectory from trajectories.json in real-time.

Much faster than full point cloud visualization - useful for quick feedback during testing.

Usage:
    # Terminal 1: Start online tracker
    python online_tracker.py --cam_dirs data/cam0/ --output_dir results_online/ --device cuda --test_name my_test ...

    # Terminal 2: Watch trajectory in real-time
    python quick_traj_viewer.py --traj_file results_online/my_test_cam0/trajectories.json --watch
"""

import argparse
import json
import numpy as np
import open3d as o3d
import time
from pathlib import Path


class QuickTrajectoryViewer:
    def __init__(self, traj_file, watch_mode=False):
        self.traj_file = Path(traj_file)
        self.watch_mode = watch_mode
        self.vis = None
    
    def load_traj(self):
        """Load trajectory from JSON."""
        try:
            with open(self.traj_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def extract_positions(self, trajectories):
        """Extract camera positions from trajectory data."""
        if not trajectories:
            return None, None
        
        # Get first camera
        cam_key = [k for k in trajectories.keys() if k != 'metadata'][0] if 'metadata' in trajectories else list(trajectories.keys())[0]
        frames = trajectories[cam_key]['frames']
        
        positions = np.array([f['position'] for f in frames])
        confs = np.array([f.get('conf', 1.0) for f in frames])
        
        return positions, confs
    
    def visualize_once(self, trajectories):
        """Single visualization of current trajectory."""
        positions, confs = self.extract_positions(trajectories)
        
        if positions is None or len(positions) < 2:
            print(f"Not enough frames: {len(positions) if positions is not None else 0}")
            return
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Trajectory ({len(positions)} frames)")
        
        # Trajectory line
        lines = [[i, i+1] for i in range(len(positions)-1)]
        colors = [[1, 0, 0] for _ in lines]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)
        
        # Points colored by confidence
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(positions)
        
        # Color by confidence (green=high, red=low)
        norm_confs = (confs - confs.min()) / (confs.max() - confs.min() + 1e-6)
        colors_conf = np.stack([1-norm_confs, norm_confs, np.zeros_like(norm_confs)], axis=1)
        point_cloud.colors = o3d.utility.Vector3dVector(colors_conf)
        vis.add_geometry(point_cloud)
        
        # Coordinate frame
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        vis.add_geometry(mesh_frame)
        
        vis.run()
        vis.destroy_window()
    
    def visualize_live(self):
        """Continuous visualization with updates."""
        print(f"Watching {self.traj_file} for updates...")
        print("Press Ctrl+C to exit")
        
        last_frame_count = 0
        try:
            while True:
                traj = self.load_traj()
                if traj:
                    positions, confs = self.extract_positions(traj)
                    frame_count = len(positions) if positions is not None else 0
                    
                    if frame_count > last_frame_count and frame_count >= 10:
                        print(f"\n📍 {frame_count} frames - visualizing...")
                        print(f"   Mean confidence: {confs.mean():.2f}")
                        print(f"   Travel distance: {np.linalg.norm(np.diff(positions, axis=0)).sum():.2f}m")
                        
                        self.visualize_once(traj)
                        last_frame_count = frame_count
                    else:
                        if frame_count > 0:
                            print(f"⏳ {frame_count} frames collected ({frame_count}/10 min for viz)...", end='\r')
                        time.sleep(1.0)
                else:
                    print("Waiting for trajectories.json...", end='\r')
                    time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n👋 Exit")


def main():
    parser = argparse.ArgumentParser(description="Quick trajectory viewer for online_tracker")
    parser.add_argument(
        "--traj_file", type=str, required=True,
        help="Path to trajectories.json (e.g., results_online/my_test_cam0/trajectories.json)"
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Watch file for updates and re-visualize periodically"
    )
    
    args = parser.parse_args()
    
    viewer = QuickTrajectoryViewer(args.traj_file, watch_mode=args.watch)
    
    if args.watch:
        viewer.visualize_live()
    else:
        traj = viewer.load_traj()
        if traj:
            viewer.visualize_once(traj)
        else:
            print(f"Could not load {args.traj_file}")


if __name__ == "__main__":
    main()
