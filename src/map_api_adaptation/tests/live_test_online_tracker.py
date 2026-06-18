"""
Live testing script for online_tracker output.
Visualizes point clouds and camera trajectories in real-time as the online tracker processes frames.

Usage:
    # Terminal 1: Start the online tracker
    python online_tracker.py \
        --cam_dirs data/cam0/ \
        --output_dir results_online/ \
        --device cuda \
        --test_name my_test \
        --i2p_weights checkpoints/slam3r_i2p_release.ckpt \
        --l2w_weights checkpoints/slam3r_l2w_release.ckpt

    # Terminal 2: Run this live visualizer
    python live_test_online_tracker.py \
        --output_dir results_online/my_test_cam0 \
        --conf_thres_l2w 12 \
        --refresh_interval 1.0
"""

import argparse
import json
import numpy as np
import torch
import open3d as o3d
import time
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm
import threading

from slam3r.utils.recon_utils import estimate_focal_knowing_depth, estimate_camera_pose
from slam3r.viz import render_frames


class LiveOnlineTrackerVisualizer:
    def __init__(self, output_dir, conf_thres_l2w=12, refresh_interval=1.0):
        self.output_dir = Path(output_dir)
        self.conf_thres_l2w = conf_thres_l2w
        self.refresh_interval = refresh_interval
        
        self.preds_dir = self.output_dir / 'preds'
        self.traj_file = self.output_dir / 'trajectories.json'
        
        self.vis = None
        self.current_frame_count = 0
        self.last_traj_update = 0
        self.final_ply = self.find_final_ply()
        
        print(f"Monitoring: {self.output_dir}")
        print(f"Predictions dir: {self.preds_dir}")
        print(f"Trajectory file: {self.traj_file}")
        print(f"Final pointcloud: {self.final_ply if self.final_ply is not None else 'None'}")
    
    def find_final_ply(self):
        candidates = sorted(self.output_dir.glob("*.ply"))
        return candidates[0] if candidates else None

    def load_trajectories(self):
        """Load trajectory data from JSON."""
        if not self.traj_file.exists():
            return None
        try:
            with open(self.traj_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def load_preds_incrementally(self):
        """Load available predictions from preds directory."""
        if not self.preds_dir.exists():
            return None
        
        try:
            # Check if full predictions are available
            local_pcds_path = self.preds_dir / 'local_pcds.npy'
            registered_pcds_path = self.preds_dir / 'registered_pcds.npy'
            local_confs_path = self.preds_dir / 'local_confs.npy'
            registered_confs_path = self.preds_dir / 'registered_confs.npy'
            rgb_imgs_path = self.preds_dir / 'input_imgs.npy'
            
            if not all([local_pcds_path.exists(), registered_pcds_path.exists(), 
                       local_confs_path.exists(), registered_confs_path.exists(),
                       rgb_imgs_path.exists()]):
                return None
            
            local_pcds = np.load(local_pcds_path)
            registered_pcds = np.load(registered_pcds_path)
            local_confs = np.load(local_confs_path)
            registered_confs = np.load(registered_confs_path)
            rgb_imgs = np.load(rgb_imgs_path)
            
            return {
                'local_pcds': local_pcds,
                'registered_pcds': registered_pcds,
                'local_confs': local_confs,
                'registered_confs': registered_confs,
                'rgb_imgs': rgb_imgs
            }
        except Exception as e:
            print(f"Error loading predictions: {e}")
            return None
    
    def load_frame_plys(self):
        """Load individual frame PLY files."""
        ply_files = sorted(glob(str(self.output_dir / 'frame_*.ply')))
        if not ply_files:
            return None
        
        pcds = []
        rgbs = []
        for ply_path in ply_files:
            try:
                pcd = o3d.io.read_point_cloud(ply_path)
                if len(pcd.points) > 0:
                    pcds.append(np.asarray(pcd.points))
                    if pcd.has_colors():
                        rgbs.append(np.asarray(pcd.colors))
                    else:
                        rgbs.append(np.ones((len(pcd.points), 3)) * 0.5)
            except:
                pass
        
        if pcds:
            return {'pcds': pcds, 'rgbs': rgbs}
        return None
    
    def load_final_cloud(self):
        if self.final_ply is None or not self.final_ply.exists():
            return None
        try:
            return o3d.io.read_point_cloud(str(self.final_ply))
        except Exception as e:
            print(f"Error loading final point cloud {self.final_ply}: {e}")
            return None
    
    def visualize_final_output(self):
        trajectories = self.load_trajectories()
        if not trajectories:
            print("No trajectories found yet")
            return False
        
        frames = list(trajectories.values())[1]['frames']
        positions = np.array([f['position'] for f in frames])
        if len(positions) < 2:
            print(f"Only {len(positions)} frames, need at least 2")
            return False
        
        pcd = self.load_final_cloud()
        if pcd is None:
            print("No final point cloud available")
            return False
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Online Tracker Final Output")
        self.vis.add_geometry(pcd)
        
        lines = [[i, i+1] for i in range(len(positions)-1)]
        colors = [[1, 0, 0] for _ in lines]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(line_set)
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(positions)
        point_cloud.paint_uniform_color([0, 1, 0])
        self.vis.add_geometry(point_cloud)
        
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(mesh_frame)
        
        print(f"Visualizing final output: {len(positions)} camera positions")
        self.vis.run()
        self.vis.destroy_window()
        return True
    
    def estimate_camera_poses(self, registered_pcds, input_imgs):
        """Estimate camera poses from registered point clouds."""
        num_views = registered_pcds.shape[0]
        principal_point = torch.tensor((224//2, 224//2))
        
        # Estimate intrinsics from first view
        focal = estimate_focal_knowing_depth(
            torch.tensor(registered_pcds[0:1]).float(),
            principal_point,
            focal_mode='weiszfeld'
        )
        
        mean_intrinsics = np.eye(3)
        mean_intrinsics[0, 0] = focal.item()
        mean_intrinsics[1, 1] = focal.item()
        mean_intrinsics[:2, 2] = principal_point.numpy()
        
        c2ws = []
        for i in tqdm(range(num_views), desc="Estimating camera poses"):
            registered_pcd = registered_pcds[i]
            
            # Apply coordinate transformation (flip y,z) to match visualize.py
            registered_pcd_transformed = registered_pcd.copy()
            registered_pcd_transformed[..., 1:] *= -1
            
            c2w, success = estimate_camera_pose(registered_pcd_transformed, mean_intrinsics)
            if success:
                c2ws.append(c2w)
            else:
                c2ws.append(np.eye(4))
        
        return np.stack(c2ws), mean_intrinsics
    
    def build_point_cloud(self, registered_pcds, registered_confs, rgb_imgs, 
                         conf_threshold=None, max_points=2000000):
        """Build a combined point cloud from all frames."""
        if conf_threshold is None:
            conf_threshold = self.conf_thres_l2w
        
        all_points = []
        all_colors = []
        
        num_frames = registered_pcds.shape[0]
        for i in tqdm(range(num_frames), desc="Building point cloud"):
            pcd = registered_pcds[i]  # (224, 224, 3)
            conf = registered_confs[i]  # (224, 224)
            img = rgb_imgs[i]  # (224, 224, 3)
            
            # Flatten and filter by confidence
            pcd_flat = pcd.reshape(-1, 3)
            conf_flat = conf.reshape(-1)
            img_flat = img.reshape(-1, 3)
            
            # Convert img to [0, 1] if needed
            if img_flat.max() > 1.0:
                img_flat = img_flat / 255.0
            
            mask = conf_flat > conf_threshold
            all_points.append(pcd_flat[mask])
            all_colors.append(img_flat[mask])
        
        points = np.vstack(all_points) if all_points else np.zeros((0, 3))
        colors = np.vstack(all_colors) if all_colors else np.zeros((0, 3))
        
        # Resample if too many points
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            colors = colors[idx]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def visualize_with_poses(self, registered_pcds, rgb_imgs, c2ws):
        """Visualize point cloud with camera poses."""
        pcd = self.build_point_cloud(
            registered_pcds, 
            np.ones_like(registered_pcds[..., 0]) * self.conf_thres_l2w,  # Simple confidence mask
            rgb_imgs
        )
        
        # Create Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Online Tracker Live Visualization")
        self.vis.add_geometry(pcd)
        
        # Add camera frustums
        for i, c2w in enumerate(c2ws[::5]):  # Show every 5th camera
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                intrinsic=o3d.camera.PinholeCameraIntrinsic(224, 224, 224/2, 224/2, 224/2, 224/2),
                extrinsic=np.linalg.inv(c2w),
                scale=0.1
            )
            frustum.paint_uniform_color([0.1 * (i % 10), 0.5, 0.9])
            self.vis.add_geometry(frustum)
        
        # Add trajectory
        trajectory_points = c2ws[:, :3, 3]
        lines = [[i, i+1] for i in range(len(trajectory_points)-1)]
        colors = [[1, 0, 0] for _ in lines]
        
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(trajectory_points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(line_set)
        
        # Render
        self.vis.run()
        self.vis.destroy_window()
    
    def visualize_from_trajectories(self):
        """Visualize using trajectories.json data."""
        trajectories = self.load_trajectories()
        if not trajectories:
            print("No trajectories found yet")
            return False
        
        # Extract camera positions from trajectories
        frames = list(trajectories.values())[1]['frames']  # Get first camera
        positions = np.array([f['position'] for f in frames])
        forwards = np.array([f['forward'] for f in frames])
        
        if len(positions) < 2:
            print(f"Only {len(positions)} frames, need at least 2")
            return False
        
        # Create trajectory visualization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Online Tracker Trajectory")
        
        # Add trajectory line
        lines = [[i, i+1] for i in range(len(positions)-1)]
        colors = [[1, 0, 0] for _ in lines]
        
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(line_set)
        
        # Add points
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(positions)
        point_cloud.paint_uniform_color([0, 1, 0])
        self.vis.add_geometry(point_cloud)
        
        # Add coordinate frame at origin
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(mesh_frame)
        
        print(f"Visualizing {len(positions)} camera positions")
        self.vis.run()
        self.vis.destroy_window()
        return True
    
    def run_live_mode(self):
        """Run live visualization with periodic updates."""
        print("\n=== Live Visualization Mode ===")
        print("Waiting for predictions or final output...")
        
        last_frame_count = 0
        while True:
            self.final_ply = self.find_final_ply()
            # Load prediction arrays if available
            preds = self.load_preds_incrementally()
            
            if preds and preds['registered_pcds'].shape[0] > last_frame_count + 5:
                last_frame_count = preds['registered_pcds'].shape[0]
                print(f"\nLoaded {last_frame_count} frames, visualizing...")
                
                try:
                    c2ws, intrinsics = self.estimate_camera_poses(
                        preds['registered_pcds'],
                        preds['rgb_imgs']
                    )
                    self.visualize_with_poses(
                        preds['registered_pcds'],
                        preds['rgb_imgs'],
                        c2ws
                    )
                    break
                except Exception as e:
                    print(f"Error during visualization: {e}")
                    time.sleep(self.refresh_interval)
            elif self.final_ply is not None and self.traj_file.exists():
                print(f"Found final point cloud {self.final_ply}, visualizing output.")
                if self.visualize_final_output():
                    break
                print("Final output found, but visualization failed. Retrying...")
                time.sleep(self.refresh_interval)
            else:
                if preds:
                    print(f"Loaded {preds['registered_pcds'].shape[0]} frames, waiting for more...")
                else:
                    print("Waiting for predictions or final pointcloud...")
                time.sleep(self.refresh_interval)


def main():
    parser = argparse.ArgumentParser(description="Live visualization for online_tracker output")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory from online_tracker (e.g., results_online/my_test_cam0)"
    )
    parser.add_argument(
        "--conf_thres_l2w", type=float, default=12,
        help="Confidence threshold for filtering L2W points"
    )
    parser.add_argument(
        "--refresh_interval", type=float, default=1.0,
        help="Interval in seconds to check for new data"
    )
    parser.add_argument(
        "--traj_only", action="store_true",
        help="Only visualize trajectory from trajectories.json (faster)"
    )
    
    args = parser.parse_args()
    
    vis = LiveOnlineTrackerVisualizer(
        args.output_dir,
        conf_thres_l2w=args.conf_thres_l2w,
        refresh_interval=args.refresh_interval
    )
    
    if args.traj_only:
        # Quick trajectory-only visualization
        if vis.visualize_from_trajectories():
            print("Done")
    else:
        # Full live visualization
        vis.run_live_mode()


if __name__ == "__main__":
    main()
