"""
Example client for Multi-Drone SLAM Tracker API.

Demonstrates how to:
1. Submit frames for processing
2. Retrieve drone status
3. Download point clouds
4. Clear drone data
5. Monitor system status
"""

import requests
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import time

class TrackerAPIClient:
    """Client for interacting with Multi-Drone SLAM Tracker API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if API is running."""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            return response.status_code == 200
        except:
            return False
    
    def process_frame(self, drone_id: str, frame_idx: int,
                     registered_pcd: np.ndarray,
                     registered_conf: np.ndarray,
                     rgb_img: np.ndarray) -> Dict:
        """
        Submit a frame for processing.
        
        Args:
            drone_id: Unique drone identifier
            frame_idx: Frame index
            registered_pcd: (H, W, 3) point cloud
            registered_conf: (H, W) confidence map
            rgb_img: (H, W, 3) RGB image
            
        Returns:
            Response with drone location and depth matrix
        """
        payload = {
            "drone_id": drone_id,
            "frame_idx": frame_idx,
            "registered_pcd": registered_pcd.tolist(),
            "registered_conf": registered_conf.tolist(),
            "rgb_img": rgb_img.tolist()
        }
        
        response = self.session.post(
            f"{self.base_url}/api/process_frame",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_pointcloud(self, drone_id: str, conf_threshold: float = 12.0,
                       format: str = "npy", save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Download point cloud for a drone.
        
        Args:
            drone_id: Unique drone identifier
            conf_threshold: Confidence threshold
            format: "npy" or "ply"
            save_path: Optional path to save the file
            
        Returns:
            (points, colors) as numpy arrays if format is "npy"
        """
        response = self.session.get(
            f"{self.base_url}/api/pointcloud/{drone_id}",
            params={"conf_threshold": conf_threshold, "format": format}
        )
        response.raise_for_status()
        
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Saved to {save_path}")
        
        if format.lower() == "npy":
            buffer = np.load(__import__('io').BytesIO(response.content))
            return buffer['points'], buffer['colors']
        
        return None, None
    
    def get_drone_status(self, drone_id: str) -> Dict:
        """Get status of a specific drone."""
        response = self.session.get(f"{self.base_url}/api/drones/{drone_id}")
        response.raise_for_status()
        return response.json()
    
    def get_all_status(self) -> Dict:
        """Get status of all drones."""
        response = self.session.get(f"{self.base_url}/api/status")
        response.raise_for_status()
        return response.json()
    
    def clear_drone(self, drone_id: str) -> Dict:
        """Clear all data for a drone."""
        response = self.session.delete(f"{self.base_url}/api/drones/{drone_id}")
        response.raise_for_status()
        return response.json()
    
    def compute_depth_matrix(self, registered_pcd: np.ndarray) -> np.ndarray:
        """
        Compute depth matrix without storing frame data.
        
        Args:
            registered_pcd: (H, W, 3) point cloud
            
        Returns:
            8x8 depth matrix
        """
        payload = {
            "drone_id": "temp",
            "frame_idx": 0,
            "registered_pcd": registered_pcd.tolist(),
            "registered_conf": np.ones(registered_pcd.shape[:2]).tolist(),
            "rgb_img": np.ones(registered_pcd.shape).tolist()
        }
        
        response = self.session.post(
            f"{self.base_url}/api/depth_matrix",
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        return np.array(result['depth_matrix'])
    
    def get_info(self) -> Dict:
        """Get API information and available endpoints."""
        response = self.session.get(f"{self.base_url}/api/info")
        response.raise_for_status()
        return response.json()


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of using the API client."""
    client = TrackerAPIClient("http://localhost:8000")
    
    # Check if API is running
    if not client.health_check():
        print("Error: API server is not running!")
        print("Start it with: uvicorn tracker_api:app --host 0.0.0.0 --port 8000")
        return
    
    print("✓ API is running")
    
    # Get API info
    info = client.get_info()
    print(f"\nAPI Version: {info['version']}")
    print(f"Available endpoints: {len(info['endpoints'])}")
    
    # Create dummy frame data
    h, w = 224, 224
    dummy_pcd = np.random.rand(h, w, 3) * 10  # Random point cloud
    dummy_conf = np.random.rand(h, w) * 20  # Random confidence
    dummy_img = np.random.rand(h, w, 3) * 255  # Random RGB image
    
    # Process frame for drone_1
    print("\nProcessing frame for drone_1...")
    result = client.process_frame(
        drone_id="drone_1",
        frame_idx=0,
        registered_pcd=dummy_pcd,
        registered_conf=dummy_conf,
        rgb_img=dummy_img
    )
    print(f"  Location: {result['location']}")
    print(f"  Depth matrix shape: {np.array(result['depth_matrix']).shape}")
    print(f"  Frame count: {result['frame_count']}")
    
    # Process another frame
    print("\nProcessing frame for drone_1 (frame 1)...")
    result = client.process_frame(
        drone_id="drone_1",
        frame_idx=1,
        registered_pcd=dummy_pcd + np.random.rand(h, w, 3) * 0.1,
        registered_conf=dummy_conf,
        rgb_img=dummy_img
    )
    print(f"  Location: {result['location']}")
    print(f"  Frame count: {result['frame_count']}")
    
    # Process frame for drone_2
    print("\nProcessing frame for drone_2...")
    result = client.process_frame(
        drone_id="drone_2",
        frame_idx=0,
        registered_pcd=dummy_pcd * 0.8,
        registered_conf=dummy_conf,
        rgb_img=dummy_img
    )
    print(f"  Location: {result['location']}")
    
    # Get status
    print("\nGetting system status...")
    status = client.get_all_status()
    print(f"  Total drones: {status['total_drones']}")
    for drone_id, drone_status in status['drones'].items():
        print(f"  - {drone_id}: {drone_status['frame_count']} frames")
    
    # Get specific drone status
    print("\nGetting drone_1 status...")
    drone_status = client.get_drone_status("drone_1")
    print(f"  Status: {json.dumps(drone_status['status'], indent=2)}")
    
    # Download point cloud
    print("\nDownloading point cloud for drone_1...")
    try:
        points, colors = client.get_pointcloud(
            "drone_1",
            save_path="pointcloud_drone_1.npz"
        )
        print(f"  Points shape: {points.shape}")
        print(f"  Colors shape: {colors.shape}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Compute depth matrix
    print("\nComputing depth matrix...")
    depth_matrix = client.compute_depth_matrix(dummy_pcd)
    print(f"  Matrix shape: {depth_matrix.shape}")
    print(f"  Min depth: {np.min(depth_matrix):.3f}")
    print(f"  Max depth: {np.max(depth_matrix):.3f}")
    
    # Clear drone data
    print("\nClearing drone_2 data...")
    result = client.clear_drone("drone_2")
    print(f"  Result: {result['message']}")
    
    print("\nDone!")


if __name__ == "__main__":
    example_usage()
