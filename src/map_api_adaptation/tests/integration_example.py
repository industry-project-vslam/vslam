"""
Integration example: Using the Tracker API with SLAM3R Online Tracker.

This shows how to:
1. Run the online tracker
2. Submit frames to the API as they're processed
3. Get drone location and depth matrix in real-time
4. Use depth matrix for drone navigation
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import threading
import time

try:
    from tests.tracker_api_client import TrackerAPIClient
except ImportError:
    try:
        from tracker_api_client import TrackerAPIClient
    except ImportError:
        print("Error: tracker_api_client.py not found. Make sure you're in the SLAM3R or tests directory.")
        exit(1)


class OnlineTrackerAPIBridge:
    """
    Bridge between SLAM3R Online Tracker and the API.
    
    This class takes outputs from the online tracker and submits them to the API
    while collecting results for real-time navigation feedback.
    """
    
    def __init__(self, api_url: str = "http://localhost:8000", drone_id: str = "main_drone"):
        """
        Initialize the bridge.
        
        Args:
            api_url: API server URL
            drone_id: Unique identifier for this drone
        """
        self.client = TrackerAPIClient(api_url)
        self.drone_id = drone_id
        self.frame_count = 0
        self.latest_location = None
        self.latest_depth_matrix = None
        self.lock = threading.Lock()
    
    def check_api_ready(self) -> bool:
        """Check if API server is running."""
        if not self.client.health_check():
            print(f"❌ API server not running at {self.client.base_url}")
            print("   Start it with: uvicorn tracker_api:app --host 0.0.0.0 --port 8000")
            return False
        print("✅ API server is ready")
        return True
    
    def submit_frame(self, registered_pcd: np.ndarray,
                    registered_conf: np.ndarray,
                    rgb_img: np.ndarray,
                    frame_idx: int = None) -> Dict:
        """
        Submit a frame from the online tracker to the API.
        
        Args:
            registered_pcd: (H, W, 3) registered point cloud
            registered_conf: (H, W) confidence map
            rgb_img: (H, W, 3) RGB image
            frame_idx: Optional frame index (auto-incremented if not provided)
            
        Returns:
            API response with drone location and depth matrix
        """
        if frame_idx is None:
            frame_idx = self.frame_count
        
        try:
            result = self.client.process_frame(
                drone_id=self.drone_id,
                frame_idx=frame_idx,
                registered_pcd=registered_pcd,
                registered_conf=registered_conf,
                rgb_img=rgb_img
            )
            
            with self.lock:
                self.frame_count = max(self.frame_count, frame_idx + 1)
                self.latest_location = np.array(result['location'])
                self.latest_depth_matrix = np.array(result['depth_matrix'])
            
            return result
            
        except Exception as e:
            print(f"Error submitting frame: {e}")
            return None
    
    def get_drone_location(self) -> np.ndarray:
        """Get latest drone location from API."""
        with self.lock:
            return self.latest_location.copy() if self.latest_location is not None else None
    
    def get_depth_matrix(self) -> np.ndarray:
        """Get latest 8x8 depth matrix from API."""
        with self.lock:
            return self.latest_depth_matrix.copy() if self.latest_depth_matrix is not None else None
    
    def get_navigation_feedback(self) -> Dict:
        """
        Get navigation feedback from the latest depth matrix.
        
        Returns:
            Dict with:
            - safe_directions: List of safe directions (lowest depth values)
            - obstacle_map: 2D obstacle representation
            - recommended_direction: Direction with most clearance
            - min_distance: Minimum distance to obstacles
        """
        depth_matrix = self.get_depth_matrix()
        if depth_matrix is None:
            return None
        
        # Find safe areas (low depth values)
        safe_areas = depth_matrix < np.percentile(depth_matrix, 33)
        
        # Find obstruction areas
        obstacles = depth_matrix > np.percentile(depth_matrix, 67)
        
        # Find recommended direction (center of lowest depth area)
        if np.any(safe_areas):
            safe_indices = np.argwhere(safe_areas)
            recommended = safe_indices.mean(axis=0)
        else:
            recommended = np.array([4, 4])  # Center
        
        return {
            'safe_areas': safe_areas,
            'obstacles': obstacles,
            'recommended_direction': recommended.tolist(),
            'min_depth': float(np.min(depth_matrix)),
            'max_depth': float(np.max(depth_matrix)),
            'mean_depth': float(np.mean(depth_matrix[depth_matrix > 0]))
        }
    
    def download_map(self, save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Download the current point cloud map.
        
        Args:
            save_path: Optional path to save the point cloud
            
        Returns:
            (points, colors) tuple
        """
        return self.client.get_pointcloud(self.drone_id, save_path=save_path)
    
    def clear_map(self) -> bool:
        """Clear the drone's map from the API."""
        try:
            result = self.client.clear_drone(self.drone_id)
            return result['success']
        except Exception as e:
            print(f"Error clearing map: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get current map status."""
        try:
            return self.client.get_drone_status(self.drone_id)
        except Exception as e:
            print(f"Error getting status: {e}")
            return None


# ============================================================================
# Integration Example with Online Tracker
# ============================================================================

def example_integration():
    """
    Example of integrating with SLAM3R online tracker.
    
    This demonstrates the data flow from online tracker → API → Navigation feedback
    """
    
    print("\n" + "="*70)
    print("Online Tracker → API Integration Example")
    print("="*70)
    
    # Initialize bridge
    bridge = OnlineTrackerAPIBridge(
        api_url="http://localhost:8000",
        drone_id="test_drone"
    )
    
    # Check API is ready
    if not bridge.check_api_ready():
        return
    
    print(f"\n📡 Using drone ID: {bridge.drone_id}")
    print("   API URL: http://localhost:8000")
    
    # Simulate processing frames from online tracker
    print("\n🎬 Simulating frame processing from online tracker...\n")
    
    h, w = 224, 224
    num_frames = 5
    
    for frame_idx in range(num_frames):
        print(f"Frame {frame_idx}:")
        
        # Simulate frame data from online tracker
        # In real scenario, these would come from:
        # - registered_pcds from tracker.preds_dir / 'registered_pcds.npy'
        # - registered_confs from tracker.preds_dir / 'registered_confs.npy'
        # - rgb_imgs from tracker.preds_dir / 'input_imgs.npy'
        
        registered_pcd = np.random.rand(h, w, 3) * (10 + frame_idx)
        registered_conf = np.random.rand(h, w) * 20
        rgb_img = np.random.rand(h, w, 3) * 255
        
        # Submit to API
        result = bridge.submit_frame(registered_pcd, registered_conf, rgb_img, frame_idx)
        
        if result:
            location = result['location']
            print(f"  ✓ Location: ({location[0]:.2f}, {location[1]:.2f}, {location[2]:.2f})")
            print(f"  ✓ Frames processed: {result['frame_count']}")
            
            # Get navigation feedback
            nav = bridge.get_navigation_feedback()
            if nav:
                print(f"  ✓ Recommended direction: {nav['recommended_direction']}")
                print(f"  ✓ Min depth: {nav['min_depth']:.2f}, Max depth: {nav['max_depth']:.2f}")
        else:
            print(f"  ✗ Failed to process frame")
        
        print()
        time.sleep(0.5)
    
    # Get final status
    print("\n📊 Final Status:")
    status = bridge.get_status()
    if status:
        s = status['status']
        print(f"  Created at: {s['created_at']}")
        print(f"  Last updated: {s['last_updated']}")
        print(f"  Total frames: {s['frame_count']}")
        print(f"  Total points: {s['max_points']:,}")
        print(f"  Current location: {s['location']}")
    
    print("\n✅ Integration example completed!")


# ============================================================================
# Real Online Tracker Integration
# ============================================================================

def integrate_with_online_tracker(output_dir: str, api_url: str = "http://localhost:8000",
                                   drone_id: str = "main_drone"):
    """
    Real integration with SLAM3R online tracker output.
    
    This function monitors the online tracker output directory and submits
    frames to the API as they become available.
    
    Args:
        output_dir: Output directory from online_tracker (e.g., results_online/my_test_cam0)
        api_url: API server URL
        drone_id: Unique drone identifier
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    preds_dir = output_path / 'preds'
    
    if not preds_dir.exists():
        print(f"❌ Predictions directory not found: {preds_dir}")
        return
    
    bridge = OnlineTrackerAPIBridge(api_url, drone_id)
    
    if not bridge.check_api_ready():
        return
    
    print(f"\n📡 Monitoring: {output_dir}")
    print(f"   Drone ID: {drone_id}")
    print(f"   API URL: {api_url}\n")
    
    try:
        # Load prediction arrays
        registered_pcds = np.load(preds_dir / 'registered_pcds.npy')
        registered_confs = np.load(preds_dir / 'registered_confs.npy')
        rgb_imgs = np.load(preds_dir / 'input_imgs.npy')
        
        num_frames = registered_pcds.shape[0]
        print(f"Found {num_frames} frames to process\n")
        
        last_processed = 0
        while True:
            current_frames = registered_pcds.shape[0]
            
            # Process new frames
            while last_processed < current_frames:
                frame_idx = last_processed
                result = bridge.submit_frame(
                    registered_pcds[frame_idx],
                    registered_confs[frame_idx],
                    rgb_imgs[frame_idx],
                    frame_idx
                )
                
                if result:
                    print(f"✓ Frame {frame_idx}: Location {result['location']}")
                
                last_processed += 1
            
            # Check for new frames periodically
            time.sleep(1.0)
            
            # Reload to check for updates
            try:
                registered_pcds = np.load(preds_dir / 'registered_pcds.npy')
            except:
                pass
    
    except KeyboardInterrupt:
        print("\n\nStopped monitoring")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Multi-Drone SLAM Tracker API - Online Tracker Integration")
    
    # Run example
    example_integration()
    
    # To use with real online tracker:
    # integrate_with_online_tracker("results_online/my_test_cam0", drone_id="my_drone")
