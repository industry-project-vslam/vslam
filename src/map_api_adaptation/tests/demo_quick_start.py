"""
Quick start examples for the Multi-Drone SLAM Tracker API.

This script demonstrates:
1. Starting the API server
2. Submitting frames from multiple drones
3. Monitoring drone status in real-time
4. Downloading point clouds
5. Clearing drone data
"""

import numpy as np
import json
import time
import subprocess
import sys
from pathlib import Path
import threading

def check_dependencies():
    """Check if all required packages are installed."""
    required = ['fastapi', 'uvicorn', 'requests', 'numpy', 'torch']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies installed")
    return True


def start_server():
    """Start the FastAPI server in background."""
    print("\n📡 Starting API server...")
    print("   Command: uvicorn api.main:app --host 0.0.0.0 --port 8000")
    
    try:
        # Start server in background
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app", 
             "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            import requests
            response = requests.get("http://localhost:8000/api/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server started successfully")
                return process
        except:
            pass
        
        print("⚠️  Server may have failed to start. Check logs.")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None


def demo_single_drone():
    """Demo: Process frames from a single drone."""
    print("\n" + "="*70)
    print("DEMO 1: Single Drone - Frame Processing")
    print("="*70)
    
    try:
        from tests.tracker_api_client import TrackerAPIClient
    except ImportError:
        try:
            from tracker_api_client import TrackerAPIClient
        except ImportError:
            print("❌ tracker_api_client not found in current directory")
            return False
    
    client = TrackerAPIClient("http://localhost:8000")
    
    # Check API is running
    if not client.health_check():
        print("❌ API server is not responding")
        return False
    
    print("\n📊 Creating dummy frame data...")
    h, w = 224, 224
    drone_id = "demo_drone_1"
    
    # Process multiple frames
    for frame_idx in range(3):
        print(f"\n  Frame {frame_idx}:")
        
        # Create synthetic data
        pcd = np.random.rand(h, w, 3) * 10
        conf = np.random.rand(h, w) * 20
        rgb = np.random.rand(h, w, 3) * 255
        
        try:
            result = client.process_frame(drone_id, frame_idx, pcd, conf, rgb)
            
            print(f"    ✓ Location: {[round(x, 2) for x in result['location']]}")
            print(f"    ✓ Depth matrix: {len(result['depth_matrix'])}x{len(result['depth_matrix'][0])}")
            print(f"    ✓ Frames accumulated: {result['frame_count']}")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return False
    
    print("\n✅ Single drone demo completed")
    return True


def demo_multi_drone():
    """Demo: Process frames from multiple drones."""
    print("\n" + "="*70)
    print("DEMO 2: Multi-Drone - Concurrent Processing")
    print("="*70)
    
    try:
        from tests.tracker_api_client import TrackerAPIClient
    except ImportError:
        try:
            from tracker_api_client import TrackerAPIClient
        except ImportError:
            print("❌ tracker_api_client not found in current directory")
            return False
    
    client = TrackerAPIClient("http://localhost:8000")
    
    print("\n🚁 Processing frames for 3 drones concurrently...\n")
    
    h, w = 224, 224
    drones = ["drone_A", "drone_B", "drone_C"]
    
    def process_drone_frames(drone_id):
        for frame_idx in range(2):
            pcd = np.random.rand(h, w, 3) * (10 + np.random.rand() * 5)
            conf = np.random.rand(h, w) * 20
            rgb = np.random.rand(h, w, 3) * 255
            
            try:
                result = client.process_frame(drone_id, frame_idx, pcd, conf, rgb)
                print(f"  {drone_id}: Frame {frame_idx} - Location {result['location']}")
            except Exception as e:
                print(f"  {drone_id}: Error - {e}")
    
    # Process drones in parallel
    threads = []
    for drone_id in drones:
        t = threading.Thread(target=process_drone_frames, args=(drone_id,))
        t.start()
        threads.append(t)
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    print("\n✅ Multi-drone processing completed")
    return True


def demo_status_monitoring():
    """Demo: Monitor system status."""
    print("\n" + "="*70)
    print("DEMO 3: Status Monitoring")
    print("="*70)
    
    try:
        from tests.tracker_api_client import TrackerAPIClient
    except ImportError:
        try:
            from tracker_api_client import TrackerAPIClient
        except ImportError:
            print("❌ tracker_api_client not found in current directory")
            return False
    
    client = TrackerAPIClient("http://localhost:8000")
    
    print("\n📊 System Status:\n")
    
    try:
        status = client.get_all_status()
        print(f"  Total active drones: {status['total_drones']}")
        print(f"  Server time: {status['timestamp']}\n")
        
        for drone_id, drone_status in status['drones'].items():
            print(f"  {drone_id}:")
            print(f"    - Frames: {drone_status['frame_count']}")
            print(f"    - Points: {drone_status['max_points']:,}")
            print(f"    - Created: {drone_status['created_at'].split('T')[0]}")
            print(f"    - Last update: {drone_status['last_updated']}")
            if drone_status['location']:
                loc = drone_status['location']
                print(f"    - Position: ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f})")
            print()
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    print("✅ Status monitoring completed")
    return True


def demo_depth_matrix():
    """Demo: Generate and inspect depth matrix."""
    print("\n" + "="*70)
    print("DEMO 4: Depth Matrix Generation")
    print("="*70)
    
    try:
        from tests.tracker_api_client import TrackerAPIClient
    except ImportError:
        try:
            from tracker_api_client import TrackerAPIClient
        except ImportError:
            print("❌ tracker_api_client not found in current directory")
            return False
    
    client = TrackerAPIClient("http://localhost:8000")
    
    print("\n🔍 Computing depth matrix...\n")
    
    # Create synthetic point cloud with gradient depth
    h, w = 224, 224
    pcd = np.zeros((h, w, 3), dtype=np.float32)
    
    # Create diagonal depth gradient
    for i in range(h):
        for j in range(w):
            pcd[i, j, 0] = j / w * 10  # x: left to right
            pcd[i, j, 1] = i / h * 10  # y: top to bottom
            pcd[i, j, 2] = (i + j) / (h + w) * 15  # z: diagonal gradient
    
    try:
        result = client.compute_depth_matrix(pcd)
        
        matrix = np.array(result['depth_matrix'])
        print("  8x8 Depth Matrix:")
        print("  " + "-" * 50)
        for row in matrix:
            print("  " + "  ".join(f"{v:6.2f}" for v in row))
        print("  " + "-" * 50)
        print(f"\n  Statistics:")
        print(f"    Min: {result['min']:.3f}")
        print(f"    Max: {result['max']:.3f}")
        print(f"    Mean: {result['mean']:.3f}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    print("\n✅ Depth matrix generation completed")
    return True


def demo_cleanup():
    """Demo: Clear drone data."""
    print("\n" + "="*70)
    print("DEMO 5: Data Cleanup")
    print("="*70)
    
    try:
        from tests.tracker_api_client import TrackerAPIClient
    except ImportError:
        try:
            from tracker_api_client import TrackerAPIClient
        except ImportError:
            print("❌ tracker_api_client not found in current directory")
            return False
    
    client = TrackerAPIClient("http://localhost:8000")
    
    print("\n🧹 Clearing drone data...\n")
    
    try:
        # Get current status
        status_before = client.get_all_status()
        print(f"  Before: {status_before['total_drones']} drones")
        
        # Clear first drone
        if status_before['drones']:
            first_drone = list(status_before['drones'].keys())[0]
            result = client.clear_drone(first_drone)
            print(f"  - Cleared: {first_drone}")
            print(f"    Message: {result['message']}")
            
            # Get status after
            status_after = client.get_all_status()
            print(f"\n  After: {status_after['total_drones']} drones")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    print("\n✅ Cleanup demo completed")
    return True


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("Multi-Drone SLAM Tracker API - Quick Start Demo")
    print("="*70)
    
    # Check dependencies
    if not check_dependencies():
        print("\nInstall missing dependencies and try again.")
        return
    
    # Start server
    server_process = start_server()
    if not server_process:
        print("\nCould not start API server. Exiting.")
        return
    
    try:
        # Run demos
        demos = [
            demo_single_drone,
            demo_multi_drone,
            demo_status_monitoring,
            demo_depth_matrix,
            demo_cleanup
        ]
        
        for demo in demos:
            try:
                if not demo():
                    print(f"⚠️  {demo.__name__} failed")
            except Exception as e:
                print(f"❌ {demo.__name__} error: {e}")
            
            time.sleep(1)
        
        print("\n" + "="*70)
        print("✅ All demos completed!")
        print("="*70)
        print("\n📚 Next steps:")
        print("  1. View API documentation: http://localhost:8000/docs")
        print("  2. Check status: curl http://localhost:8000/api/status")
        print("  3. View API_README.md for more information")
        print("\nKeep the server running with: uvicorn tracker_api:app --reload")
        
    finally:
        print("\n\nPress Ctrl+C to stop the server...")
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
            server_process.terminate()
            server_process.wait(timeout=5)


if __name__ == "__main__":
    main()
