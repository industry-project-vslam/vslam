import time
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import json


# Camera settings
CAMERA_ID = 1
SCALE = 2.0
CONF = 0.25

# Depth estimation settings
DEPTH_MODEL_NAME = "DPT_Large"  # MiDaS DPT_Large - fast and accurate

# Output settings
OUTPUT_DIR = "captured_frames"
SAVE_INTERVAL = 1  # Save every N frames (1 = save all)


class DepthEstimator:
    """MiDaS depth estimator - standalone, no transformers/huggingface."""
    
    def __init__(self, model_name="DPT_Large"):
        print(f"Loading MiDaS depth model: {model_name}")
        print("This may take a moment on first run...")
        
        # Load MiDaS model from PyTorch Hub
        self.model = torch.hub.load("isl-org/MiDaS", model_name, pretrained=True)
        self.model.eval()
        
        # Set to CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"MiDaS model loaded on {self.device}")
    
    def estimate_depth(self, image):
        """
        Estimate depth from an image.
        
        Args:
            image: numpy array (H, W, 3) in BGR format
            
        Returns:
            depth_map: numpy array (H, W) with depth values
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get the transform from the model
        original_height, original_width = rgb.shape[:2]
        
        with torch.no_grad():
            # Get the transform from the model's config
            if hasattr(self.model, 'transform'):
                transform = self.model.transform
            else:
                # Use default MiDaS transform
                from torchvision.transforms import functional as F
                transform = None
            
            # Prepare input
            input_image = rgb.astype(np.float32)
            input_image = input_image / 255.0
            input_image = torch.from_numpy(input_image).unsqueeze(0)
            input_image = input_image.to(self.device)
            
            # Predict depth
            prediction = self.model(input_image)
            
            # Post-process
            depth = prediction[0]
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(original_height, original_width),
                mode="bicubic",
                align_corners=False
            )
            depth = depth.squeeze()
        
        # Convert to numpy
        depth_map = depth.cpu().numpy()
        
        return depth_map


def depth_to_pointcloud(depth_map, frame):
    """
    Convert depth map to 3D point cloud and render as 2D visualization.
    
    Args:
        depth_map: Depth values (numpy array, HxW)
        frame: Original RGB frame (HxWx3)
    
    Returns:
        pointcloud_vis: 2D visualization of point cloud (HxWx3)
        x_3d, y_3d, z_3d: 3D coordinate arrays
    """
    h, w = depth_map.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create camera matrix (assuming 640x480 camera)
    fx = 640.0 / 2.0  # Approximate focal length
    fy = 480.0 / 2.0
    cx = w / 2.0
    cy = h / 2.0
    
    # Generate 3D point cloud from depth map
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert to 3D coordinates (in meters, assuming metric depth)
    z_3d = depth_map.astype(np.float32)
    x_3d = (x_coords - cx) * z_3d / fx
    y_3d = (y_coords - cy) * z_3d / fy
    
    # Create visualization by rendering points with depth-based coloring
    pointcloud_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalize depth for color mapping
    valid_depths = z_3d[z_3d > 0]
    if len(valid_depths) > 0:
        depth_min, depth_max = valid_depths.min(), valid_depths.max()
        if depth_max > depth_min:
            # Color by depth (blue=close, red=far)
            depth_normalized = (z_3d - depth_min) / (depth_max - depth_min)
            depth_colors = cv2.applyColorMap(
                (depth_normalized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            pointcloud_vis = depth_colors
        else:
            # Uniform color if all depths are same
            pointcloud_vis = rgb
    
    return pointcloud_vis, x_3d, y_3d, z_3d


# Open camera
camera = cv2.VideoCapture(CAMERA_ID)

if not camera.isOpened():
    print("Error: Could not open camera")
    exit(1)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

model = YOLO("yolo26n.pt")

# Initialize depth estimator
depth_estimator = DepthEstimator(model_name=DEPTH_MODEL_NAME)

start = time.time()
count = 0
save_count = 0

cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
cv2.namedWindow("Point Cloud Map", cv2.WINDOW_NORMAL)

print("\n" + "="*60)
print("CAMERA CAPTURE APP")
print("="*60)
print("Press 'q' to quit")
print("Press 's' to save current frame immediately")
print("Frames will be saved to:", OUTPUT_DIR)
print("="*60 + "\n")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame from camera")
        continue

    count += 1
    meanTimePerImage = (time.time() - start) / count
    print(f"\n{meanTimePerImage:.4f} sec/img | {1/meanTimePerImage:.2f} FPS | Frame {count}")

    results = model.predict(frame, conf=CONF, verbose=False)
    annotated = results[0].plot()

    # Run depth estimation
    depth_map = depth_estimator.estimate_depth(frame)
    
    pointcloud_vis, x_3d, y_3d, z_3d = depth_to_pointcloud(depth_map, frame)
    
    pointcloud_scaled = cv2.resize(
        pointcloud_vis, 
        (int(pointcloud_vis.shape[1] * SCALE), int(pointcloud_vis.shape[0] * SCALE)),
        interpolation=cv2.INTER_LINEAR
    )

    h, w = annotated.shape[:2]
    display = cv2.resize(annotated, (int(w * SCALE), int(h * SCALE)), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Detections", display)
    cv2.imshow("Point Cloud Map", pointcloud_scaled)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break
    elif key == ord('s'):
        save_count += 1
        frame_path = os.path.join(OUTPUT_DIR, f"frame_{save_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        pc_path = os.path.join(OUTPUT_DIR, f"frame_{save_count:04d}_pc.jpg")
        cv2.imwrite(pc_path, pointcloud_vis)
        
        pc_data = {
            "points": [
                {"x": float(x), "y": float(y), "z": float(z)}
                for x, y, z in zip(x_3d.flatten(), y_3d.flatten(), z_3d.flatten())
            ]
        }
        json_path = os.path.join(OUTPUT_DIR, f"frame_{save_count:04d}_points.json")
        with open(json_path, 'w') as f:
            json.dump(pc_data, f)
        
        print(f"Saved frame {save_count}: {frame_path}")
        print(f"Saved point cloud: {pc_path}")
        print(f"Saved 3D data: {json_path}")

    if count % SAVE_INTERVAL == 0:
        save_count += 1
        frame_path = os.path.join(OUTPUT_DIR, f"frame_{save_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        pc_path = os.path.join(OUTPUT_DIR, f"frame_{save_count:04d}_pc.jpg")
        cv2.imwrite(pc_path, pointcloud_vis)
        
        pc_data = {
            "points": [
                {"x": float(x), "y": float(y), "z": float(z)}
                for x, y, z in zip(x_3d.flatten(), y_3d.flatten(), z_3d.flatten())
            ]
        }
        json_path = os.path.join(OUTPUT_DIR, f"frame_{save_count:04d}_points.json")
        with open(json_path, 'w') as f:
            json.dump(pc_data, f)
        
        print(f"Auto-saved frame {save_count}")

print(f"\nTotal frames captured: {count}")
print(f"Total frames saved: {save_count}")
print(f"Output directory: {OUTPUT_DIR}")

cv2.destroyAllWindows()
camera.release()
print("Camera released. App closed.")