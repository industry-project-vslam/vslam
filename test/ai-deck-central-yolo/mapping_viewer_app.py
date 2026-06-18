import cv2
import numpy as np
import json
import os
from pathlib import Path


class PointCloudViewer:
    def __init__(self, data_dir="captured_frames"):
        self.data_dir = Path(data_dir)
        self.points = []
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.rotation_angle = 0
        self.load_pointcloud_data()
    
    def load_pointcloud_data(self):
        print(f"Loading point cloud data from: {self.data_dir}")
        
        if not self.data_dir.exists():
            print(f"Warning: Directory {self.data_dir} does not exist")
            return
        
        json_files = sorted(self.data_dir.glob("*_points.json"))
        
        if len(json_files) == 0:
            print("No point cloud JSON files found!")
            return
        
        print(f"Found {len(json_files)} point cloud files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for point in data["points"]:
                    self.points.append((point["x"], point["y"], point["z"]))
                
                print(f"  Loaded {len(data['points'])} points from {json_file.name}")
            except Exception as e:
                print(f"  Error loading {json_file.name}: {e}")
        
        print(f"\nTotal points loaded: {len(self.points)}")
    
    def project_3d_to_2d(self, points_3d, view_width=800, view_height=800):
        if len(points_3d) == 0:
            return np.zeros((view_height, view_width, 3), dtype=np.uint8)
        
        image = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        image[:] = [20, 20, 30]
        
        xs = np.array([p[0] for p in points_3d])
        ys = np.array([p[1] for p in points_3d])
        zs = np.array([p[2] for p in points_3d])
        
        xs = xs - xs.mean()
        ys = ys - ys.mean()
        
        scale = min(view_height / 2, view_width / 2) / max(np.abs(xs).max(), np.abs(ys).max(), 1)
        scale *= self.zoom
        
        cos_a = np.cos(self.rotation_angle)
        sin_a = np.sin(self.rotation_angle)
        xs_rot = xs * cos_a - ys * sin_a
        ys_rot = xs * sin_a + ys * cos_a
        
        x_proj = (xs_rot * scale + view_width / 2 + self.offset_x).astype(int)
        y_proj = (-ys_rot * scale + view_height / 2 + self.offset_y).astype(int)
        
        if len(zs) > 0:
            z_min, z_max = zs.min(), zs.max()
            if z_max > z_min:
                depth_normalized = (zs - z_min) / (z_max - z_min)
            else:
                depth_normalized = np.zeros(len(zs))
            
            for i, (x, y, depth) in enumerate(zip(x_proj, y_proj, depth_normalized)):
                if 0 <= y < view_height and 0 <= x < view_width:
                    color = cv2.applyColorMap(
                        np.uint8(depth * 255),
                        cv2.COLORMAP_JET
                    )[0]
                    image[y, x] = color
        
        return image
    
    def render_pointcloud(self, view_size=800):
        return self.project_3d_to_2d(self.points, view_size, view_size)
    
    def run_viewer(self):
        print("\n" + "="*60)
        print("POINT CLOUD MAP VIEWER")
        print("="*60)
        print("Controls:")
        print("  Arrow keys: Move/pan the view")
        print("  + / - : Zoom in/out")
        print("  r : Rotate view")
        print("  s : Save current view as image")
        print("  q : Quit")
        print("="*60 + "\n")
        
        if len(self.points) == 0:
            print("No point cloud data to display!")
            print("Make sure you have captured frames with the camera app.")
            return
        
        view_size = 800
        cv2.namedWindow("Point Cloud Map", cv2.WINDOW_NORMAL)
        
        while True:
            image = self.render_pointcloud(view_size)
            
            overlay = image.copy()
            cv2.putText(overlay, f"Points: {len(self.points)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Zoom: {self.zoom:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Rotation: {self.rotation_angle:.2f} rad", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Point Cloud Map", overlay)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = "pointcloud_map.png"
                cv2.imwrite(filename, overlay)
                print(f"Saved: {filename}")
            elif key == ord('+') or key == ord('='):
                self.zoom *= 1.1
                print(f"Zoom: {self.zoom:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.zoom *= 0.9
                print(f"Zoom: {self.zoom:.2f}")
            elif key == ord('r'):
                self.rotation_angle += 0.5
                print(f"Rotation: {self.rotation_angle:.2f} rad")
            elif key == cv2.KEY_UP:
                self.offset_y += 20
            elif key == cv2.KEY_DOWN:
                self.offset_y -= 20
            elif key == cv2.KEY_LEFT:
                self.offset_x -= 20
            elif key == cv2.KEY_RIGHT:
                self.offset_x += 20
        
        cv2.destroyAllWindows()


def main():
    viewer = PointCloudViewer(data_dir="captured_frames")
    viewer.run_viewer()


if __name__ == "__main__":
    main()