"""
Loop Closure Detection + Pose Graph Optimization for SLAM3R Trajectories
=========================================================================

Refines camera positions by:
1. Detecting loop closures (when camera revisits locations)
2. Building a pose graph with temporal + loop closure constraints
3. Optimizing all poses globally for consistency using scipy

Usage:
    python refine_trajectory.py \
        --trajectory_file results_online/my_test_cam0/trajectories.json \
        --image_dir results_online/my_test_cam0/ \
        --output_refined refined_trajectories.json \
        --loop_match_threshold 0.7 \
        --temporal_weight 1.0 \
        --loop_weight 5.0

Output: Refined trajectories.json with optimized positions + loop info
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from collections import defaultdict
from scipy.optimize import minimize


def load_trajectories(filepath: str) -> Dict:
    """Load trajectory data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_trajectories(data: Dict, filepath: str) -> None:
    """Save trajectory data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_images(image_dir: Path, camera_id: str) -> Dict[int, np.ndarray]:
    """Load images for a camera."""
    images = {}
    
    # Try common image formats
    for ext in ['*.jpg', '*.png', '*.JPG', '*.PNG']:
        for img_path in sorted(image_dir.glob(ext)):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Extract frame number from filename (e.g., "frame_123.jpg" -> 123)
                    stem = img_path.stem
                    if stem.isdigit():
                        frame_num = int(stem)
                    else:
                        # Try extracting last numbers
                        import re
                        match = re.search(r'(\d+)$', stem)
                        if match:
                            frame_num = int(match.group(1))
                        else:
                            continue
                    images[frame_num] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
    
    return images


def extract_orb_features(image: np.ndarray, max_features: int = 500) -> Tuple:
    """Extract ORB features from image."""
    orb = cv2.ORB_create(nfeatures=max_features)
    kp, des = orb.detectAndCompute(image, None)
    return kp, des


def match_features(des1: np.ndarray, des2: np.ndarray, threshold: float = 0.75) -> List:
    """Match features using Lowe's ratio test."""
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m in matches:
        if len(m) == 2:
            m, n = m
            if m.distance < threshold * n.distance:
                good.append(m)
    
    return good


def detect_loop_closures(
    images: Dict[int, np.ndarray],
    frames: List[Dict],
    min_frame_gap: int = 10,
    match_threshold: float = 0.7,
    min_matches: int = 20,
) -> List[Tuple[int, int, float]]:
    """
    Detect loop closures by matching features between frame pairs.
    
    Returns: List of (frame_i, frame_j, match_score) tuples
    """
    frame_nums = sorted([f['frame'] for f in frames if f.get('valid', False)])
    
    # Extract features for all frames
    print("Extracting features...")
    features = {}
    for frame_num in frame_nums:
        if frame_num not in images:
            continue
        kp, des = extract_orb_features(images[frame_num])
        features[frame_num] = (kp, des)
    
    # Find loop closures
    print("Detecting loop closures...")
    loop_closures = []
    
    for i, frame_i in enumerate(frame_nums):
        if frame_i not in features:
            continue
        
        kp_i, des_i = features[frame_i]
        if des_i is None:
            continue
        
        # Compare with frames far ahead (at least min_frame_gap frames apart)
        for frame_j in frame_nums[i + min_frame_gap:]:
            if frame_j not in features:
                continue
            
            kp_j, des_j = features[frame_j]
            if des_j is None:
                continue
            
            # Match features
            matches = match_features(des_i, des_j, threshold=match_threshold)
            
            if len(matches) >= min_matches:
                match_score = len(matches) / max(len(kp_i), len(kp_j))
                loop_closures.append((frame_i, frame_j, match_score))
                print(f"  Loop closure: frame {frame_i} ↔ frame {frame_j} "
                      f"({len(matches)} matches, score: {match_score:.3f})")
    
    return loop_closures


def build_optimization_problem(
    frames: List[Dict],
    loop_closures: List[Tuple[int, int, float]],
    temporal_weight: float = 1.0,
    loop_weight: float = 5.0,
) -> Tuple[np.ndarray, List[Tuple], Dict[int, int]]:
    """
    Build an optimization problem for pose graph.
    
    Returns:
        - Initial pose vector (flattened 3D positions)
        - Constraints (edges with weights)
        - Frame to index mapping
    """
    valid_frames = [f for f in frames if f.get('valid', False)]
    frame_to_idx = {f['frame']: i for i, f in enumerate(valid_frames)}
    
    # Initial poses: flatten all positions into one vector
    initial_poses = np.concatenate([
        np.array(f['position']) for f in valid_frames
    ])
    
    # Build constraints
    constraints = []
    
    # Temporal constraints (consecutive frames)
    for i in range(len(valid_frames) - 1):
        pos_i = np.array(valid_frames[i]['position'])
        pos_j = np.array(valid_frames[i + 1]['position'])
        
        constraints.append({
            'type': 'temporal',
            'frame_i': i,
            'frame_j': i + 1,
            'measured_delta': pos_j - pos_i,
            'weight': temporal_weight,
        })
    
    # Loop closure constraints
    for frame_i, frame_j, score in loop_closures:
        if frame_i not in frame_to_idx or frame_j not in frame_to_idx:
            continue
        
        idx_i = frame_to_idx[frame_i]
        idx_j = frame_to_idx[frame_j]
        
        pos_i = np.array(valid_frames[idx_i]['position'])
        pos_j = np.array(valid_frames[idx_j]['position'])
        
        constraints.append({
            'type': 'loop_closure',
            'frame_i': idx_i,
            'frame_j': idx_j,
            'measured_delta': pos_j - pos_i,
            'weight': loop_weight * score,
        })
    
    return initial_poses, constraints, frame_to_idx


def residual_function(poses_flat, constraints, n_frames):
    """
    Compute residuals for all constraints.
    
    Args:
        poses_flat: Flattened position vector [x0, y0, z0, x1, y1, z1, ...]
        constraints: List of constraint dictionaries
        n_frames: Number of frames
    
    Returns: Residual vector (smaller = better fit)
    """
    poses = poses_flat.reshape(n_frames, 3)
    residuals = []
    
    for constraint in constraints:
        idx_i = constraint['frame_i']
        idx_j = constraint['frame_j']
        measured_delta = constraint['measured_delta']
        weight = constraint['weight']
        
        # Predicted delta
        pred_delta = poses[idx_j] - poses[idx_i]
        
        # Residual (should be close to measured)
        error = pred_delta - measured_delta
        weighted_error = error * np.sqrt(weight)
        
        residuals.append(weighted_error)
    
    return np.concatenate(residuals)


def optimize_poses(
    trajectory_file: str,
    image_dir: str,
    output_file: str,
    loop_match_threshold: float = 0.7,
    min_matches: int = 20,
    temporal_weight: float = 1.0,
    loop_weight: float = 5.0,
    max_iterations: int = 100,
) -> None:
    """
    Refine trajectory using loop closure detection and pose graph optimization.
    """
    trajectory_file = Path(trajectory_file)
    image_dir = Path(image_dir)
    output_file = Path(output_file)
    
    print(f"Loading trajectories from {trajectory_file}...")
    data = load_trajectories(str(trajectory_file))
    
    print(f"Loading images from {image_dir}...")
    images = load_images(image_dir, "cam0")
    
    if not images:
        print(f"No images found in {image_dir}")
        return
    
    # Process each camera
    metadata = data.get('metadata', {})
    cameras = metadata.get('cameras', [])
    
    for cam_id in cameras:
        if cam_id.startswith('metadata') or cam_id not in data:
            continue
        
        cam_data = data[cam_id]
        frames = cam_data.get('frames', [])
        
        print(f"\nProcessing camera {cam_id}...")
        print(f"Total frames: {len(frames)}")
        
        # Detect loop closures
        loop_closures = detect_loop_closures(
            images, frames,
            min_frame_gap=10,
            match_threshold=loop_match_threshold,
            min_matches=min_matches,
        )
        
        if loop_closures:
            print(f"Found {len(loop_closures)} loop closures")
        else:
            print("No loop closures detected")
        
        # Build and optimize pose graph using scipy
        print("Building optimization problem...")
        try:
            initial_poses, constraints, frame_to_idx = build_optimization_problem(
                frames, loop_closures,
                temporal_weight=temporal_weight,
                loop_weight=loop_weight,
            )
            
            n_frames = len([f for f in frames if f.get('valid', False)])
            
            # Define objective function
            def objective(poses_flat):
                residuals = residual_function(poses_flat, constraints, n_frames)
                return np.sum(residuals ** 2)  # Minimize sum of squared residuals
            
            print(f"Optimizing {n_frames} poses ({len(constraints)} constraints)...")
            
            # Run optimization
            result = minimize(
                objective,
                initial_poses,
                method='L-BFGS-B',
                options={'maxiter': max_iterations, 'ftol': 1e-6},
            )
            
            optimized_poses = result.x.reshape(n_frames, 3)
            
            # Extract optimized poses
            print("Extracting optimized positions...")
            valid_frame_list = [f for f in frames if f.get('valid', False)]
            for idx, frame_info in enumerate(valid_frame_list):
                frame_info['position'] = optimized_poses[idx].tolist()
            
            print(f"Optimization complete (final error: {result.fun:.6f})")
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            print("Keeping original positions")
        
        # Add loop closure info to metadata
        if loop_closures:
            if 'loop_closures' not in metadata:
                metadata['loop_closures'] = {}
            metadata['loop_closures'][cam_id] = [
                {'frame_i': i, 'frame_j': j, 'score': float(score)}
                for i, j, score in loop_closures
            ]
    
    # Save refined trajectory
    print(f"\nSaving refined trajectories to {output_file}...")
    save_trajectories(data, str(output_file))
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Refine SLAM3R trajectories with loop closure detection and optimization'
    )
    parser.add_argument(
        '--trajectory_file',
        required=True,
        help='Path to trajectories.json'
    )
    parser.add_argument(
        '--image_dir',
        required=True,
        help='Directory containing images for feature matching'
    )
    parser.add_argument(
        '--output_refined',
        default='trajectories_refined.json',
        help='Output refined trajectories.json'
    )
    parser.add_argument(
        '--loop_match_threshold',
        type=float,
        default=0.7,
        help='Lowe ratio test threshold for feature matching (0.0-1.0)'
    )
    parser.add_argument(
        '--min_matches',
        type=int,
        default=20,
        help='Minimum feature matches to consider a loop closure'
    )
    parser.add_argument(
        '--temporal_weight',
        type=float,
        default=1.0,
        help='Weight for temporal consistency constraints'
    )
    parser.add_argument(
        '--loop_weight',
        type=float,
        default=5.0,
        help='Weight for loop closure constraints'
    )
    parser.add_argument(
        '--max_iterations',
        type=int,
        default=100,
        help='Maximum optimization iterations'
    )
    
    args = parser.parse_args()
    optimize_poses(
        args.trajectory_file,
        args.image_dir,
        args.output_refined,
        loop_match_threshold=args.loop_match_threshold,
        min_matches=args.min_matches,
        temporal_weight=args.temporal_weight,
        loop_weight=args.loop_weight,
        max_iterations=args.max_iterations,
    )
