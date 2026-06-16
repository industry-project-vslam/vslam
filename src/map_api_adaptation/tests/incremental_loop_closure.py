"""
Incremental Loop Closure Detection and Pose Graph Optimization
==============================================================

Detects loop closures and refines poses incrementally as frames are added to trajectory.
Designed to be integrated into online tracking pipelines.

Usage:
    detector = IncrementalLoopClosureDetector(
        image_dir=Path("results_online/cam0"),
        min_frame_gap=10,
        match_threshold=0.7,
        min_matches=20
    )
    
    # After each frame is registered:
    detector.process_frame(frame_num, frame_image)
    refined_positions = detector.optimize()  # Returns optimized trajectory so far
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import json


class IncrementalLoopClosureDetector:
    """Detects loop closures and optimizes pose graph incrementally."""
    
    def __init__(
        self,
        image_dir: Path,
        min_frame_gap: int = 10,
        match_threshold: float = 0.7,
        min_matches: int = 20,
        temporal_weight: float = 1.0,
        loop_weight: float = 5.0,
        optimize_every_n_frames: int = 5,
    ):
        """
        Args:
            image_dir: Directory to load images from
            min_frame_gap: Minimum frames between loop closure candidates
            match_threshold: Feature match quality threshold (Lowe ratio test)
            min_matches: Minimum feature matches to consider a loop closure
            temporal_weight: Weight for consecutive frame constraints
            loop_weight: Weight for loop closure constraints
            optimize_every_n_frames: Optimize poses every N frames (not after each frame)
        """
        self.image_dir = Path(image_dir)
        self.min_frame_gap = min_frame_gap
        self.match_threshold = match_threshold
        self.min_matches = min_matches
        self.temporal_weight = temporal_weight
        self.loop_weight = loop_weight
        self.optimize_every_n_frames = optimize_every_n_frames
        
        self.frames = {}  # frame_num -> {'image': img, 'kp': keypoints, 'des': descriptors}
        self.loop_closures = []  # List of (frame_i, frame_j, score)
        self.frame_positions = {}  # frame_num -> [x, y, z]
        self.frame_numbers = []  # Sorted list of processed frame numbers
        self.frame_count = 0
        
        self.orb = cv2.ORB_create(nfeatures=500)
    
    def process_frame(
        self,
        frame_num: int,
        image: np.ndarray,
        position: np.ndarray,
        forward: np.ndarray,
    ) -> None:
        """
        Process a new frame: extract features and store position.
        
        Args:
            frame_num: Frame index
            image: Image array (BGR or grayscale)
            position: Camera position [x, y, z]
            forward: Camera forward direction [fx, fy, fz]
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Extract features
        kp, des = self.orb.detectAndCompute(gray, None)
        
        self.frames[frame_num] = {
            'image': gray,
            'kp': kp,
            'des': des,
        }
        
        self.frame_positions[frame_num] = np.array(position)
        self.frame_numbers.append(frame_num)
        self.frame_numbers.sort()
        self.frame_count += 1
        
        # Detect loop closures with this new frame
        self._detect_loops_for_frame(frame_num)
        
        # Optimize every N frames
        if self.frame_count % self.optimize_every_n_frames == 0:
            print(f"[Optimization] Optimizing poses after frame {frame_num}...")
            self.optimize()
    
    def _detect_loops_for_frame(self, frame_num: int) -> None:
        """Detect loop closures between a new frame and previous frames."""
        if frame_num not in self.frames or len(self.frame_numbers) < 2:
            return
        
        kp_new, des_new = self.frames[frame_num]['kp'], self.frames[frame_num]['des']
        if des_new is None or len(des_new) == 0:
            return
        
        # Compare with frames that are far enough back
        for prev_frame_num in self.frame_numbers:
            if prev_frame_num >= frame_num - self.min_frame_gap:
                continue
            
            if prev_frame_num not in self.frames:
                continue
            
            kp_prev, des_prev = self.frames[prev_frame_num]['kp'], self.frames[prev_frame_num]['des']
            if des_prev is None or len(des_prev) == 0:
                continue
            
            # Match features
            matches = self._match_features(des_prev, des_new)
            
            if len(matches) >= self.min_matches:
                match_score = len(matches) / max(len(kp_prev), len(kp_new))
                
                # Check if this is a new loop closure
                is_new = True
                for (existing_i, existing_j, _) in self.loop_closures:
                    if (existing_i == prev_frame_num and existing_j == frame_num):
                        is_new = False
                        break
                
                if is_new:
                    self.loop_closures.append((prev_frame_num, frame_num, match_score))
                    print(f"[Loop Closure] frame {prev_frame_num} ↔ {frame_num} "
                          f"({len(matches)} matches, score: {match_score:.3f})")
    
    def _match_features(self, des1: np.ndarray, des2: np.ndarray) -> List:
        """Match features using Lowe's ratio test."""
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return []
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except:
            return []
        
        good = []
        for m in matches:
            if len(m) == 2:
                m_obj, n_obj = m
                if m_obj.distance < self.match_threshold * n_obj.distance:
                    good.append(m_obj)
        
        return good
    
    def optimize(self) -> Dict[int, List[float]]:
        """
        Optimize all poses globally using loop closure constraints.
        
        Returns: Dictionary mapping frame_num -> optimized position [x, y, z]
        """
        if len(self.frame_numbers) == 0:
            return {}
        
        # Build optimization problem
        initial_poses = np.concatenate([
            self.frame_positions[fn] for fn in self.frame_numbers
        ])
        
        constraints = []
        
        # Temporal constraints
        for i in range(len(self.frame_numbers) - 1):
            fn_i = self.frame_numbers[i]
            fn_j = self.frame_numbers[i + 1]
            
            pos_i = self.frame_positions[fn_i]
            pos_j = self.frame_positions[fn_j]
            
            constraints.append({
                'type': 'temporal',
                'idx_i': i,
                'idx_j': i + 1,
                'measured_delta': pos_j - pos_i,
                'weight': self.temporal_weight,
            })
        
        # Loop closure constraints
        frame_to_idx = {fn: i for i, fn in enumerate(self.frame_numbers)}
        for fn_i, fn_j, score in self.loop_closures:
            if fn_i not in frame_to_idx or fn_j not in frame_to_idx:
                continue
            
            idx_i = frame_to_idx[fn_i]
            idx_j = frame_to_idx[fn_j]
            
            pos_i = self.frame_positions[fn_i]
            pos_j = self.frame_positions[fn_j]
            
            constraints.append({
                'type': 'loop_closure',
                'idx_i': idx_i,
                'idx_j': idx_j,
                'measured_delta': pos_j - pos_i,
                'weight': self.loop_weight * score,
            })
        
        if not constraints:
            return {fn: self.frame_positions[fn].tolist() for fn in self.frame_numbers}
        
        # Define objective function
        def objective(poses_flat):
            poses = poses_flat.reshape(len(self.frame_numbers), 3)
            residual = 0.0
            
            for constraint in constraints:
                idx_i = constraint['idx_i']
                idx_j = constraint['idx_j']
                measured_delta = constraint['measured_delta']
                weight = constraint['weight']
                
                pred_delta = poses[idx_j] - poses[idx_i]
                error = np.linalg.norm(pred_delta - measured_delta)
                residual += weight * error ** 2
            
            return residual
        
        # Optimize
        result = minimize(
            objective,
            initial_poses,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-6},
        )
        
        optimized_poses = result.x.reshape(len(self.frame_numbers), 3)
        
        # Update frame positions
        for i, fn in enumerate(self.frame_numbers):
            self.frame_positions[fn] = optimized_poses[i]
        
        # Return as dictionary
        refined_positions = {
            fn: self.frame_positions[fn].tolist()
            for fn in self.frame_numbers
        }
        
        return refined_positions
    
    def get_loop_closures(self) -> List[Tuple[int, int, float]]:
        """Return detected loop closures."""
        return self.loop_closures.copy()
    
    def update_trajectory_json(self, trajectory_file: Path) -> None:
        """Update trajectories.json with refined positions."""
        try:
            with open(trajectory_file, 'r') as f:
                data = json.load(f)
            
            # Update positions in all cameras
            for cam_id in data.get('metadata', {}).get('cameras', []):
                if cam_id not in data or cam_id.startswith('metadata'):
                    continue
                
                for frame_info in data[cam_id].get('frames', []):
                    frame_num = frame_info['frame']
                    if frame_num in self.frame_positions:
                        frame_info['position'] = self.frame_positions[frame_num].tolist()
            
            # Add loop closure info to metadata
            if self.loop_closures:
                if 'loop_closures' not in data['metadata']:
                    data['metadata']['loop_closures'] = {}
                for cam_id in data.get('metadata', {}).get('cameras', []):
                    data['metadata']['loop_closures'][cam_id] = [
                        {'frame_i': int(i), 'frame_j': int(j), 'score': float(score)}
                        for i, j, score in self.loop_closures
                    ]
            
            with open(trajectory_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Updated {trajectory_file} with refined positions and loop info")
        
        except Exception as e:
            print(f"Failed to update trajectory file: {e}")
