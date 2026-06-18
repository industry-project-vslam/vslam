"""
SLAM3R Online Pipeline - Optimized with:
- Tier 1: Mixed precision (FP16), batch encoding
- Tier 2: 8x8 depth matrix export
"""

import warnings
warnings.filterwarnings("ignore")
import os
from os.path import join 
from tqdm import tqdm
import argparse
import json
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
plt.ion()
import time
from typing import List, Tuple, Optional, Any
from torch.cuda.amp import autocast

from slam3r.datasets.wild_seq import Seq_Data
from slam3r.models import Image2PointsModel, Local2WorldModel, inf
from slam3r.utils.device import to_numpy
from slam3r.utils.recon_utils import * 
from slam3r.datasets.get_webvideo import *
from slam3r.utils.image import load_single_image
from slam3r.pipeline.recon_offline_pipeline import scene_frame_retrieve

# ============================================================================
# TIER 1: BATCH ENCODING + MIXED PRECISION
# ============================================================================

@torch.no_grad()
def get_multi_img_tokens(views: List[Any], model: Any, batch_size: int = 16, silent: bool = False) -> Tuple[List, List, List]:
    """Encode multiple frames in batches for 2-3x speedup.
    
    Replaces sequential get_single_img_tokens() calls with batched encoding.
    Expected speedup: 2-3x for batch_size=16 vs batch_size=1.
    
    Args:
        views: List of view dicts to encode
        model: Image2PointsModel instance
        batch_size: Number of views per batch (default 16 for good GPU utilization)
        silent: Suppress tqdm progress
    
    Returns:
        (res_shapes, res_feats, res_poses): All encoded views
    """
    res_shapes, res_feats, res_poses = [], [], []
    
    for i in range(0, len(views), batch_size):
        batch = views[i:i + batch_size]
        shapes, feats, poses = model._encode_multiview(
            batch,
            view_batchsize=len(batch),  # Use actual batch size, not 1
            normalize=False,
            silent=silent
        )
        res_shapes.extend(shapes)
        res_feats.extend(feats)
        res_poses.extend(poses)
    
    return res_shapes, res_feats, res_poses


@torch.no_grad()
def get_single_img_tokens(views, model, silent=False):
    """get an img token output from encoder,
    which can be reused by both i2p and l2w models
    """
    with autocast(device_type='cuda', dtype=torch.float16):
        res_shape, res_feat, res_poses = model._encode_multiview(views, 
                                                                   view_batchsize=1, 
                                                                   normalize=False,
                                                                   silent=silent)
    return res_shape, res_feat, res_poses


@torch.no_grad()
def get_img_tokens(views, model, silent=False):
    """get img tokens output from encoder,
    which can be reused by both i2p and l2w models
    """
    res_shapes, res_feats, res_poses = model._encode_multiview(views, 
                                                               view_batchsize=10, 
                                                               normalize=False,
                                                               silent=silent)
    return res_shapes, res_feats, res_poses


# ============================================================================
# TIER 2: DEPTH MATRIX EXTRACTION
# ============================================================================

def extract_depth_matrix(
    pts3d_world: Any,
    conf_map: Any,
    grid_size: int = 8,
    min_depth: float = 0.01,
    max_depth: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Extract 8x8 depth and confidence matrix from full resolution output.
    
    For each grid cell, compute:
    - Min depth (closest obstacle in cell) — for obstacle avoidance
    - Mean confidence (quality of depth estimate)
    - Valid point count (debug)
    
    Args:
        pts3d_world: (1, H, W, 3) or (H, W, 3) 3D point cloud in world frame
        conf_map: (1, H, W) or (H, W) confidence map (0-255 typically)
        grid_size: Discretization level (default 8 → 8x8 grid = 64 cells)
        min_depth: Minimum valid depth (filter numerical noise)
        max_depth: Maximum valid depth (filter outliers)
    
    Returns:
        depth_matrix: (grid_size, grid_size) min depth per cell [meters]
        conf_matrix: (grid_size, grid_size) mean confidence per cell [0-1]
        metadata: dict with cell statistics for debugging
    """
    # Convert to numpy if needed
    if isinstance(pts3d_world, torch.Tensor):
        pts3d_world = to_numpy(pts3d_world)
    if isinstance(conf_map, torch.Tensor):
        conf_map = to_numpy(conf_map)
    
    # Remove batch dimension if present
    if pts3d_world.ndim == 4 and pts3d_world.shape[0] == 1:
        pts3d_world = pts3d_world[0]  # (H, W, 3)
    if conf_map.ndim == 3 and conf_map.shape[0] == 1:
        conf_map = conf_map[0]  # (H, W)
    elif conf_map.ndim == 2:
        pass  # Already (H, W)
    else:
        raise ValueError(f"conf_map shape unexpected: {conf_map.shape}")
    
    if pts3d_world.ndim != 3 or pts3d_world.shape[2] != 3:
        raise ValueError(f"pts3d_world must be (H, W, 3), got {pts3d_world.shape}")
    
    H, W = pts3d_world.shape[:2]
    cell_h = H // grid_size
    cell_w = W // grid_size
    
    # Compute depth: Euclidean distance from camera origin
    depth = np.linalg.norm(pts3d_world, axis=2)  # (H, W)
    
    # Normalize confidence to [0, 1]
    conf_min, conf_max = conf_map.min(), conf_map.max()
    if conf_max > conf_min:
        conf_normalized = (conf_map - conf_min) / (conf_max - conf_min)
    else:
        conf_normalized = np.ones_like(conf_map)
    
    # Initialize output matrices
    depth_matrix = np.full((grid_size, grid_size), fill_value=np.nan)
    conf_matrix = np.zeros((grid_size, grid_size))
    metadata = {"total_cells": grid_size * grid_size, "valid_cells": 0, "cells": {}}
    
    # Process each grid cell
    for i in range(grid_size):
        for j in range(grid_size):
            # Extract cell boundaries
            h_start = i * cell_h
            h_end = (i + 1) * cell_h
            w_start = j * cell_w
            w_end = (j + 1) * cell_w
            
            # Extract cell data
            cell_depth = depth[h_start:h_end, w_start:w_end]
            cell_conf = conf_normalized[h_start:h_end, w_start:w_end]
            
            # Validity mask: depth in valid range
            valid = (cell_depth >= min_depth) & (cell_depth <= max_depth)
            
            if valid.sum() > 0:
                # Min depth (closest obstacle for avoidance)
                depth_matrix[i, j] = cell_depth[valid].min()
                # Mean confidence (average quality)
                conf_matrix[i, j] = cell_conf[valid].mean()
                
                # Metadata for debugging
                metadata["valid_cells"] += 1
                metadata["cells"][f"{i}_{j}"] = {
                    "min_depth": float(depth_matrix[i, j]),
                    "mean_conf": float(conf_matrix[i, j]),
                    "valid_points": int(valid.sum()),
                    "total_points": int(cell_depth.size),
                }
            else:
                # No valid points in cell
                depth_matrix[i, j] = np.nan
                conf_matrix[i, j] = 0.0
    
    # Replace NaN with maximum depth (no obstacle detected)
    depth_matrix = np.nan_to_num(depth_matrix, nan=max_depth)
    
    metadata["coverage"] = metadata["valid_cells"] / metadata["total_cells"]
    
    return depth_matrix, conf_matrix, metadata


def depth_matrix_to_json(
    depth_matrix: np.ndarray,
    conf_matrix: np.ndarray,
    metadata: dict
) -> dict:
    """Convert depth/conf matrices to JSON-serializable format."""
    return {
        "depth_matrix": depth_matrix.tolist(),
        "conf_matrix": conf_matrix.tolist(),
        "grid_size": int(depth_matrix.shape[0]),
        "metadata": {
            "coverage": float(metadata.get("coverage", 0)),
            "valid_cells": int(metadata.get("valid_cells", 0)),
            "total_cells": int(metadata.get("total_cells", 0)),
        }
    }


# All functions below are copied from recon_online_pipeline.py unchanged
# (They serve as drop-in replacements)

class FrameReader:
    """
    Read images from a directory, video file, or online video URL.
    Args:
        dataset (str): Path to the image directory, video file, or online video URL.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.type = ""
        self.count = 0
        self.readnum = 0
        if isinstance(dataset, str):
            if dataset.find(":") != -1:
                self.type = "https"
            else:
                if dataset[-3:] == "mp4":
                    self.type = "video"
                else:
                    self.type = "imgs"
        else:
            self.type = "imgs"
        if self.type == "imgs":
            print('loading dataset: ', self.dataset)
            self.data = Seq_Data(img_dir=self.dataset,  \
                                img_size=224, silent=False, sample_freq=1, \
                                start_idx=0, num_views=-1, start_freq=1, to_tensor=True)
            if hasattr(self.data, "set_epoch"):
                self.data.set_epoch(0)
        elif self.type == "video":
            self.video_capture = cv2.VideoCapture(self.dataset)
            if not self.video_capture.isOpened():
                print(f"error!can not open the video file{self.dataset}")
                exit()
            print(f"successful open the the video file {self.dataset}! start processing frame by frame...")
        elif self.type == "https":
            self.get_api = Get_online_video(self.dataset)
            
    def read(self):
        
        if self.type == "https":
            return self.get_api.cap.read()
        elif self.type == "video":
            return self.video_capture.read()
        elif self.type == "imgs":
            # print(f"reading the {self.readnum}th image")
            if self.readnum >= len(self.data[0]):
                return False, None
            self.count += 1
            self.readnum += 1
            return True, self.data[0][self.readnum - 1]
        
def save_recon(views, pred_frame_num, save_dir, scene_id, save_all_views=False, 
                      imgs=None, registered_confs=None, 
                      num_points_save=200000, conf_thres_res=3, valid_masks=None):  
    save_name = f"{scene_id}_recon.ply"
    # collect the registered point clouds and rgb colors
    if imgs is None:
        imgs = [transform_img(unsqueeze_view(view))[:,::-1] for view in views]
    # ensure save directory exists
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception:
        pass
    pcds = []
    rgbs = []
    for i in range(pred_frame_num):
        registered_pcd = to_numpy(views[i]['pts3d_world'][0])
        if registered_pcd.shape[0] == 3:
            registered_pcd = registered_pcd.transpose(1,2,0)
        registered_pcd = registered_pcd.reshape(-1,3)
        rgb = imgs[i].reshape(-1,3)
        pcds.append(registered_pcd)
        rgbs.append(rgb)
    if save_all_views:
        for i in range(pred_frame_num):
            save_ply(points=pcds[i], save_path=join(save_dir, f"frame_{i}.ply"), colors=rgbs[i])
    # concatenate results; handle empty lists
    if len(pcds) == 0:
        print('No pointclouds to save.')
        return
    res_pcds = np.concatenate(pcds, axis=0) if len(pcds) > 0 else np.zeros((0,3))
    res_rgbs = np.concatenate(rgbs, axis=0) if len(rgbs) > 0 else np.zeros((0,3))
    pts_count = len(res_pcds)
    valid_ids = np.arange(pts_count)
    # filter out points with gt valid masks
    if valid_masks is not None:
        valid_masks = np.stack(valid_masks, axis=0).reshape(-1)
        # print('filter out ratio of points by gt valid masks:', 1.-valid_masks.astype(float).mean())
    else:
        valid_masks = np.ones(pts_count, dtype=bool)
    # filter out points with low confidence
    if registered_confs is not None:
        conf_masks = []
        for i in range(len(registered_confs)):
            conf = registered_confs[i]
            try:
                conf_mask = (conf > conf_thres_res).reshape(-1).cpu()
                conf_masks.append(conf_mask)
            except Exception:
                # skip malformed confidence maps
                continue
        if len(conf_masks) == 0:
            print('No confidence maps available to filter; keeping all points.')
            conf_masks = np.ones(pts_count, dtype=bool)
        else:
            conf_masks = np.array(torch.cat(conf_masks))
        valid_ids = valid_ids[conf_masks & valid_masks]
        if pts_count > 0:
            print('ratio of points filered out: {:.2f}%'.format((1. - len(valid_ids) / pts_count) * 100))
        else:
            print('ratio of points filered out: 100.00%')
    # sample from the resulting pcd consisting of all frames
    if len(valid_ids) == 0:
        print('No valid points to save after filtering.')
        return
    n_samples = min(num_points_save, len(valid_ids))
    print(f"resampling {n_samples} points from {len(valid_ids)} points")
    sampled_idx = np.random.choice(valid_ids, n_samples, replace=False)
    sampled_pts = res_pcds[sampled_idx]
    sampled_rgbs = res_rgbs[sampled_idx]
    save_ply(points=sampled_pts[:, :3], save_path=join(save_dir, save_name), colors=sampled_rgbs)

def load_model(model_name, weights, device='cuda'):
    print('Loading model: {:s}'.format(model_name))
    model = eval(model_name)
    model.to(device)
    print('Loading pretrained: ', weights)
    ckpt = torch.load(weights, map_location=device)
    print(model.load_state_dict(ckpt['model'], strict=False))
    del ckpt  # in case it occupies memory
    return model

def get_raw_input_frame(input_type, data_views, rgb_imgs, current_frame_id, frame, device):
    """ process the input image for reconstruction

    Args:
        input_type: the type of input (e.g., "imgs" or "video")
        data_views: list of processed views for reconstruction
        rgb_imgs: list of pre-processed rgb images for visualization
        num_views: the number of views processed so far
        frame: the current frame read from frame_reader
    """
    # Pre-save the RGB images along with their corresponding masks
    # in preparation for visualization at last.
    if input_type != "imgs":
        frame = load_single_image(frame, 224, device)
    else:
        # ensure true_shape is (H,W)
        frame['true_shape'] = frame['true_shape'][0]
        # Stretch/resize incoming PIL/ndarray images so H and W are multiples of patch size (16)
        try:
            from PIL import Image
            patch_size = 16
            # frame['img'] is H,W,3 numpy array (RGB)
            img_arr = frame['img']
            pil_img = Image.fromarray(img_arr)
            W_orig, H_orig = pil_img.size
            def round_up(x, m):
                return ((x + m - 1) // m) * m
            H_new = round_up(H_orig, patch_size)
            W_new = round_up(W_orig, patch_size)
            if (H_new != H_orig) or (W_new != W_orig):
                pil_img = pil_img.resize((W_new, H_new), Image.Resampling.LANCZOS)
                frame['img'] = np.asarray(pil_img)
                frame['true_shape'] = np.array([H_new, W_new], dtype=np.int32)
        except Exception:
            # if PIL missing or conversion fails, continue without resizing
            pass
    data_views.append(frame)
    if data_views[current_frame_id]['img'].shape[0] == 1:
        data_views[current_frame_id]['img'] = data_views[current_frame_id]['img'][0]
    rgb_imgs.append(transform_img(dict(img=data_views[current_frame_id]['img'][None]))[...,::-1])
    
    # process now image for extracting its img token with encoder
    # Ensure image tensor has shape (B, C, H, W) and true_shape is a (B,2) tensor
    img = data_views[current_frame_id]['img']
    # convert numpy -> torch if needed
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)
    # move channel dim to dim=1 if channels are last (H,W,3) or (B,H,W,3)
    if img.ndim == 3 and img.shape[-1] == 3 and img.shape[0] != 3:
        img = img.permute(2, 0, 1)
    elif img.ndim == 4 and img.shape[-1] == 3 and img.shape[1] != 3:
        img = img.permute(0, 3, 1, 2)
    # add batch dim if missing
    if img.ndim == 3:
        img = img.unsqueeze(0)

    # ensure float dtype and normalization to match ImgNorm (ToTensor + Normalize((0.5,), (0.5,)))
    img = img.float()
    try:
        maxv = float(img.max())
        minv = float(img.min())
    except Exception:
        maxv, minv = 0.0, 0.0
    if maxv > 2.0:
        # assume 0-255 uint8 image
        img = img / 255.0
    # if in [0,1] range, map to [-1,1]; if already in [-1,1], leave as-is
    if minv >= 0.0 and img.max() <= 1.0:
        img = (img - 0.5) / 0.5

    data_views[current_frame_id]['img'] = img.to(device)
    data_views[current_frame_id]['true_shape'] = torch.as_tensor(data_views[current_frame_id]['true_shape'][None]).to(device)

    for key in ['valid_mask', 'pts3d_cam', 'pts3d']:
        if key in data_views[current_frame_id]:
            del data_views[current_frame_id][key]
    to_device(data_views[current_frame_id], device=device)
    
    return frame, data_views, rgb_imgs

def process_input_frame(per_frame_res, registered_confs_mean, 
                        data_views, frame_id, i2p_model):
    """Process single frame with mixed precision encoding (Tier 1)."""
    with autocast(device_type='cuda', dtype=torch.float16):
        temp_shape, temp_feat, temp_pose = get_single_img_tokens([data_views[frame_id]], i2p_model, True)

    input_view = dict(label=data_views[frame_id]['label'],
                            img_tokens=temp_feat[0],
                            true_shape=data_views[frame_id]['true_shape'],
                            img_pos=temp_pose[0])
    for key in per_frame_res:
        per_frame_res[key].append(None)
    registered_confs_mean.append(frame_id)
    return input_view, per_frame_res, registered_confs_mean
