import argparse
import os
import json
import time
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import torch
import numpy as np

from slam3r.models import Image2PointsModel, Local2WorldModel
from slam3r.pipeline.recon_online_pipeline import scene_recon_pipeline_online, FrameReader
from tests.incremental_loop_closure import IncrementalLoopClosureDetector


def load_model(model_name: str, weights: str, device: str = 'cuda'):
    """Create a model instance and load a local checkpoint."""
    print(f'Loading model: {model_name}')
    model = eval(model_name)
    model.to(device)
    print('Loading pretrained: ', weights)
    ckpt = torch.load(weights, map_location=device)
    print(model.load_state_dict(ckpt['model'], strict=False))
    del ckpt
    return model


class WebcamFrameReader:
    """A minimal webcam frame reader compatible with the SLAM3R online pipeline."""
    def __init__(self, cam_index: int):
        self.type = 'video'
        self.video_capture = cv2.VideoCapture(cam_index)
        if not self.video_capture.isOpened():
            raise RuntimeError(f'Cannot open webcam {cam_index}')

    def read(self):
        """Read a frame from the webcam."""
        return self.video_capture.read()


def run_online_source(source: Any, cam_id: str, args: argparse.Namespace):
    """Run the online pipeline for a single source and save results."""
    save_dir = os.path.join(args.output_dir, f'{args.test_name}_{cam_id}')
    os.makedirs(save_dir, exist_ok=True)
    print(f'\n=== Running online reconstruction for {cam_id} ===')
    print(f'Source: {source}')
    
    # expose camera id to the pipeline so it can label trajectories
    args.camera_name = cam_id
    
    # Run the online reconstruction pipeline
    scene_recon_pipeline_online(args.i2p_model_obj, args.l2w_model_obj, source, args, save_dir)
    
    print(f'Finished {cam_id}. Results saved to {save_dir}')
    
    # Post-process with loop closure detection if enabled
    if args.enable_loop_closure:
        print(f'\n=== Refining trajectory with loop closure detection ===')
        refine_trajectory_with_loops(save_dir, cam_id, args)


def refine_trajectory_with_loops(save_dir: str, cam_id: str, args: argparse.Namespace):
    """
    Refine trajectory using loop closure detection and pose graph optimization.
    
    Args:
        save_dir: Directory containing frames and trajectories.json
        cam_id: Camera ID
        args: Command line arguments
    """
    save_dir = Path(save_dir)
    traj_file = save_dir / 'trajectories.json'
    
    if not traj_file.exists():
        print(f"Trajectory file not found: {traj_file}")
        return
    
    # Load original trajectory
    with open(traj_file, 'r') as f:
        data = json.load(f)
    
    frames = data.get(cam_id, {}).get('frames', [])
    if not frames:
        print(f"No frames found for camera {cam_id}")
        return
    
    # Create loop closure detector
    detector = IncrementalLoopClosureDetector(
        image_dir=save_dir,
        min_frame_gap=args.loop_min_frame_gap,
        match_threshold=args.loop_match_threshold,
        min_matches=args.loop_min_matches,
        temporal_weight=args.loop_temporal_weight,
        loop_weight=args.loop_weight,
        optimize_every_n_frames=args.loop_optimize_every_n,
    )
    
    # Load images and extract features for all frames
    print(f"Loading {len(frames)} frames for loop closure detection...")
    
    # Build frame_num -> image path mapping
    frame_image_map = {}
    for ext in ['.jpg', '.png', '.JPG', '.PNG']:
        for img_path in sorted(save_dir.glob(f"*{ext}")):
            # Try to extract frame number from filename
            stem = img_path.stem
            if stem.isdigit():
                frame_image_map[int(stem)] = img_path
            else:
                # Try regex extraction
                import re
                match = re.search(r'(\d+)$', stem)
                if match:
                    frame_image_map[int(match.group(1))] = img_path
    
    for i, frame_info in enumerate(frames):
        frame_num = frame_info['frame']
        position = np.array(frame_info['position'])
        forward = np.array(frame_info['forward'])
        
        # Load frame image
        image = None
        if frame_num in frame_image_map:
            img_path = frame_image_map[frame_num]
            image = cv2.imread(str(img_path))
        
        if image is None:
            if i < 5:  # Only warn for first few
                print(f"  Warning: Image for frame {frame_num} not found")
            continue
        
        # Process frame (extract features, detect loops)
        detector.process_frame(frame_num, image, position, forward)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(frames)} frames")
    
    # Final optimization
    print("Running final optimization...")
    detector.optimize()
    
    # Get loop closures info
    loop_closures = detector.get_loop_closures()
    print(f"Found {len(loop_closures)} loop closures")
    if loop_closures:
        for frame_i, frame_j, score in loop_closures[:10]:  # Show first 10
            print(f"  Loop: frame {frame_i} ↔ {frame_j} (score: {score:.3f})")
        if len(loop_closures) > 10:
            print(f"  ... and {len(loop_closures) - 10} more")
    
    # Update trajectory file with refined positions
    detector.update_trajectory_json(traj_file)
    print(f"Refined trajectory saved to {traj_file}")


def main():
    """Parse arguments and run online reconstruction for all configured sources."""
    parser = argparse.ArgumentParser(description='Online SLAM3R tracking using recon.py pipeline')
    parser.add_argument(
        '--cam_dirs', nargs='+', default=None,
        help='Camera source folders for testing mode'
    )
    parser.add_argument(
        '--webcams', nargs='+', type=int, default=None,
        help='Webcam indices for live mode'
    )
    parser.add_argument(
        '--output_dir', default='results_online/',
        help='Output directory'
    )
    parser.add_argument(
        '--device', default='cuda',
        help='Device: cuda or cpu'
    )
    parser.add_argument(
        '--test_name', required=True,
        help='Name of the test'
    )
    parser.add_argument(
        '--i2p_weights', type=str,
        help='Path to the weights of i2p model'
    )
    parser.add_argument(
        '--l2w_weights', type=str,
        help='Path to the weights of l2w model'
    )
    parser.add_argument(
        '--i2p_model',
        type=str,
        default=(
            "Image2PointsModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', "
            "output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), "
            "enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, "
            "dec_depth=12, dec_num_heads=12, mv_dec1='MultiviewDecoderBlock_max', "
            "mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 11)"
        ),
        help='I2P model constructor string'
    )
    parser.add_argument(
        '--l2w_model',
        type=str,
        default=(
            "Local2WorldModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', "
            "output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), "
            "enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, "
            "dec_depth=12, dec_num_heads=12, mv_dec1='MultiviewDecoderBlock_max', "
            "mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 11, need_encoder=False)"
        ),
        help='L2W model constructor string'
    )
    parser.add_argument(
        '--keyframe_stride', type=int, default=3,
        help='Stride of sampling keyframes'
    )
    parser.add_argument(
        '--initial_winsize', type=int, default=5,
        help='Number of initial frames to use'
    )
    parser.add_argument(
        '--win_r', type=int, default=3,
        help='Radius of the input window for I2P model'
    )
    parser.add_argument(
        '--conf_thres_i2p', type=float, default=1.5,
        help='Confidence threshold for I2P'
    )
    parser.add_argument(
        '--num_scene_frame', type=int, default=10,
        help='Number of scene frames selected as reference'
    )
    parser.add_argument(
        '--max_num_register', type=int, default=10,
        help='Max frames registered in one go'
    )
    parser.add_argument(
        '--conf_thres_l2w', type=float, default=12,
        help='Confidence threshold for L2W save'
    )
    parser.add_argument(
        '--num_points_save', type=int, default=2000000,
        help='Number of points to save'
    )
    parser.add_argument(
        '--norm_input', action='store_true',
        help='Whether to normalize input pointmaps for L2W'
    )
    parser.add_argument(
        '--save_frequency', type=int, default=3,
        help='Per xxx frames to save'
    )
    parser.add_argument(
        '--save_each_frame', action='store_true', default=True,
        help='Whether to save each frame to .ply'
    )
    parser.add_argument(
        '--retrieve_freq', type=int, default=1,
        help='Frequency of retrieving reference frames (online only)'
    )
    parser.add_argument(
        '--update_buffer_intv', type=int, default=1,
        help='Interval of updating buffering set'
    )
    parser.add_argument(
        '--buffer_size', type=int, default=100,
        help='Max size of buffering set, -1 if infinite'
    )
    parser.add_argument(
        '--buffer_strategy', type=str,
        choices=['reservoir', 'fifo'],
        default='reservoir',
        help='Buffer maintenance strategy'
    )
    parser.add_argument(
        '--save_online', action='store_true',
        help='Whether to save the construct result online'
    )
    parser.add_argument(
        '--save_all_views', action='store_true',
        help='Save point clouds for each frame as separate PLY files'
    )
    parser.add_argument(
        '--save_preds', action='store_true',
        help='Save per-frame predictions for evaluation'
    )
    parser.add_argument(
        '--save_for_eval', action='store_true',
        help='Save predictions in evaluation format'
    )
    parser.add_argument(
        '--keyframe_adapt_min', type=int, default=1,
        help='Min keyframe stride for adaptation'
    )
    parser.add_argument(
        '--keyframe_adapt_max', type=int, default=20,
        help='Max keyframe stride for adaptation'
    )
    parser.add_argument(
        '--keyframe_adapt_stride', type=int, default=1,
        help='Stride for trying different keyframe stride'
    )
    parser.add_argument(
        '--perframe', type=int, default=1,
        help='Frame interval for online processing'
    )
    parser.add_argument(
        '--seed', type=int, default=11,
        help='Random seed'
    )
    parser.add_argument(
        '--gpu_id', type=int, default=-1,
        help='GPU id, -1 for auto select'
    )
    
    # Loop closure detection arguments
    parser.add_argument(
        '--enable_loop_closure', action='store_true',
        help='Enable loop closure detection and pose graph refinement'
    )
    parser.add_argument(
        '--loop_min_frame_gap', type=int, default=10,
        help='Minimum frames between loop closure candidates'
    )
    parser.add_argument(
        '--loop_match_threshold', type=float, default=0.7,
        help='Feature match quality threshold (Lowe ratio test, 0.0-1.0)'
    )
    parser.add_argument(
        '--loop_min_matches', type=int, default=20,
        help='Minimum feature matches to consider a loop closure'
    )
    parser.add_argument(
        '--loop_temporal_weight', type=float, default=1.0,
        help='Weight for consecutive frame temporal constraints'
    )
    parser.add_argument(
        '--loop_weight', type=float, default=5.0,
        help='Weight for loop closure constraints'
    )
    parser.add_argument(
        '--loop_optimize_every_n', type=int, default=5,
        help='Optimize pose graph every N frames'
    )

    args = parser.parse_args()
    if args.gpu_id == -1:
        args.gpu_id = 0
    torch.cuda.set_device(f'cuda:{args.gpu_id}')
    np.random.seed(args.seed)

    if args.i2p_weights is not None:
        args.i2p_model_obj = load_model(args.i2p_model, args.i2p_weights, args.device)
    else:
        args.i2p_model_obj = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
        args.i2p_model_obj.to(args.device)

    if args.l2w_weights is not None:
        args.l2w_model_obj = load_model(args.l2w_model, args.l2w_weights, args.device)
    else:
        args.l2w_model_obj = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
        args.l2w_model_obj.to(args.device)

    args.i2p_model_obj.eval()
    args.l2w_model_obj.eval()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    sources: List[Tuple[Any, str]] = []
    if args.cam_dirs:
        # Wrap folder paths with FrameReader so the online pipeline can call .read()
        sources = [(FrameReader(d), f'cam{i}') for i, d in enumerate(args.cam_dirs)]
    elif args.webcams:
        sources = [(WebcamFrameReader(w), f'cam{i}') for i, w in enumerate(args.webcams)]
    else:
        raise ValueError('Please provide --cam_dirs or --webcams')

    for source, cam_id in sources:
        run_online_source(source, cam_id, args)


if __name__ == '__main__':
    main()
