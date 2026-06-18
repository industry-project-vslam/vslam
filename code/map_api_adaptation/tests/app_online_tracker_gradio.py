"""
Gradio app for SLAM3R online tracker interaction.

This app saves uploaded frames into a temporary image folder and reruns
`online_tracker.py`'s online reconstruction pipeline over all collected frames
whenever a new frame is added.

Usage:
    python app_online_tracker_gradio.py
"""

import json
import os
import shutil
from argparse import Namespace
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image
import time
import cv2

from slam3r.models import Image2PointsModel, Local2WorldModel
from slam3r.pipeline.recon_online_pipeline import (
    save_recon,
    get_raw_input_frame,
    process_input_frame,
    initial_scene_for_accumulated_frames,
    recover_points_in_initial_window,
    select_ids_as_reference,
    pointmap_local_recon,
    pointmap_global_register,
    update_buffer_set,
    estimate_camera_intrinsics_from_frames,
    estimate_pose_from_pcd,
    estimate_camera_pose_from_correspondences,
    is_valid_position_jump,
)
from slam3r.utils.device import to_numpy


DEFAULT_I2P_MODEL = (
    "Image2PointsModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', "
    "output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), "
    "enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, "
    "dec_depth=12, dec_num_heads=12, mv_dec1='MultiviewDecoderBlock_max', "
    "mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 11)"
)

DEFAULT_L2W_MODEL = (
    "Local2WorldModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', "
    "output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), "
    "enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, "
    "dec_depth=12, dec_num_heads=12, mv_dec1='MultiviewDecoderBlock_max', "
    "mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 11, need_encoder=False)"
)


def load_model(model_name: str, weights: str, device: str = 'cuda'):
    print(f'Loading model: {model_name}')
    model = eval(model_name)
    model.to(device)
    print('Loading pretrained: ', weights)
    ckpt = torch.load(weights, map_location=device)
    print(model.load_state_dict(ckpt['model'], strict=False))
    del ckpt
    return model


class OnlineTrackerApp:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.frames_dir = Path('tmp_online_tracker_frames')
        self.results_dir = Path('results_online_gradio')
        self.camera_name = 'cam0'
        self.args = self._build_args()
        self.i2p_model = None
        self.l2w_model = None
        self.trajectory = None
        self.frame_paths = []
        self.last_status = ''
        self.data_views = []
        self.rgb_imgs = []
        self.input_views = []
        self.per_frame_res = {
            'i2p_pcds': [],
            'i2p_confs': [],
            'l2w_pcds': [],
            'l2w_confs': [],
        }
        self.registered_confs_mean = []
        self.local_confs_mean_up2now = []
        self.buffering_set_ids = []
        self.last_ref_ids_buffer = []
        self.fail_view = {}
        self.milestone = 0
        self.candi_frame_id = 0
        self.prev_valid_position = None
        self.mean_intrinsics = None
        self.num_frame_read = 0
        self.num_frame_pass = 0
        self.scene_id = f'gradio_{self.camera_name}'
        self.save_dir = self.results_dir / self.camera_name
        self._initialize_dirs()
        self._load_models()

    def _build_args(self) -> Namespace:
        args = Namespace()
        args.win_r = 1
        args.initial_winsize = 2
        args.keyframe_stride = 1
        args.num_scene_frame = 2
        args.retrieve_freq = 1
        args.update_buffer_intv = 1
        args.buffer_size = 10
        args.buffer_strategy = 'reservoir'
        args.conf_thres_i2p = 1.5
        args.conf_thres_l2w = 3.0
        args.num_points_save = 1000000
        args.perframe = 1
        args.save_each_frame = True
        args.save_all_views = False
        args.save_preds = False
        args.save_for_eval = False
        args.norm_input = False
        args.device = self.device
        args.test_name = 'gradio_online'
        args.camera_name = self.camera_name
        args.seed = 11
        args.gpu_id = 0
        return args

    def _initialize_dirs(self):
        shutil.rmtree(self.frames_dir, ignore_errors=True)
        shutil.rmtree(self.results_dir, ignore_errors=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_models(self):
        if self.i2p_model is None:
            try:
                self.i2p_model = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
                self.l2w_model = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
            except Exception as e:
                raise RuntimeError(
                    'Failed to load pretrained models. ' \
                    'Please ensure the model checkpoint paths are available or install required packages. ' \
                    f'Error: {e}'
                )
            self.i2p_model.to(self.device)
            self.l2w_model.to(self.device)
            self.i2p_model.eval()
            self.l2w_model.eval()

    def reset(self):
        self.camera_name = 'cam0'
        self.args.camera_name = self.camera_name
        self.args.test_name = 'gradio_online'
        self.frame_paths = []
        self.trajectory = None
        self.last_status = ''
        self.data_views = []
        self.rgb_imgs = []
        self.input_views = []
        self.per_frame_res = {
            'i2p_pcds': [],
            'i2p_confs': [],
            'l2w_pcds': [],
            'l2w_confs': [],
        }
        self.registered_confs_mean = []
        self.local_confs_mean_up2now = []
        self.buffering_set_ids = []
        self.last_ref_ids_buffer = []
        self.fail_view = {}
        self.milestone = 0
        self.candi_frame_id = 0
        self.prev_valid_position = None
        self.mean_intrinsics = None
        self.num_frame_read = 0
        self.num_frame_pass = 0
        self.scene_id = f'gradio_{self.camera_name}'
        self.save_dir = self.results_dir / self.camera_name
        self._initialize_dirs()

    def add_frame(self, camera_name: str, image_input) -> tuple[str, go.Figure]:
        if image_input is None:
            self.last_status = 'Error: No frame uploaded.'
            return self.last_status, self._empty_plot()

        camera_name = camera_name.strip() if camera_name and camera_name.strip() else 'cam0'
        if self.camera_name != camera_name and self.frame_paths:
            self.reset()
        self.camera_name = camera_name
        self.args.camera_name = camera_name
        self.args.test_name = f'gradio_{camera_name}'
        self.scene_id = f'gradio_{camera_name}'
        self.save_dir = self.results_dir / self.camera_name

        try:
            frame = self._convert_image(image_input)
        except Exception as e:
            self.last_status = f'Error: Failed to parse uploaded frame: {e}'
            return self.last_status, self._empty_plot()

        frame_name = f'frame_{len(self.frame_paths):04d}.png'
        frame_path = self.frames_dir / frame_name
        frame.save(frame_path)
        self.frame_paths.append(frame_path)

        self.last_status, fig = self._process_new_frame(frame, frame_name)
        return self.last_status, fig

    def _ensure_trajectory(self):
        if self.trajectory is None or self.camera_name not in self.trajectory:
            self.trajectory = {
                'metadata': {'cameras': [self.camera_name], 'num_frames': 0},
                self.camera_name: {'frames': []},
            }

    def _save_current_frame_results(self):
        os.makedirs(self.save_dir, exist_ok=True)
        try:
            save_recon(
                self.input_views,
                self.num_frame_read,
                str(self.save_dir),
                self.scene_id,
                self.args.save_all_views,
                self.rgb_imgs,
                registered_confs=self.per_frame_res['l2w_confs'],
                num_points_save=self.args.num_points_save,
                conf_thres_res=self.args.conf_thres_l2w,
            )
        except Exception as e:
            print(f'Warning: failed to save current frame recon: {e}')

    def _write_trajectory_file(self):
        self._ensure_trajectory()
        self.trajectory['metadata']['num_frames'] = len(self.trajectory[self.camera_name]['frames'])
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.save_dir / 'trajectories.json', 'w') as f:
            json.dump(self.trajectory, f, indent=2)

    def _process_new_frame(self, pil_image: Image.Image, frame_label: str) -> tuple[str, go.Figure]:
        current_frame_id = self.num_frame_read
        frame = {
            'img': np.asarray(pil_image.convert('RGB')),
            'true_shape': np.array([pil_image.size[::-1]], dtype=np.int32),
            'label': frame_label,
        }

        _, self.data_views, self.rgb_imgs = get_raw_input_frame(
            'imgs', self.data_views, self.rgb_imgs, current_frame_id, frame, self.device
        )
        input_view, self.per_frame_res, self.registered_confs_mean = process_input_frame(
            self.per_frame_res,
            self.registered_confs_mean,
            self.data_views,
            current_frame_id,
            self.i2p_model,
        )
        self.input_views.append(input_view)
        self.num_frame_read += 1
        self.num_frame_pass += 1

        init_threshold = (self.args.initial_winsize - 1) * self.args.keyframe_stride
        if current_frame_id < init_threshold:
            status = (
                f'Added frame {len(self.frame_paths)}. Waiting for '
                f'{self.args.initial_winsize * self.args.keyframe_stride} frames to initialize the online tracker.'
            )
            return status, self._empty_plot()

        if current_frame_id == init_threshold:
            self.buffering_set_ids, self.init_ref_id, self.init_num, self.input_views, self.per_frame_res, self.registered_confs_mean = initial_scene_for_accumulated_frames(
                self.input_views,
                self.args.initial_winsize,
                self.args.keyframe_stride,
                self.i2p_model,
                self.per_frame_res,
                self.registered_confs_mean,
                self.args.buffer_size,
                self.args.conf_thres_i2p,
            )
            self.local_confs_mean_up2now, self.per_frame_res, self.input_views = recover_points_in_initial_window(
                current_frame_id,
                self.buffering_set_ids,
                self.args.keyframe_stride,
                self.init_ref_id,
                self.per_frame_res,
                self.input_views,
                self.i2p_model,
                self.args.conf_thres_i2p,
            )
            self.milestone = self.init_num * self.args.keyframe_stride + 1
            self.candi_frame_id = len(self.buffering_set_ids)
            if self.args.save_each_frame:
                self._save_current_frame_results()
            status = f'Initialized scene with {self.init_num} frames. Processed frame {current_frame_id}.'
            return status, self._build_trajectory_plot()

        ref_ids, self.last_ref_ids_buffer = select_ids_as_reference(
            self.buffering_set_ids,
            current_frame_id,
            self.input_views,
            self.i2p_model,
            self.args.num_scene_frame,
            self.args.win_r,
            self.args.keyframe_stride,
            self.args.retrieve_freq,
            self.last_ref_ids_buffer,
        )
        self.local_confs_mean_up2now, self.per_frame_res, self.input_views = pointmap_local_recon(
            [self.input_views[current_frame_id]] + [self.input_views[id] for id in ref_ids],
            self.i2p_model,
            current_frame_id,
            0,
            self.per_frame_res,
            self.input_views,
            self.args.conf_thres_i2p,
            self.local_confs_mean_up2now,
        )
        self.input_views, self.per_frame_res, self.registered_confs_mean = pointmap_global_register(
            [self.input_views[id] for id in ref_ids],
            self.input_views,
            self.l2w_model,
            self.per_frame_res,
            self.registered_confs_mean,
            current_frame_id,
            device=self.device,
            norm_input=self.args.norm_input,
        )

        next_frame_id = current_frame_id + 1
        if next_frame_id - self.milestone >= self.args.update_buffer_intv * self.args.keyframe_stride:
            self.milestone, self.candi_frame_id, self.buffering_set_ids = update_buffer_set(
                next_frame_id,
                self.args.buffer_size,
                self.args.keyframe_stride,
                self.buffering_set_ids,
                self.args.buffer_strategy,
                self.registered_confs_mean,
                self.local_confs_mean_up2now,
                self.candi_frame_id,
                self.milestone,
            )

        conf = self.registered_confs_mean[current_frame_id]
        if isinstance(conf, torch.Tensor):
            conf = float(conf.cpu())
        if conf < 10:
            self.fail_view[current_frame_id] = conf

        if self.args.save_each_frame:
            self._save_current_frame_results()

        pose_mat = None
        forward = [0.0, 0.0, 0.0]
        pos = [0.0, 0.0, 0.0]
        axes = None
        valid_pose = False
        try:
            if 'pts3d_world' in self.input_views[current_frame_id] and self.input_views[current_frame_id]['pts3d_world'] is not None:
                pts3d_world = self.input_views[current_frame_id]['pts3d_world']
                pts3d_cam = self.input_views[current_frame_id].get('pts3d_cam')
                if self.mean_intrinsics is None:
                    recent_views = self.input_views[max(0, current_frame_id-10):current_frame_id+1]
                    self.mean_intrinsics = estimate_camera_intrinsics_from_frames(recent_views, torch.tensor((224//2, 224//2)))
                if pts3d_cam is not None:
                    if hasattr(pts3d_world, 'shape') and len(pts3d_world.shape) == 4 and pts3d_world.shape[0] == 1:
                        pts3d_world = pts3d_world[0]
                    if hasattr(pts3d_cam, 'shape') and len(pts3d_cam.shape) == 4 and pts3d_cam.shape[0] == 1:
                        pts3d_cam = pts3d_cam[0]
                    c2w, pose_success = estimate_camera_pose_from_correspondences(pts3d_cam, pts3d_world)
                else:
                    c2w, pose_success = None, False
                if not pose_success and self.mean_intrinsics is not None:
                    c2w, pose_success = estimate_pose_from_pcd(pts3d_world, self.mean_intrinsics)
                if pose_success and c2w is not None:
                    pose_mat = c2w
                    pos = pose_mat[:3, 3].tolist()
                    x_axis = pose_mat[:3, 0]
                    y_axis = pose_mat[:3, 1]
                    z_axis = pose_mat[:3, 2]
                    forward = z_axis.tolist()
                    norm = np.linalg.norm(forward)
                    if norm > 1e-6:
                        forward = (np.array(forward) / norm).tolist()
                    axes = [x_axis.tolist(), y_axis.tolist(), z_axis.tolist()]
                    valid_pose = True
                    self.prev_valid_position = pos
            elif 'c2w' in self.input_views[current_frame_id] and self.input_views[current_frame_id]['c2w'] is not None:
                pose_mat = to_numpy(self.input_views[current_frame_id]['c2w']).reshape(4, 4)
                pos = pose_mat[:3, 3].tolist()
                x_axis = pose_mat[:3, 0]
                y_axis = pose_mat[:3, 1]
                z_axis = pose_mat[:3, 2]
                forward = z_axis.tolist()
                norm = np.linalg.norm(forward)
                if norm > 1e-6:
                    forward = (np.array(forward) / norm).tolist()
                axes = [x_axis.tolist(), y_axis.tolist(), z_axis.tolist()]
                valid_pose = True
                self.prev_valid_position = pos
            else:
                if 'pts3d_cam' in self.input_views[current_frame_id]:
                    pts3d = self.input_views[current_frame_id]['pts3d_cam']
                    if pts3d is not None:
                        if self.mean_intrinsics is None:
                            recent_views = self.input_views[max(0, current_frame_id-10):current_frame_id+1]
                            self.mean_intrinsics = estimate_camera_intrinsics_from_frames(recent_views, torch.tensor((224//2, 224//2)))
                        if self.mean_intrinsics is not None:
                            c2w, pose_success = estimate_pose_from_pcd(pts3d, self.mean_intrinsics)
                            if pose_success and c2w is not None:
                                pose_mat = c2w
                                pos = pose_mat[:3, 3].tolist()
                                x_axis = pose_mat[:3, 0]
                                y_axis = pose_mat[:3, 1]
                                z_axis = pose_mat[:3, 2]
                                forward = z_axis.tolist()
                                norm_forward = np.linalg.norm(forward)
                                if norm_forward > 1e-6:
                                    forward = (np.array(forward) / norm_forward).tolist()
                                axes = [x_axis.tolist(), y_axis.tolist(), z_axis.tolist()]
                                if is_valid_position_jump(pos, self.prev_valid_position, max_jump=1.0):
                                    valid_pose = True
                                    self.prev_valid_position = pos
                                else:
                                    print(f'Warning: frame {current_frame_id} has unrealistic position jump, marking as invalid')
                                    pos = [0.0, 0.0, 0.0]
                                    forward = [0.0, 0.0, 0.0]
                                    axes = None
        except Exception as e:
            print(f'Warning: failed to estimate pose for frame {current_frame_id}: {e}')

        self._ensure_trajectory()
        self.trajectory[self.camera_name]['frames'].append({
            'frame': current_frame_id,
            'valid': bool(valid_pose),
            'position': pos,
            'forward': forward,
            'axes': axes,
            'conf': float(conf),
            'timestamp': time.time(),
        })
        self._write_trajectory_file()

        # Always save the combined reconstruction/map after processing this frame
        try:
            self._save_current_frame_results()
        except Exception as e:
            print(f"Warning: failed to save current frame recon: {e}")

        status = f'Processed frame {current_frame_id}. Conf: {conf:.2f}.'
        return status, self._build_trajectory_plot()

    def _convert_image(self, image_input) -> Image.Image:
        if isinstance(image_input, str):
            return Image.open(image_input).convert('RGB')
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input.astype('uint8'), 'RGB')
        if isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        raise ValueError('Unsupported image input type')

    def _load_trajectory(self, save_dir: Path):
        traj_path = save_dir / 'trajectories.json'
        if not traj_path.exists():
            return None
        with open(traj_path, 'r') as f:
            return json.load(f)

    def _build_trajectory_plot(self) -> go.Figure:
        fig = go.Figure()
        if self.trajectory is None:
            return self._empty_plot()

        frames = self.trajectory.get(self.args.camera_name, {}).get('frames', [])
        if not frames:
            return self._empty_plot()

        positions = np.array([f['position'] for f in frames if f['valid']], dtype=np.float32)
        if positions.size == 0:
            return self._empty_plot()

        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers+lines',
            marker=dict(size=4, color='black'),
            line=dict(color='gray', width=2),
            name=self.args.camera_name,
        ))

        scale = 0.1
        for frame in frames:
            if not frame.get('valid'):
                continue
            axes = frame.get('axes')
            if not axes:
                continue
            pos = np.array(frame['position'], dtype=np.float32)
            for axis, color, label in zip(axes, ['red', 'green', 'blue'], ['x', 'y', 'z']):
                vec = np.array(axis, dtype=np.float32)
                if np.linalg.norm(vec) < 1e-6:
                    continue
                vec = vec / np.linalg.norm(vec) * scale
                fig.add_trace(go.Scatter3d(
                    x=[pos[0], pos[0] + vec[0]],
                    y=[pos[1], pos[1] + vec[1]],
                    z=[pos[2], pos[2] + vec[2]],
                    mode='lines',
                    line=dict(color=color, width=4),
                    showlegend=False,
                    hoverinfo='skip',
                ))
        fig.update_layout(
            title='Online Tracker Trajectory',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
            ),
            width=1000,
            height=700,
        )
        return fig

    def get_status_text(self) -> str:
        if self.trajectory is None:
            if len(self.frame_paths) < self.args.initial_winsize * self.args.keyframe_stride:
                return (
                    f'Camera: {self.camera_name}\n'
                    f'Frames stored: {len(self.frame_paths)}\n'
                    f'Waiting for {self.args.initial_winsize * self.args.keyframe_stride} frames before running the online tracker.'
                )
            return (
                f'Camera: {self.camera_name}\n'
                f'Frames stored: {len(self.frame_paths)}\n'
                f'Enough frames collected. Online tracker may have run but did not produce a valid trajectory.\n'
                f'Last status: {self.last_status}'
            )

        frames = self.trajectory.get(self.args.camera_name, {}).get('frames', [])
        valid = sum(1 for f in frames if f.get('valid'))
        total = len(frames)
        return (
            f'Camera: {self.camera_name}\n'
            f'Trajectory frames: {total}\n'
            f'Valid poses: {valid}\n'
            f'Save dir: {self.results_dir / self.args.camera_name}'
        )

    def _empty_plot(self) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title='Trajectory will appear after online tracker initialization',
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data',
            ),
            width=1000,
            height=700,
        )
        return fig


app = OnlineTrackerApp()


def on_add_frame(camera_name: str, image_input, conf_thres: float):
    # apply confidence threshold from UI
    try:
        app.args.conf_thres_l2w = float(conf_thres)
    except Exception:
        pass
    return app.add_frame(camera_name, image_input)


def on_process_video(camera_name: str, video_file, frame_skip: int, conf_thres: float):
    """Process uploaded video, extract frames with given skip and feed to app.add_frame.
    This is a generator that yields (status, plot) after each processed frame for live updates.
    """
    # apply confidence threshold from UI
    try:
        app.args.conf_thres_l2w = float(conf_thres)
    except Exception:
        pass

    if video_file is None:
        yield 'Error: No video uploaded.', app._empty_plot()
        return

    # video_file can be a tempfile path (str) from Gradio
    video_path = video_file if isinstance(video_file, str) else getattr(video_file, 'name', None)
    if video_path is None:
        yield 'Error: Cannot read uploaded video file.', app._empty_plot()
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield f'Error: cannot open video {video_path}', app._empty_plot()
        return

    frame_idx = 0
    processed = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % max(1, frame_skip) != 0:
            frame_idx += 1
            continue

        # frame is BGR numpy array; convert to RGB before passing (app expects RGB numpy)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            status, fig = app.add_frame(camera_name, rgb)
        except Exception as e:
            yield f'Error processing frame {frame_idx}: {e}', app._empty_plot()
            cap.release()
            return

        processed += 1
        frame_idx += 1
        # include progress in status
        if total_frames:
            prog = f'Processed {processed} frames (frame {frame_idx}/{total_frames})'
        else:
            prog = f'Processed {processed} frames (frame {frame_idx})'
        yield prog + ' - ' + status, fig

    cap.release()
    yield f'Finished processing video. Total frames processed: {processed}', app._build_trajectory_plot()


def on_reset():
    app.reset()
    return 'Reset complete.', app._empty_plot()


def build_gradio_interface():
    with gr.Blocks(title='SLAM3R Online Tracker Gradio') as demo:
        gr.Markdown('# SLAM3R Online Tracker (One Frame at a Time)')
        gr.Markdown(
            'Upload a frame one-by-one. Frames are accumulated in a temporary directory, '
            'and the online tracker pipeline is rerun when enough frames are available.'
        )

        with gr.Row():
            with gr.Column(scale=1):
                camera_name = gr.Textbox(label='Camera Name', value='cam0')
                frame_upload = gr.Image(label='Upload Frame', type='numpy')
                gr.Markdown('---')
                gr.Markdown('### Video mode')
                video_upload = gr.File(label='Upload Video (MP4, AVI, ... )')
                frame_skip = gr.Slider(label='Frame skip (process every Nth frame)', minimum=1, maximum=30, step=1, value=5)
                conf_slider = gr.Slider(label='L2W confidence threshold', minimum=0.0, maximum=20.0, step=0.1, value=3.0)
                add_button = gr.Button('Add Frame', variant='primary')
                process_video_button = gr.Button('Process Video', variant='primary')
                reset_button = gr.Button('Reset', variant='stop')
                status_box = gr.Textbox(label='Status', interactive=False, lines=6)

            with gr.Column(scale=2):
                gr.Markdown('### Trajectory Output')
                trajectory_plot = gr.Plot(label='Trajectory')

        add_button.click(
            fn=on_add_frame,
            inputs=[camera_name, frame_upload, conf_slider],
            outputs=[status_box, trajectory_plot],
        )

        process_video_button.click(
            fn=on_process_video,
            inputs=[camera_name, video_upload, frame_skip, conf_slider],
            outputs=[status_box, trajectory_plot],
        )

        reset_button.click(
            fn=on_reset,
            inputs=[],
            outputs=[status_box, trajectory_plot],
        )

    return demo


if __name__ == '__main__':
    build_gradio_interface().launch(server_name='0.0.0.0', server_port=7860, share=False)
