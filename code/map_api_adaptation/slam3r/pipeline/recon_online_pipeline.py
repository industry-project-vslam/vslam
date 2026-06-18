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

from slam3r.datasets.wild_seq import Seq_Data
from slam3r.models import Image2PointsModel, Local2WorldModel, inf
from slam3r.utils.device import to_numpy
from slam3r.utils.recon_utils import * 
from slam3r.datasets.get_webvideo import *
from slam3r.utils.image import load_single_image
from slam3r.pipeline.recon_offline_pipeline import scene_frame_retrieve

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

@torch.no_grad()
def get_img_tokens(views, model,silent=False):
    """get img tokens output from encoder,
    which can be reused by both i2p and l2w models
    """
    res_shapes, res_feats, res_poses = model._encode_multiview(views, 
                                                               view_batchsize=10, 
                                                               normalize=False,
                                                               silent=silent)
    return res_shapes, res_feats, res_poses

@torch.no_grad()
def get_single_img_tokens(views, model, silent=False):
    """get an img token output from encoder,
    which can be reused by both i2p and l2w models
    """
    res_shape, res_feat, res_poses = model._encode_multiview(views, 
                                                               view_batchsize=1, 
                                                               normalize=False,
                                                               silent=silent)
    return res_shape, res_feat, res_poses

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
    temp_shape, temp_feat, temp_pose = get_single_img_tokens([data_views[frame_id]], i2p_model, True)
    # print(f"finish pre-extracting img token of view {frame_id}")

    input_view = dict(label=data_views[frame_id]['label'],
                            img_tokens=temp_feat[0],
                            true_shape=data_views[frame_id]['true_shape'],
                            img_pos=temp_pose[0])
    for key in per_frame_res:
        per_frame_res[key].append(None)
    registered_confs_mean.append(frame_id)
    return input_view, per_frame_res, registered_confs_mean
    

def register_initial_window_frames(init_num, kf_stride, buffering_set_ids, 
                                   input_views, l2w_model, per_frame_res, 
                                   registered_confs_mean, device="cuda", norm_input=False):
    """
    initially register the frames within the initial window with L2W model
    """
    max_conf_mean = -1
    for view_id in tqdm(range((init_num - 1) * kf_stride), desc="pre-registering"):
        if view_id % kf_stride == 0:
            continue
        
        # construct the input for L2W model
        l2w_input_views = [input_views[view_id]] + [input_views[id] for id in buffering_set_ids]
        # (for defination of ref_ids, seee the doc of l2w_model)
        output = l2w_inference(l2w_input_views, l2w_model,
                                ref_ids=list(range(1,len(l2w_input_views))),
                                device=device,
                                normalize=norm_input)
        # process the output of L2W model
        input_views[view_id]['pts3d_world'] = output[0]['pts3d_in_other_view'] # 1,224,224,3
        conf_map = output[0]['conf'] # 1,224,224
        per_frame_res['l2w_confs'][view_id] = conf_map[0] # 224,224
        registered_confs_mean[view_id] = conf_map.mean().cpu()
        per_frame_res['l2w_pcds'][view_id] = input_views[view_id]['pts3d_world']
        
        if registered_confs_mean[view_id] > max_conf_mean:
            max_conf_mean = registered_confs_mean[view_id]
    print(f'finish aligning {(init_num)*kf_stride} head frames, with a max mean confidence of {max_conf_mean:.2f}')
    return max_conf_mean, input_views, per_frame_res

def initialize_scene(views:list, model:Image2PointsModel, winsize=5, conf_thres=5, return_ref_id=False):    
    """initialize the scene with the first several frames.
    Try to find the best window size and the best ref_id.
    """
    init_ref_id = 0
    max_med_conf = 0     
    window_views = views[:winsize]
    # traverse all views in the window to find the best ref_id
    for i in range(winsize):
        ref_id = i
        output = i2p_inference_batch([window_views], model, ref_id=ref_id, 
                                    tocpu=True, unsqueeze=False)
        preds = output['preds']
        # choose the ref_id with the highest median confidence
        med_conf = np.array([preds[j]['conf'].mean() for j in range(winsize)]).mean()
        if med_conf > max_med_conf:
            max_med_conf = med_conf
            init_ref_id = ref_id
    output = i2p_inference_batch([views[:winsize]], model, ref_id=init_ref_id, 
                                    tocpu=False, unsqueeze=False)
    initial_pcds = []
    initial_confs = []
    for j in range(winsize):
        if j == init_ref_id:
            initial_pcds.append(output['preds'][j]['pts3d'])
        else:
            initial_pcds.append(output['preds'][j]['pts3d_in_other_view'])
        initial_confs.append(output['preds'][j]['conf'])
    print(f'initialize scene with {winsize} views, with a mean confidence of {max_med_conf:.2f}')
    if return_ref_id:
        return initial_pcds, initial_confs, init_ref_id
    return initial_pcds, initial_confs

def select_ids_as_reference(buffering_set_ids, current_frame_id,
                            input_views, i2p_model, 
                            num_scene_frame, win_r, adj_distance, 
                            retrieve_freq, last_ref_ids_buffer):
    """select the ids of scene frames from the buffering set
    
    next_register_id: the id of the next view to be registered
    """

    # select sccene frames in the buffering set to work as a global reference
    cand_ref_ids = buffering_set_ids

    if current_frame_id % retrieve_freq == 0 or len(last_ref_ids_buffer) == 0:
        _, sel_pool_ids = scene_frame_retrieve(
            [input_views[i] for i in cand_ref_ids],
            input_views[current_frame_id: current_frame_id + 1],
            i2p_model, sel_num=num_scene_frame,
            depth = 2)
        if isinstance(sel_pool_ids, torch.Tensor):
            sel_pool_ids = sel_pool_ids.cpu().numpy().tolist()
        ref_ids = sel_pool_ids
    else:
        ref_ids = last_ref_ids_buffer
        sel_pool_ids = last_ref_ids_buffer

    # also add several adjacent frames to enhance the stability
    for j in range(1, win_r + 1):
        adj_frame_id = current_frame_id - j * adj_distance
        if adj_frame_id >= 0 and adj_frame_id not in ref_ids:
            ref_ids.append(current_frame_id - j * adj_distance)
    
    return ref_ids, sel_pool_ids

def initial_scene_for_accumulated_frames(input_views, initial_winsize, 
                                         kf_stride, i2p_model, per_frame_res, 
                                         registered_confs_mean, buffer_size, 
                                         conf_thres_i2p):
    """initialize the scene with the first several frames.
    Set up the world coordinates with the initial window.
    """
    initial_pcds, initial_confs, init_ref_id = initialize_scene(input_views[:initial_winsize*kf_stride:kf_stride],
                                                                i2p_model,winsize=initial_winsize,return_ref_id=True)
    # set up the world coordinates with the initial window
    init_num = len(initial_pcds)
    for j in range(init_num):
        per_frame_res['l2w_confs'][j * kf_stride] = initial_confs[j][0].cpu()
        registered_confs_mean[j * kf_stride] = per_frame_res['l2w_confs'][j * kf_stride].mean().cpu()
    # initialize the buffering set with the initial window
    assert buffer_size <= 0 or buffer_size >= init_num 
    buffering_set_ids = [j*kf_stride for j in range(init_num)]
    # set ip the woeld coordinates with frames in the initial window
    for j in range(init_num):
        input_views[j*kf_stride]['pts3d_world'] = initial_pcds[j]
    initial_valid_masks = [conf > conf_thres_i2p for conf in initial_confs]
    normed_pts = normalize_views([view['pts3d_world'] for view in input_views[:init_num*kf_stride:kf_stride]],
                                                initial_valid_masks)
    for j in range(init_num):
        input_views[j*kf_stride]['pts3d_world'] = normed_pts[j]
        # filter out points with low confidence
        input_views[j*kf_stride]['pts3d_world'][~initial_valid_masks[j]] = 0
        per_frame_res['l2w_pcds'][j*kf_stride] = normed_pts[j]
        
    return buffering_set_ids, init_ref_id, init_num, input_views, per_frame_res, registered_confs_mean

def recover_points_in_initial_window(window_last_id, buffering_set_ids, 
                                     kf_stride, init_ref_id, 
                                     per_frame_res, input_views, 
                                     i2p_model, conf_thres_i2p):
    """
    recover the points in their local coordinates for all frames up to now
    """

    for view_id in range(window_last_id + 1):
        # skip the views in the initial window(which is also in the buffering set)
        if view_id in buffering_set_ids:
            # trick to mark the keyframe in the initial window
            if view_id // kf_stride == init_ref_id:
                per_frame_res['i2p_pcds'][view_id] = per_frame_res['l2w_pcds'][view_id].cpu()
            else:
                per_frame_res['i2p_pcds'][view_id] = torch.zeros_like(per_frame_res['l2w_pcds'][view_id], device="cpu")
            per_frame_res['i2p_confs'][view_id] = per_frame_res['l2w_confs'][view_id].cpu()
            print(f"finish revocer pcd of frame {view_id} in their local coordinates(in buffer set), with a mean confidence of {per_frame_res['i2p_confs'][view_id].mean():.2f}.")
            continue
        # construct the local window with the initial views
        # sel_ids = [view_id]
        # for j in range(1, win_r + 1):
        #     if view_id - j * adj_distance >= 0:
        #         sel_ids.append(view_id - j * adj_distance)
        #     if view_id + j * adj_distance < window_last_id:
        #         sel_ids.append(view_id + j * adj_distance)
        sel_ids = [view_id] + buffering_set_ids
        local_views = [input_views[id] for id in sel_ids]
        ref_id = 0

        # recover poionts in the initial window, and save the keyframe points and confs
        output = i2p_inference_batch([local_views], i2p_model, ref_id=ref_id,
                                        tocpu=False, unsqueeze=False)['preds']
        # save results of the i2p model for the initial window
        per_frame_res['i2p_pcds'][view_id] = output[ref_id]['pts3d'].cpu()
        per_frame_res['i2p_confs'][view_id] = output[ref_id]['conf'][0].cpu()

        # construct the input for L2W model
        input_views[view_id]['pts3d_cam'] = output[ref_id]['pts3d']
        valid_mask = output[ref_id]['conf'] > conf_thres_i2p
        input_views[view_id]['pts3d_cam'] = normalize_views([input_views[view_id]['pts3d_cam']],
                                                                [valid_mask])[0]
        input_views[view_id]['pts3d_cam'][~valid_mask] = 0
        print(f"finish revocer pcd of frame {view_id} in their local coordinates, with a mean confidence of {per_frame_res['i2p_confs'][view_id].mean():.2f}.")
    local_confs_mean_up2now = [conf.mean() for conf in per_frame_res['i2p_confs'] if conf is not None]
    return local_confs_mean_up2now, per_frame_res, input_views

def pointmap_local_recon(local_views, i2p_model, 
                            current_frame_id, ref_id, 
                            per_frame_res, input_views, 
                            conf_thres_i2p, local_confs_mean_up2now):
    output = i2p_inference_batch([local_views], i2p_model, ref_id=ref_id,
                                    tocpu=False, unsqueeze=False)['preds']
    # save results of the i2p model for the initial window
    per_frame_res['i2p_pcds'][current_frame_id] = output[ref_id]['pts3d'].cpu()
    per_frame_res['i2p_confs'][current_frame_id] = output[ref_id]['conf'][0].cpu()

    # construct the input for L2W model
    input_views[current_frame_id]['pts3d_cam'] = output[ref_id]['pts3d']
    valid_mask = output[ref_id]['conf'] > conf_thres_i2p
    input_views[current_frame_id]['pts3d_cam'] = normalize_views([input_views[current_frame_id]['pts3d_cam']],
                                                            [valid_mask])[0]
    input_views[current_frame_id]['pts3d_cam'][~valid_mask] = 0

    local_confs_mean_up2now.append(per_frame_res['i2p_confs'][current_frame_id].mean())
    # print(f"finish revocer pcd of frame {current_frame_id} in their local coordinates")
    return local_confs_mean_up2now, per_frame_res, input_views, 

def pointmap_global_register(ref_views, input_views, l2w_model,
                             per_frame_res, registered_confs_mean, 
                             current_frame_id, device="cuda", norm_input=False):

    view_to_register = input_views[current_frame_id]
    l2w_input_views = ref_views + [view_to_register]
    
    output = l2w_inference(l2w_input_views, l2w_model,
                            ref_ids=list(range(len(ref_views))),
                            device=device,
                            normalize=norm_input)
    
    conf_map = output[-1]['conf'] # 1,224,224

    input_views[current_frame_id]['pts3d_world'] = output[-1]['pts3d_in_other_view'] # 1,224,224,3
    per_frame_res['l2w_confs'][current_frame_id] = conf_map[0]
    registered_confs_mean[current_frame_id] = conf_map[0].mean().cpu()
    per_frame_res['l2w_pcds'][current_frame_id] = input_views[current_frame_id]['pts3d_world']
    # store estimated camera-to-world pose (if available) for this source view
    if isinstance(output[-1], dict) and 'c2w' in output[-1] and output[-1]['c2w'] is not None:
        input_views[current_frame_id]['c2w'] = output[-1]['c2w']
    
    return input_views, per_frame_res, registered_confs_mean

def update_buffer_set(next_register_id, max_buffer_size, 
                      kf_stride, buffering_set_ids, strategy, 
                      registered_confs_mean, local_confs_mean_up2now, 
                      candi_frame_id, milestone):
    """Update the buffer set with the newly registered views.

    Args:
        next_register_id: the id of the next view to be registered
        buffering_set_ids: used for buffering the registered views
        strategy: used for selecting the views to be buffered
        candi_frame_id: used for reservoir sampling
    """
    while(next_register_id - milestone >= kf_stride):
        candi_frame_id += 1
        full_flag = max_buffer_size > 0 and len(buffering_set_ids) >= max_buffer_size
        insert_flag = (not full_flag) or ((strategy == 'fifo') or 
                                        (strategy == 'reservoir' and np.random.rand() < max_buffer_size/candi_frame_id))
        if not insert_flag: 
            milestone += kf_stride
            continue
        # Use offest to ensure the selected view is not too close to the last selected view
        # If the last selected view is 0, 
        # the next selected view should be at least kf_stride*3//4 frames away
        start_ids_offset = max(0, buffering_set_ids[-1]+kf_stride*3//4 - milestone)
            
        # get the mean confidence of the candidate views
        mean_cand_recon_confs = torch.stack([registered_confs_mean[i]
                                for i in range(milestone+start_ids_offset, milestone+kf_stride)])
        mean_cand_local_confs = torch.stack([local_confs_mean_up2now[i]
                                for i in range(milestone+start_ids_offset, milestone+kf_stride)])
        # normalize the confidence to [0,1], to avoid overconfidence
        mean_cand_recon_confs = (mean_cand_recon_confs - 1)/mean_cand_recon_confs # transform to sigmoid
        mean_cand_local_confs = (mean_cand_local_confs - 1)/mean_cand_local_confs
        # the final confidence is the product of the two kinds of confidences
        mean_cand_confs = mean_cand_recon_confs*mean_cand_local_confs
        
        most_conf_id = mean_cand_confs.argmax().item()
        most_conf_id += start_ids_offset
        id_to_buffer = milestone + most_conf_id
        buffering_set_ids.append(id_to_buffer)
        # print(f"add ref view {id_to_buffer}")                
        # since we have inserted a new frame, overflow must happen when full_flag is True
        if full_flag:
            if strategy == 'reservoir':
                buffering_set_ids.pop(np.random.randint(max_buffer_size))
            elif strategy == 'fifo':
                buffering_set_ids.pop(0)
        # print(next_register_id, buffering_set_ids)
        milestone += kf_stride
    return milestone, candi_frame_id, buffering_set_ids, 

def estimate_camera_intrinsics_from_frames(views, principal_point):
    """
    Estimate camera intrinsics (focal length) from depth maps.
    Returns mean intrinsics for robust PnP pose estimation.
    """
    try:
        import torch as torch_module
        # Get depth from first view's local pcd
        if len(views) == 0:
            return None
        
        valid_focals = []
        for view in views:
            if not isinstance(view, dict):
                continue
            if 'pts3d_cam' in view and view['pts3d_cam'] is not None:
                try:
                    pts3d = view['pts3d_cam']
                    if hasattr(pts3d, 'shape') and len(pts3d.shape) == 3:
                        pts3d = pts3d[None]
                    focal = estimate_focal_knowing_depth(
                        pts3d,
                        principal_point,
                        focal_mode='weiszfeld'
                    )
                    if focal is not None:
                        valid_focals.append(focal)
                except Exception:
                    pass
        
        if len(valid_focals) > 0:
            mean_focal = torch.mean(torch.stack(valid_focals))
            intrinsic = np.eye(3)
            intrinsic[0, 0] = float(mean_focal.cpu() if isinstance(mean_focal, torch.Tensor) else mean_focal)
            intrinsic[1, 1] = float(mean_focal.cpu() if isinstance(mean_focal, torch.Tensor) else mean_focal)
            intrinsic[:2, 2] = principal_point.numpy() if isinstance(principal_point, torch.Tensor) else principal_point
            return intrinsic
    except Exception as e:
        print(f"Warning: failed to estimate intrinsics: {e}")
    return None


def estimate_pose_from_pcd(pts3d, intrinsic):
    """
    Estimate camera pose (c2w) from 3D point cloud using PnP RANSAC.
    Returns (c2w_matrix, valid_flag).
    """
    try:
        if intrinsic is None:
            return None, False
        
        c2w, success = estimate_camera_pose(pts3d, intrinsic)
        return c2w, success
    except Exception as e:
        return None, False


def estimate_camera_pose_from_correspondences(pts3d_cam, pts3d_world):
    """
    Estimate camera pose (c2w) from corresponding camera-frame and world-frame point clouds.
    This anchors pose estimation to the existing registered map.
    """
    try:
        cam_pts = to_numpy(pts3d_cam)
        world_pts = to_numpy(pts3d_world)

        if cam_pts.ndim == 4 and cam_pts.shape[0] == 1:
            cam_pts = cam_pts[0]
        if world_pts.ndim == 4 and world_pts.shape[0] == 1:
            world_pts = world_pts[0]

        if cam_pts.ndim != 3 or world_pts.ndim != 3 or cam_pts.shape[2] != 3 or world_pts.shape[2] != 3:
            raise ValueError('estimate_camera_pose_from_correspondences expects (H, W, 3) point clouds')

        cam_flat = cam_pts.reshape(-1, 3)
        world_flat = world_pts.reshape(-1, 3)
        valid_cam = np.isfinite(cam_flat).all(axis=1) & (np.linalg.norm(cam_flat, axis=1) > 1e-6)
        valid_world = np.isfinite(world_flat).all(axis=1) & (np.linalg.norm(world_flat, axis=1) > 1e-6)
        valid = valid_cam & valid_world

        if np.sum(valid) < 20:
            return np.eye(4), False

        c2w = estimate_rigid_transform(cam_flat[valid], world_flat[valid])
        return c2w, True
    except Exception:
        return np.eye(4), False


def is_valid_position_jump(pos_current, pos_prev, max_jump=1.0):
    """Check if position jump is within reasonable bounds (max_jump in meters)."""
    if pos_prev is None:
        return True
    try:
        distance = np.linalg.norm(np.array(pos_current) - np.array(pos_prev))
        return distance <= max_jump
    except:
        return False


def scene_recon_pipeline_online(i2p_model:Image2PointsModel,
                                l2w_model:Local2WorldModel,
                                frame_reader:FrameReader,
                                args:argparse.Namespace,
                                save_dir = "results"):
    win_r = args.win_r
    num_scene_frame = args.num_scene_frame
    initial_winsize = args.initial_winsize
    conf_thres_l2w = args.conf_thres_l2w
    conf_thres_i2p = args.conf_thres_i2p
    num_points_save = args.num_points_save
    kf_stride = args.keyframe_stride
    retrieve_freq = args.retrieve_freq
    update_buffer_intv = kf_stride*args.update_buffer_intv   # update the buffering set every update_buffer_intv frames
    max_buffer_size = args.buffer_size
    strategy = args.buffer_strategy    
    
    scene_id = args.test_name + "_online"
    # Prepare trajectory exporter
    cam_id = getattr(args, 'camera_name', None) or (frame_reader.dataset if isinstance(frame_reader.dataset, str) else 'cam0')
    trajectories = {'metadata': {'cameras': [cam_id], 'num_frames': None}, cam_id: {'frames': []}}
    data_views = [] # store the processed views for reconstruction
    rgb_imgs = []
    input_views = [] # store the views with img tokens and predicted point clouds, which serve as input to i2p and l2w models

    local_confs_mean_up2now = []
    adj_distance = kf_stride
    fail_view = {}
    last_ref_ids_buffer = []
    
    assert initial_winsize >= 2, "not enough views for initializing the scene reconstruction"
    per_frame_res = dict(i2p_pcds=[], i2p_confs=[], l2w_pcds=[], l2w_confs=[])
    registered_confs_mean = []
    num_frame_read = 0
    num_frame_pass = 0
    
    # Track intrinsics and previous position for pose estimation
    principal_point = torch.tensor((224//2, 224//2))  # Assuming 224x224 input
    mean_intrinsics = None
    prev_valid_position = None
    
    while True:
        success, frame = frame_reader.read()
        if not success:
            break
        num_frame_pass += 1

        if (num_frame_pass - 1) % args.perframe == 0:
            num_frame_read += 1
            current_frame_id = num_frame_read - 1
            frame, data_views, rgb_imgs = get_raw_input_frame(
                                                frame_reader.type, data_views, 
                                                rgb_imgs, current_frame_id, 
                                                frame, args.device)
            input_view, per_frame_res, registered_confs_mean =\
                                process_input_frame(per_frame_res, registered_confs_mean, 
                                                    data_views, current_frame_id, i2p_model)
            input_views.append(input_view)
            
            # accumulate enough frames for scene initialization
            if current_frame_id < (initial_winsize - 1) * kf_stride:
                continue
            # it's time to initialize the scene
            elif current_frame_id == (initial_winsize - 1) * kf_stride:
                initialSceneOutput = initial_scene_for_accumulated_frames(
                                        input_views, initial_winsize, kf_stride,
                                        i2p_model, per_frame_res, registered_confs_mean,
                                        args.buffer_size, conf_thres_i2p)
                buffering_set_ids = initialSceneOutput[0]
                init_ref_id = initialSceneOutput[1]
                init_num = initialSceneOutput[2]
                input_views = initialSceneOutput[3]
                per_frame_res = initialSceneOutput[4]
                registered_confs_mean = initialSceneOutput[5]
                
                local_confs_mean_up2now, per_frame_res, input_views =\
                                recover_points_in_initial_window(
                                                 current_frame_id, buffering_set_ids, kf_stride,
                                                 init_ref_id, per_frame_res,
                                                 input_views, i2p_model, conf_thres_i2p)
                
                # Special treatment: register the frames within the range of initial window with L2W model
                if kf_stride > 1:
                    max_conf_mean, input_views, per_frame_res = \
                        register_initial_window_frames(init_num, kf_stride, buffering_set_ids,
                                                       input_views, l2w_model, per_frame_res, 
                                                       registered_confs_mean, args.device, args.norm_input)
                    # A problem is that the registered_confs_mean of the initial window is generated by I2P model,
                    # while the registered_confs_mean of the frames within the initial window is generated by L2W model,
                    # so there exists a gap. Here we try to align it.
                    max_initial_conf_mean = -1
                    for ii in range(init_num):
                        if registered_confs_mean[ii*kf_stride] > max_initial_conf_mean:
                            max_initial_conf_mean = registered_confs_mean[ii*kf_stride]
                    factor = max_conf_mean / max_initial_conf_mean
                    # print(f'align register confidence with a factor {factor}')
                    for ii in range(init_num):
                        per_frame_res['l2w_confs'][ii*kf_stride] *= factor
                        registered_confs_mean[ii*kf_stride] = per_frame_res['l2w_confs'][ii*kf_stride].mean().cpu()
                
                # prepare for the next online reconstruction
                milestone = init_num * kf_stride + 1
                candi_frame_id = len(buffering_set_ids) # used for the reservoir sampling strategy
                continue
            
            ref_ids, ref_ids_buffer = select_ids_as_reference(buffering_set_ids, current_frame_id, 
                                                input_views, i2p_model, num_scene_frame, win_r,
                                                adj_distance, retrieve_freq, last_ref_ids_buffer)
            last_ref_ids_buffer = ref_ids_buffer

            local_views = [input_views[current_frame_id]] + [input_views[id] for id in ref_ids]
            local_confs_mean_up2now, per_frame_res, input_views = \
                pointmap_local_recon(local_views, i2p_model, current_frame_id, 0, per_frame_res,
                                           input_views, conf_thres_i2p, local_confs_mean_up2now)

            ref_views = [input_views[id] for id in ref_ids]
            input_views, per_frame_res, registered_confs_mean = pointmap_global_register(
                                            ref_views, input_views, l2w_model, 
                                            per_frame_res, registered_confs_mean, current_frame_id,
                                            device=args.device, norm_input=args.norm_input)
            
            next_frame_id = current_frame_id + 1
            if next_frame_id - milestone >= update_buffer_intv:
                milestone, candi_frame_id, buffering_set_ids = update_buffer_set(
                                                next_frame_id, max_buffer_size, 
                                                kf_stride, buffering_set_ids, strategy, 
                                                registered_confs_mean, local_confs_mean_up2now, 
                                                candi_frame_id, milestone)
            
            conf = registered_confs_mean[current_frame_id]
            if conf < 10:
                fail_view[current_frame_id] = conf.item()
            print(f"finish recover pcd of frame {current_frame_id}, with a mean confidence of {conf:.2f}.")
            # --- per-frame save and trajectory export ---
            if getattr(args, 'save_each_frame', False):
                os.makedirs(save_dir, exist_ok=True)
                try:
                    save_recon(input_views, num_frame_read, save_dir, scene_id, 
                              args.save_all_views, rgb_imgs, registered_confs=per_frame_res['l2w_confs'], 
                              num_points_save=num_points_save, conf_thres_res=conf_thres_l2w)
                except Exception as e:
                    print(f"Warning: failed to save per-frame recon: {e}")

                # export camera pose (c2w) using PnP RANSAC on registered point cloud (robust approach from visualize.py)
                pos = [0.0, 0.0, 0.0]
                forward = [0.0, 0.0, 0.0]
                valid_pose = False
                
                # Try correspondences-based pose estimation first, then fallback to PnP, then fallback to model c2w
                pose_estimated = False
                try:
                    if ('pts3d_world' in input_views[current_frame_id] and input_views[current_frame_id]['pts3d_world'] is not None):
                        registered_pcd = input_views[current_frame_id]['pts3d_world']
                        registered_cam = input_views[current_frame_id].get('pts3d_cam')

                        if registered_cam is not None:
                            c2w, success = estimate_camera_pose_from_correspondences(registered_cam, registered_pcd)
                            if success and c2w is not None:
                                pose_mat = c2w
                                pos = pose_mat[:3, 3].tolist()
                                forward = pose_mat[:3, 2].tolist()
                                valid_pose = True
                                pose_estimated = True

                        if not pose_estimated:
                            # Extract and transform registered point cloud to a 3xHxW shape if needed
                            if isinstance(registered_pcd, torch.Tensor):
                                registered_pcd = to_numpy(registered_pcd)
                            if registered_pcd.ndim == 4 and registered_pcd.shape[0] == 1:
                                registered_pcd = registered_pcd[0]
                            if registered_pcd.shape[0] == 3:
                                registered_pcd = registered_pcd.transpose(1,2,0)
                            registered_pcd = registered_pcd.copy()

                            # Estimate intrinsics if not already done
                            if mean_intrinsics is None:
                                valid_pts = [input_views[id]['pts3d_cam'] for id in range(min(len(input_views), 10)) 
                                             if 'pts3d_cam' in input_views[id] and input_views[id]['pts3d_cam'] is not None]
                                if valid_pts:
                                    mean_intrinsics = estimate_camera_intrinsics_from_frames(valid_pts, principal_point)

                            # Use PnP RANSAC to estimate pose from the registered point cloud in the model's native coordinate frame
                            if mean_intrinsics is not None:
                                c2w, success = estimate_pose_from_pcd(registered_pcd, mean_intrinsics)
                                if success and c2w is not None:
                                    pose_mat = c2w
                                    pos = pose_mat[:3, 3].tolist()
                                    forward = pose_mat[:3, 2].tolist()
                                    valid_pose = True
                                    pose_estimated = True
                except Exception as e:
                    if current_frame_id < 20:  # Only print first few errors to avoid spam
                        print(f"  Pose estimation failed for frame {current_frame_id}: {str(e)[:100]}")
                
                # Fallback to model's c2w if correspondence / PnP pose estimation failed
                if not pose_estimated and 'c2w' in input_views[current_frame_id] and input_views[current_frame_id]['c2w'] is not None:
                    try:
                        pose_mat = to_numpy(input_views[current_frame_id]['c2w']).reshape(4,4)
                        pos = pose_mat[:3, 3].tolist()
                        forward = pose_mat[:3, 2].tolist()
                        valid_pose = True
                    except Exception as e:
                        if current_frame_id < 20:
                            print(f"  Model c2w fallback failed for frame {current_frame_id}: {str(e)[:100]}")

                frame_entry = {
                    'frame': current_frame_id,
                    'valid': bool(valid_pose),
                    'position': pos,
                    'forward': forward,
                    'conf': float(conf) if not isinstance(conf, torch.Tensor) else float(conf.cpu()),
                    'timestamp': time.time()
                }
                trajectories[cam_id]['frames'].append(frame_entry)
                trajectories['metadata']['num_frames'] = len(trajectories[cam_id]['frames'])
                try:
                    with open(join(save_dir, 'trajectories.json'), 'w') as f:
                        json.dump(trajectories, f, indent=2)
                except Exception as e:
                    print(f"Warning: failed to write trajectories.json: {e}")
            
    print(f"finish reconstructing {num_frame_read} frames")
    print(f'mean confidence for whole scene reconstruction: {torch.tensor(registered_confs_mean).mean().item():.2f}')
    print(f"{len(fail_view)} views with low confidence: ", {key:round(fail_view[key],2) for key in fail_view.keys()})
    
    save_recon(input_views, num_frame_read, save_dir, scene_id, 
                      args.save_all_views, rgb_imgs, registered_confs=per_frame_res['l2w_confs'], 
                      num_points_save=num_points_save, 
                      conf_thres_res=conf_thres_l2w)
    
    if args.save_preds:
        preds_dir = join(save_dir, 'preds')
        os.makedirs(preds_dir, exist_ok=True)
        print(f">> saving per-frame predictions to {preds_dir}") 
        np.save(join(preds_dir, 'local_pcds.npy'), torch.cat(per_frame_res['i2p_pcds']).cpu().numpy())
        np.save(join(preds_dir, 'registered_pcds.npy'), torch.cat(per_frame_res['l2w_pcds']).cpu().numpy())
        np.save(join(preds_dir, 'local_confs.npy'), torch.stack([conf.cpu() for conf in per_frame_res['i2p_confs']]).numpy())
        np.save(join(preds_dir, 'registered_confs.npy'), torch.stack([conf.cpu() for conf in per_frame_res['l2w_confs']]).numpy())
        np.save(join(preds_dir, 'input_imgs.npy'), np.stack(rgb_imgs))
        
        metadata = dict(scene_id=scene_id,
                        init_winsize=init_num,
                        kf_stride=kf_stride,
                        init_ref_id=init_ref_id)
        with open(join(preds_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    elif args.save_for_eval:
        preds_dir = join(save_dir, 'preds')
        os.makedirs(preds_dir, exist_ok=True)
        print(f">> saving per-frame predictions to {preds_dir}")
        np.save(join(preds_dir, 'registered_pcds.npy'), torch.cat(per_frame_res['l2w_pcds']).cpu().numpy())
        np.save(join(preds_dir, 'registered_confs.npy'), torch.stack([conf.cpu() for conf in per_frame_res['l2w_confs']]).numpy())