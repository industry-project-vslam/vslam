import torch
import cv2
import numpy as np
from os.path import join
from tqdm import tqdm
import matplotlib.pyplot as plt
import trimesh

from slam3r.utils.device import to_numpy, collate_with_cat, to_cpu
from slam3r.inference import loss_of_one_batch_multiview, \
                                inv, get_multiview_scale
from slam3r.utils.geometry import xy_grid
from slam3r.pipeline.recon_offline_pipeline import scene_frame_retrieve # Added import for scene_frame_retrieve

try:
    import poselib  # noqa
    HAS_POSELIB = True
except Exception as e:
    HAS_POSELIB = False


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

def save_traj(views, pred_frame_num, save_dir, scene_id, args, 
              intrinsics = None, traj_name = 'traj'): 
    save_name = f"{scene_id}_{traj_name}.txt"

    c2ws = []
    H, W, _ = views[0]['pts3d_world'][0].shape
    for i in tqdm(range(pred_frame_num)):
        pts = to_numpy(views[i]['pts3d_world'][0])
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        points_2d = np.stack((u, v), axis=-1)
        dist_coeffs = np.zeros(4).astype(np.float32)
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            pts.reshape(-1, 3).astype(np.float32), 
            points_2d.reshape(-1, 2).astype(np.float32), 
            intrinsics[i].astype(np.float32), 
            dist_coeffs)
    
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        # Extrinsic parameters (4x4 matrix)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
        c2w = inv(extrinsic_matrix)
        c2ws.append(c2w)
    c2ws = np.stack(c2ws, axis=0)
    translations = c2ws[:,:3,3]
    # draw the trajectory in horizontal plane
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_traj(ax, [i for i in range(len(translations))], translations,
                '-', "black", "estimate trajectory")
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.savefig(join(save_dir, save_name.replace('.txt', '.png')), dpi=90)
    np.savetxt(join(save_dir, save_name), c2ws.reshape(-1,16))


def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib. 
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    """
    stamps.sort()
    interval = np.median([s-t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)


def estimate_camera_pose(pts3d, intrinsic):
    pts = to_numpy(pts3d)
    if pts.ndim != 3 or pts.shape[2] != 3:
        raise ValueError('estimate_camera_pose expects a (H, W, 3) point cloud')

    # Filter out invalid points so solvePnPRansac gets only meaningful correspondences.
    valid = np.linalg.norm(pts.reshape(-1, 3), axis=-1) > 1e-6
    if valid.sum() < 20:
        return np.eye(4), False

    pts_valid = pts.reshape(-1, 3)[valid]
    H, W, _ = pts.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    points_2d = np.stack((u, v), axis=-1).reshape(-1, 2)[valid]

    dist_coeffs = np.zeros(4).astype(np.float32)
    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        pts_valid.astype(np.float32),
        points_2d.astype(np.float32),
        intrinsic.astype(np.float32),
        dist_coeffs)
    if not success:
        return np.eye(4), False
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    # Extrinsic parameters (4x4 matrix)
    extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
    extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
    c2w = inv(extrinsic_matrix)

    return c2w, True


def estimate_rigid_transform(src_pts, dst_pts):
    """
    Estimates the rigid body transformation (rotation and translation)
    from source points to destination points using SVD.

    Args:
        src_pts (np.ndarray): (N, 3) array of source points.
        dst_pts (np.ndarray): (N, 3) array of destination points.

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    assert src_pts.shape == dst_pts.shape
    assert src_pts.shape[1] == 3

    num_points = src_pts.shape[0]

    # Compute centroids
    centroid_src = np.mean(src_pts, axis=0)
    centroid_dst = np.mean(dst_pts, axis=0)

    # Center the points
    centered_src = src_pts - centroid_src
    centered_dst = dst_pts - centroid_dst

    # Compute covariance matrix H
    H = centered_src.T @ centered_dst

    # Perform SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix R
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Compute translation vector t
    t = centroid_dst - R @ centroid_src

    # Form homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def estimate_intrinsics(pts3d_local):
    ##### estimate focal length
    B, H, W, _ = pts3d_local.shape
    pp = torch.tensor((W/2, H/2))
    focal = estimate_focal_knowing_depth(pts3d_local.cpu(), pp, focal_mode='weiszfeld')
    # print(f'Estimated focal of first camera: {focal.item()} (224x224)')
    intrinsic = np.eye(3)
    intrinsic[0, 0] = focal
    intrinsic[1, 1] = focal
    intrinsic[:2, 2] = pp
    return intrinsic


def estimate_focal_knowing_depth(pts3d, pp, focal_mode='median', min_focal=0., max_focal=np.inf):
    """ Reprojection method, for when the absolute depth is known:
        1) estimate the camera focal using a robust estimator
        2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    pp = pp.to(pts3d.device).float()

    if H * W > 1024 * 1024:
        raise ValueError(f"Input resolution too large for focal estimation: {H}x{W}")

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

    if focal_mode == 'median':
        with torch.no_grad():
            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == 'weiszfeld':
        # init focal with l2 closed form
        # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # homogeneous (x,y,1)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip(min=1e-8).reciprocal()
            # update the scaling with the new weights
            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy.mean(dim=1))
    else:
        raise ValueError(f'bad {focal_mode=}')

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal*focal_base, max=max_focal*focal_base)
    # print(focal)
    return focal


def unsqueeze_view(view):
    """Uunsqueeze view to batch size 1, 
    similar to collate_fn
    """
    if len(view['img'].shape) > 3:
        return view
    res = dict(img=view['img'][None], 
                 true_shape=view['true_shape'][None], 
                 idx=view['idx'], 
                 instance=view['instance'], 
                 pts3d_cam=torch.tensor(view['pts3d_cam'][None]),
                 valid_mask=torch.tensor(view['valid_mask'][None]),
                 camera_pose=torch.tensor(view['camera_pose']),
                 pts3d=torch.tensor(view['pts3d'][None])
                )
    if 'pointmap_img' in view:
        res['pointmap_img'] = view['pointmap_img'][None]
    
    return res

def transform_img(view):
    #transform to numpy, BGR, 0-255, HWC
    img = view['img'][0]
    # handle torch tensors
    if torch.is_tensor(img):
        arr = img.permute(1, 2, 0).cpu().numpy()
    else:
        # numpy array: could be HWC or CHW
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            # H, W, C (RGB)
            pass
        elif arr.ndim == 3 and arr.shape[0] == 3:
            # C, H, W -> H, W, C
            arr = arr.transpose(1, 2, 0)
        else:
            # fallback: try to squeeze or convert to HWC
            arr = np.squeeze(arr)
    # Ensure data is in RGB order and scale to 0-255
    try:
        # If already uint8 and in 0-255 range, keep as is
        if arr.dtype == np.uint8 or arr.max() > 2:
            img_bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor((arr / 2.0 + 0.5).astype(np.float32), cv2.COLOR_RGB2BGR)
            img_bgr = (img_bgr * 255.0).astype(np.uint8)
    except Exception:
        # last-resort conversion
        arr = np.clip(arr, 0, 255)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return img_bgr


def save_ply(points:np.array, save_path, colors:np.array=None, metadata:dict=None):
    #color:0-1
    if np.max(colors) > 1:
        colors = colors/255.
    pcd = trimesh.points.PointCloud(points, colors=colors)
    if metadata is not None:
        for key in metadata:
            pcd.metadata[key] = metadata[key]
    pcd.export(save_path)
    print(">> save_to", save_path)


def save_vis(points, dis, vis_path):
    cmap = plt.get_cmap('Reds')
    color = cmap(dis/0.05)
    save_ply(points=points, save_path=vis_path, colors=color)


def uni_upsample(img,scale):
    img = np.array(img)
    upsampled_img = img[:,None,:,None].repeat(scale,1).repeat(scale,3).reshape(img.shape[0]*scale,-1)
    return upsampled_img


def normalize_views(pts3d:list, valid_masks=None, return_factor=False):
    """normalize the input point clouds
    by the average distance of the valid points to the origin
    
    Args:
        pts3d: list of tensors, each tensor has shape (1,224,224,3)
        valid_masks: list of tensors, each tensor has shape (1,224,224)
        return_factor: whether to return the normalization factor
    """
    num_views = len(pts3d)  # num_views*(1,224,224,3)
    if valid_masks is None:
        valid_masks = [torch.ones(p.shape[:-1], dtype=bool, device=pts3d[0].device) for p in pts3d]
    assert num_views == len(valid_masks)
    norm_factor = get_multiview_scale([pts3d[id] for id in range(num_views)],
                                                [valid_masks[id] for id in range(num_views)], 
                                                norm_mode='avg_dis')
    normed_pts3d = [pts3d[id] / norm_factor for id in range(num_views)]
    if return_factor:
        return normed_pts3d, norm_factor
    return normed_pts3d


def to_device(view, device='cuda'):
    """ transfer the input view to the target device
    """
    for name in 'img pts3d_cam pts3d_world true_shape img_tokens'.split():
        if name in view:
            view[name] = view[name].to(device)


def _prepare_l2w_mask(mask, target_tensor):
    """Prepare mask for broadcasting multiplication with target tensor.
    
    Args:
        mask: Mask tensor with shape (..., H, W) or (..., H, W, 1)
        target_tensor: Target tensor with shape (..., H, W, 3) or similar
        
    Returns:
        mask.float() with compatible shape for broadcasting
    """
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask, device=target_tensor.device)
    if mask.dtype != torch.bool:
        mask = mask != 0
    # Ensure mask has same number of dimensions as target
    while mask.dim() < target_tensor.dim():
        mask = mask.unsqueeze(-1)
    # Broadcasting will handle the rest; no need for exact shape match
    return mask.float()


@torch.no_grad()
def i2p_inference_batch(batch_views:list, model, device='cuda', 
                       ref_id=0, 
                       tocpu=True, 
                       unsqueeze=True):
    """inference on a batch of views with the Image2Points model
    batch_views: list of list, [[view1, view2, ...], [view1, view2, ...], ...]
                                     batch1                 batch2       ...
    """
    pairs = []
    for views in batch_views:
        if unsqueeze:
            pairs.append(tuple(unsqueeze_view(view) for view in views))
        else:
            pairs.append(tuple(views))

    input = collate_with_cat(pairs)
    res = loss_of_one_batch_multiview(input, model, None, device, ref_id=ref_id)
    result = [to_cpu(res)] if tocpu else [res]
    output = collate_with_cat(result)   #views,preds,loss,view1,..pred1...
    return output


@torch.no_grad()
def l2w_inference(raw_views, l2w_model, ref_ids, 
                  masks=None,
                  normalize=False, 
                  device='cuda'):
    """Multi-keyframe co-registration with the Local2World model
    Input:
        raw_views(should be collated): list of views, each view is a dict containing:
            img_tokens: the img tokens output from encoder: (B, Patch_H, Patch_W, C)
            pts3d_cam: the point clouds in the camera coordinate: (B, H, W, 3)
            ...
        model: the Local2World model
        ref_ids: the ids of scene frames
        masks: the masks of the input pointmap
        normalize: whether to normalize the input point clouds
    """
    # construct new input to avoid modifying the raw views
    input_views = [dict(img_tokens=view['img_tokens'], 
                        true_shape=view['true_shape'],
                        img_pos=view['img_pos']) 
                   for view in raw_views]
    
    for view in input_views:
        to_device(view, device=device)    
    
    # pts3d_world in input scene frames are normalized together, 
    # while pts3d_cam in input keyframes are normalized separately
    # Here we calculate the normalized pts3d_world ahead of time
    if normalize:
        normed_pts_world, norm_factor_world = \
            normalize_views([raw_views[i]['pts3d_world'] for i in ref_ids], 
                            None if masks is None else [masks[i] for i in ref_ids],  
                            return_factor=True)

    for id,view in enumerate(raw_views):            
        if id in ref_ids:
            if normalize:
                pts_world = normed_pts_world[ref_ids.index(id)]
            else:
                pts_world = view['pts3d_world']
            if masks is not None:
                mask = _prepare_l2w_mask(masks[id], pts_world)
                pts_world = pts_world * mask
            input_views[id]['pts3d_world'] = pts_world
        else:
            if normalize:
                input_views[id]['pts3d_cam'] = normalize_views([raw_views[id]['pts3d_cam']],
                                                None if masks is None else [masks[id]])[0]
            else:
                input_views[id]['pts3d_cam'] = raw_views[id]['pts3d_cam']
            if masks is not None:
                mask = _prepare_l2w_mask(masks[id], input_views[id]['pts3d_cam'])
                input_views[id]['pts3d_cam'] = input_views[id]['pts3d_cam'] * mask
        
    with torch.no_grad():
        output = l2w_model(input_views, ref_ids=ref_ids)

    # restore the predicted points to the original scale in raw_views
    if normalize:
        for i in range(len(raw_views)):
            if i in ref_ids:
                output[i]['pts3d'] = output[i]['pts3d'] * norm_factor_world
            else:
                output[i]['pts3d_in_other_view'] = output[i]['pts3d_in_other_view'] * norm_factor_world

    # Estimate camera pose for source views and add to output
    for i in range(len(raw_views)):
        if i not in ref_ids:  # This is a source view
            # Get pts3d_cam (original points in camera frame) from raw_views
            # and pts3d_in_other_view (transformed points in world frame) from output
            src_pts_cam = raw_views[i]['pts3d_cam']
            dst_pts_world = output[i]['pts3d_in_other_view']

            # Reshape from (B, H, W, 3) to (N, 3) and convert to numpy
            src_pts_cam_np = to_numpy(src_pts_cam.flatten(start_dim=0, end_dim=-2))
            dst_pts_world_np = to_numpy(dst_pts_world.flatten(start_dim=0, end_dim=-2))

            # Filter out points where all coordinates are zero (invalid points)
            valid_src_indices = ~np.all(src_pts_cam_np == 0, axis=1)
            valid_dst_indices = ~np.all(dst_pts_world_np == 0, axis=1)
            
            # Use intersection of valid indices
            valid_indices = valid_src_indices & valid_dst_indices
            
            if np.sum(valid_indices) >= 3: # Need at least 3 points for SVD
                estimated_transform = estimate_rigid_transform(
                    src_pts_cam_np[valid_indices],
                    dst_pts_world_np[valid_indices]
                )
                output[i]['c2w'] = torch.from_numpy(estimated_transform).to(device)
            else:
                # If not enough valid points, set 'c2w' to None,
                # which _extract_pose_from_l2w can handle.
                output[i]['c2w'] = None
            
    return output


def get_free_gpu():
    # initialize PyCUDA
    try:
        import pycuda.driver as cuda
    except ImportError as e:
        print(f"{e} -- fail to import pycuda, choose GPU 0.")
        return 0
    
    cuda.init()
    device_count = cuda.Device.count()
    most_free_mem = 0
    most_free_id = 0
    for i in range(device_count):
        try:
            device = cuda.Device(i)
            context = device.make_context()
            # query the free memory on the device
            free_memory = cuda.mem_get_info()[0]
            
            # if the gpu is totally free, return it
            total_memory = device.total_memory()
            if free_memory == total_memory:
                context.pop()
                return i
            
            if(free_memory > most_free_mem):
                most_free_mem = free_memory
                most_free_id = i
            
            context.pop()
        except:
            pass
    print("No totally free GPU found! Choose the most free one.")

    return most_free_id


def estimate_intrinsics_from_local_pcds(local_pcds, init_ids, init_ref_id, principal_point):
    """
    Estimate camera intrinsics from local point clouds.
    
    Args:
        local_pcds: (V, 224, 224, 3) local point clouds
        init_ids: indices of initial reference frames
        init_ref_id: index of initial reference frame
        principal_point: torch tensor of shape (2,) with principal point
    
    Returns:
        focals: list of focal lengths (one per view)
        intrinsics: list of (3,3) intrinsic matrices
        mean_intrinsics: (3,3) mean intrinsic matrix
    """
    num_views = local_pcds.shape[0]
    
    # Estimate focal length for initial reference
    init_window_focal = estimate_focal_knowing_depth(
        torch.tensor(local_pcds[init_ref_id][None]),
        principal_point,
        focal_mode='weiszfeld'
    )
    
    focals = []
    for i in tqdm(range(num_views), desc="estimating intrinsics"):
        if i in init_ids:
            focals.append(init_window_focal)
        else:
            focal = estimate_focal_knowing_depth(
                torch.tensor(local_pcds[i:i+1]),
                principal_point,
                focal_mode='weiszfeld'
            )
            focals.append(focal)
    
    # Build intrinsic matrices
    intrinsics = []
    for i in range(num_views):
        intrinsic = np.eye(3)
        focal_val = focals[i]
        if isinstance(focal_val, torch.Tensor):
            focal_val = focal_val.item()
        intrinsic[0, 0] = focal_val
        intrinsic[1, 1] = focal_val
        if isinstance(principal_point, torch.Tensor):
            intrinsic[:2, 2] = principal_point.numpy()
        else:
            intrinsic[:2, 2] = principal_point
        intrinsics.append(intrinsic)
    
    mean_intrinsics = np.mean(np.stack(intrinsics, axis=0), axis=0)
    return focals, intrinsics, mean_intrinsics


def estimate_poses_batch(registered_pcds, intrinsics, desc="estimating camera poses"):
    """
    Estimate camera poses (c2w) from registered point clouds in batch.
    
    Args:
        registered_pcds: (V, 224, 224, 3) registered point clouds
        intrinsics: list of (3,3) intrinsic matrices or single (3,3) matrix
        desc: description for progress bar
    
    Returns:
        c2ws: list of (4,4) camera-to-world matrices
        successes: list of bool indicating pose estimation success
    """
    num_views = registered_pcds.shape[0]
    
    # Use same intrinsics for all if single matrix provided
    if isinstance(intrinsics, np.ndarray) and intrinsics.ndim == 2:
        intrinsics = [intrinsics] * num_views
    
    c2ws = []
    successes = []
    for i in tqdm(range(num_views), desc=desc):
        registered_pcd = registered_pcds[i]
        c2w, succ = estimate_camera_pose(registered_pcd, intrinsics[i])
        c2ws.append(c2w)
        successes.append(bool(succ))
        if not succ:
            print(f"  failed to estimate camera pose for view {i}")
    
    return c2ws, successes


def poses_to_trajectory_json(c2ws, successes, registered_confs=None, cam_id="cam0"):
    """
    Convert estimated poses to online tracker trajectory JSON format.
    
    Args:
        c2ws: list of (4,4) camera-to-world matrices
        successes: list of bool indicating valid poses
        registered_confs: optional (V, 224, 224) confidence maps
        cam_id: camera id label
    
    Returns:
        trajectories: dict in online tracker format
    """
    import time
    num_views = len(c2ws)
    
    trajectories = {
        "metadata": {"cameras": [cam_id], "num_frames": num_views},
        cam_id: {"frames": []}
    }
    
    for i in range(num_views):
        valid = successes[i]
        position = [0.0, 0.0, 0.0]
        forward = [0.0, 0.0, 0.0]
        
        if valid and c2ws[i] is not None:
            pose = c2ws[i]
            position = pose[:3, 3].tolist()
            forward_vec = pose[:3, 2]
            norm_forward = np.linalg.norm(forward_vec)
            if norm_forward > 1e-6:
                forward = (forward_vec / norm_forward).tolist()
        
        conf = 0.0
        if registered_confs is not None:
            conf = float(np.mean(registered_confs[i]))
        
        frame_entry = {
            "frame": i,
            "valid": valid,
            "position": position,
            "forward": forward,
            "conf": conf,
            "timestamp": time.time()
        }
        trajectories[cam_id]["frames"].append(frame_entry)
    
    return trajectories