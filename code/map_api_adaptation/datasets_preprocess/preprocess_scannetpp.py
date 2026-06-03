#!/usr/bin/env python3
# --------------------------------------------------------
# Script to pre-process the scannet++ dataset, adapted from DUSt3R
# Usage:
# python3 datasets_preprocess/preprocess_scannetpp.py --scannetpp_dir /data0/yuzheng/data/scannetpp 
# --------------------------------------------------------
import os
import argparse
import os.path as osp
import re
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation
import pyrender
import trimesh
import trimesh.exchange.ply
import numpy as np
import cv2
import PIL.Image as Image

SLAM3R_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
import sys # noqa: E402
sys.path.insert(0, SLAM3R_DIR) # noqa: E402
from slam3r.datasets.utils.cropping import rescale_image_depthmap
import slam3r.utils.geometry as geometry

inv = np.linalg.inv
norm = np.linalg.norm
REGEXPR_DSLR = re.compile(r'^DSC(?P<frameid>\d+).JPG$')
REGEXPR_IPHONE = re.compile(r'frame_(?P<frameid>\d+).jpg$')

DEBUG_VIZ = None  # 'iou'
if DEBUG_VIZ is not None:
    import matplotlib.pyplot as plt  # noqa


OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannetpp_dir', default="./data/scannetpp")
    parser.add_argument('--output_dir', default='./data/scannetpp_processed')
    parser.add_argument('--target_resolution', default=920, type=int, help="images resolution")
    parser.add_argument('--pyopengl-platform', type=str, default='egl', help='PyOpenGL env variable')
    return parser


def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)  # returns cam2world


def get_frame_number(name, cam_type='dslr'):
    if cam_type == 'dslr':
        regex_expr = REGEXPR_DSLR
        if '_' in name: # for 02dd3b53_DSC06910.JPG
            name = name.split('_')[-1]
    elif cam_type == 'iphone':
        regex_expr = REGEXPR_IPHONE
    else:
        raise NotImplementedError(f'wrong {cam_type=} for get_frame_number')
    
    matches = re.match(regex_expr, name)
    assert matches is not None, f'wrong {name=} for get_frame_number'
    
    return matches['frameid']


def load_sfm(sfm_dir, cam_type='dslr'):
    # load cameras
    with open(osp.join(sfm_dir, 'cameras.txt'), 'r') as f:
        raw = f.read().splitlines()[3:]  # skip header

    intrinsics = {}
    for camera in tqdm(raw, position=1, leave=False):
        camera = camera.split(' ')
        intrinsics[int(camera[0])] = [camera[1]] + [float(cam) for cam in camera[2:]]

    # load images
    with open(os.path.join(sfm_dir, 'images.txt'), 'r') as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith('#')]  # skip header

    img_idx = {}
    img_infos = {}
    for image, points in tqdm(zip(raw[0::2], raw[1::2]), total=len(raw) // 2, position=1, leave=False):
        image = image.split(' ')
        points = points.split(' ')

        idx = image[0]
        img_name = image[-1]
        if cam_type == 'iphone': # for video/frame_011690.jpg and iphone/frame_011690.jpg
            img_name = os.path.basename(img_name)
        assert img_name not in img_idx, 'duplicate db image: ' + img_name
        img_idx[img_name] = idx  # register image name

        current_points2D = {int(i): (float(x), float(y))
                            for i, x, y in zip(points[2::3], points[0::3], points[1::3]) if i != '-1'}
        img_infos[idx] = dict(intrinsics=intrinsics[int(image[-2])],
                              path=img_name,
                              frame_id=get_frame_number(img_name, cam_type),
                              cam_to_world=pose_from_qwxyz_txyz(image[1: -2]),
                              sparse_pts2d=current_points2D)

    # load 3D points
    with open(os.path.join(sfm_dir, 'points3D.txt'), 'r') as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith('#')]  # skip header

    points3D = {}
    observations = {idx: [] for idx in img_infos.keys()}
    for point in tqdm(raw, position=1, leave=False):
        point = point.split()
        point_3d_idx = int(point[0])
        points3D[point_3d_idx] = tuple(map(float, point[1:4]))
        if len(point) > 8:
            for idx, point_2d_idx in zip(point[8::2], point[9::2]):
                if idx in observations.keys(): # some image idx in points3D.txt are not in images.txt?
                    observations[idx].append((point_3d_idx, int(point_2d_idx)))

    return img_idx, img_infos, points3D, observations


def subsample_img_infos(img_infos, num_images, allowed_name_subset=None):
    img_infos_val = [(idx, val) for idx, val in img_infos.items()]
    if allowed_name_subset is not None:
        img_infos_val = [(idx, val) for idx, val in img_infos_val if val['path'] in allowed_name_subset]

    if len(img_infos_val) > num_images:
        img_infos_val = sorted(img_infos_val, key=lambda x: x[1]['frame_id'])
        kept_idx = np.round(np.linspace(0, len(img_infos_val) - 1, num_images)).astype(int).tolist()
        img_infos_val = [img_infos_val[idx] for idx in kept_idx]
    return {idx: val for idx, val in img_infos_val}


def undistort_images(intrinsics, rgb, mask):
    camera_type = intrinsics[0]

    width = int(intrinsics[1])
    height = int(intrinsics[2])
    fx = intrinsics[3]
    fy = intrinsics[4]
    cx = intrinsics[5]
    cy = intrinsics[6]
    distortion = np.array(intrinsics[7:])

    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1

    K = geometry.colmap_to_opencv_intrinsics(K)
    if camera_type == "OPENCV_FISHEYE":
        assert len(distortion) == 4

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            distortion,
            (width, height),
            np.eye(3),
            balance=0.0,
        )
        # Make the cx and cy to be the center of the image
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (width, height), 1, (width, height), True)
        map1, map2 = cv2.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)

    undistorted_image = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    undistorted_mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    K = geometry.opencv_to_colmap_intrinsics(K)
    return width, height, new_K, undistorted_image, undistorted_mask


def process_scenes(root, output_dir, target_resolution):
    os.makedirs(output_dir, exist_ok=True)

    # default values from
    # https://github.com/scannetpp/scannetpp/blob/main/common/configs/render.yml
    znear = 0.05
    zfar = 20.0

    with open(osp.join(root, 'splits', 'nvs_sem_train.txt'), 'r') as f:
        train_scenes = f.read().splitlines()
    with open(osp.join(root, 'splits', 'nvs_sem_val.txt'), 'r') as f:
        val_scenes = f.read().splitlines()
    scenes = train_scenes + val_scenes
    
    if not osp.isdir(osp.join(output_dir, 'splits')):
        os.system(f"cp -r {osp.join(root, 'splits')} {output_dir}")

    # for each of these, we will select some dslr images and some iphone images
    # we will undistort them and render their depth
    renderer = pyrender.OffscreenRenderer(0, 0)
    for scene in tqdm(scenes, position=0, leave=True):
        try:
            print(f"Processing scene {scene}")
            data_dir = os.path.join(root, 'data', scene)
            dir_dslr = os.path.join(data_dir, 'dslr')
            dir_iphone = os.path.join(data_dir, 'iphone')
            dir_scans = os.path.join(data_dir, 'scans')

            assert os.path.isdir(data_dir) and os.path.isdir(dir_dslr) \
                and os.path.isdir(dir_iphone) and os.path.isdir(dir_scans)

            output_dir_scene = os.path.join(output_dir, scene)
            scene_dslr_metadata_path = osp.join(output_dir_scene, 'scene_dslr_metadata.npz')
            scene_iphone_metadata_path = osp.join(output_dir_scene, 'scene_iphone_metadata.npz')
            if osp.isfile(scene_dslr_metadata_path) and osp.isfile(scene_iphone_metadata_path):
                continue

            # set up the output paths
            output_dir_scene_rgb = os.path.join(output_dir_scene, 'images')
            output_dir_scene_depth = os.path.join(output_dir_scene, 'depth')
            os.makedirs(output_dir_scene_rgb, exist_ok=True)
            os.makedirs(output_dir_scene_depth, exist_ok=True)

            ply_path = os.path.join(dir_scans, 'mesh_aligned_0.05.ply')

            sfm_dir_dslr = os.path.join(dir_dslr, 'colmap')
            rgb_dir_dslr = os.path.join(dir_dslr, 'resized_images')
            mask_dir_dslr = os.path.join(dir_dslr, 'resized_anon_masks')

            sfm_dir_iphone = os.path.join(dir_iphone, 'colmap')
            rgb_dir_iphone = os.path.join(dir_iphone, 'rgb')
            mask_dir_iphone = os.path.join(dir_iphone, 'rgb_masks')

            # load the mesh
            with open(ply_path, 'rb') as f:
                mesh_kwargs = trimesh.exchange.ply.load_ply(f)
            mesh_scene = trimesh.Trimesh(**mesh_kwargs)

            # read colmap reconstruction, we will only use the intrinsics and pose here
            img_idx_dslr, img_infos_dslr, points3D_dslr, observations_dslr = load_sfm(sfm_dir_dslr, cam_type='dslr')
            dslr_paths = {
                "in_colmap": sfm_dir_dslr,
                "in_rgb": rgb_dir_dslr,
                "in_mask": mask_dir_dslr,
            }
            # filter out the test images in dslr because they are disordered
            with open(os.path.join(dir_dslr, 'train_test_lists.json'), 'r') as f:
                test_list = json.load(f)['test']
            for img_name in test_list:    
                idx = img_idx_dslr[img_name]
                del img_infos_dslr[idx]
                del observations_dslr[idx]

            img_idx_iphone, img_infos_iphone, points3D_iphone, observations_iphone = load_sfm(
                sfm_dir_iphone, cam_type='iphone')
            iphone_paths = {
                "in_colmap": sfm_dir_iphone,
                "in_rgb": rgb_dir_iphone,
                "in_mask": mask_dir_iphone,
            }

            mesh = pyrender.Mesh.from_trimesh(mesh_scene, smooth=False)
            pyrender_scene = pyrender.Scene()
            pyrender_scene.add(mesh)

            # resize the image to a more manageable size and render depth
            for img_idx, img_infos, paths_data, out_metadata_path in [(img_idx_dslr, img_infos_dslr, dslr_paths, scene_dslr_metadata_path),
                                                                (img_idx_iphone, img_infos_iphone, iphone_paths, scene_iphone_metadata_path)]:
                rgb_dir = paths_data['in_rgb']
                mask_dir = paths_data['in_mask']
                for imgidx in tqdm(img_infos.keys()):
                    img_infos_idx = img_infos[imgidx]
                    rgb = np.array(Image.open(os.path.join(rgb_dir, img_infos_idx['path'])))
                    mask = np.array(Image.open(os.path.join(mask_dir, img_infos_idx['path'][:-3] + 'png')))

                    _, _, K, rgb, mask = undistort_images(img_infos_idx['intrinsics'], rgb, mask)

                    # rescale_image_depthmap assumes opencv intrinsics
                    intrinsics = geometry.colmap_to_opencv_intrinsics(K)
                    image, mask, intrinsics = rescale_image_depthmap(
                        rgb, mask, intrinsics, (target_resolution, target_resolution * 3.0 / 4))

                    W, H = image.size
                    intrinsics = geometry.opencv_to_colmap_intrinsics(intrinsics)

                    # update inpace img_infos_idx
                    img_infos_idx['intrinsics'] = intrinsics
                    rgb_outpath = os.path.join(output_dir_scene_rgb, img_infos_idx['path'][:-3] + 'jpg')
                    image.save(rgb_outpath)

                    depth_outpath = os.path.join(output_dir_scene_depth, img_infos_idx['path'][:-3] + 'png')
                    # render depth image
                    renderer.viewport_width, renderer.viewport_height = W, H
                    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
                    camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy, znear=znear, zfar=zfar)
                    camera_node = pyrender_scene.add(camera, pose=img_infos_idx['cam_to_world'] @ OPENGL_TO_OPENCV)

                    depth = renderer.render(pyrender_scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
                    pyrender_scene.remove_node(camera_node)  # dont forget to remove camera

                    depth = (depth * 1000).astype('uint16')
                    # invalidate depth from mask before saving
                    depth_mask = (mask < 255)
                    depth[depth_mask] = 0
                    Image.fromarray(depth).save(depth_outpath)

                trajectories = []
                intrinsics = []
                img_names = []

                for imgidx in tqdm(img_infos.keys()):
                    img_infos_idx = img_infos[imgidx]
                    intrinsics.append(img_infos_idx['intrinsics'])
                    trajectories.append(img_infos_idx['cam_to_world'])
                    img_names.append(img_infos_idx['path'][:-3] + 'jpg')

                info_to_sort = list(zip(img_names, intrinsics, trajectories))
                sorted_info = sorted(info_to_sort, key=lambda x: int(x[0][-9:-4]))
                img_names, intrinsics, trajectories = zip(*sorted_info)
                
                #sort by img id of name
                intrinsics = np.stack(list(intrinsics), axis=0)
                trajectories = np.stack(list(trajectories), axis=0)
                img_names = list(img_names)
                # save metadata for this scene
                np.savez(out_metadata_path,
                            trajectories=trajectories,
                            intrinsics=intrinsics,
                            images=img_names)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            print(f"Error processing scene {scene}: {e}")
            with open(osp.join(output_dir, 'error_scenes.txt'), 'a') as f:
                f.write(f"{scene} {e}\n")


def check_processed_data():
    dslr_data = np.load("/data/yuzheng/data/scannetpp/seq_test/1a8e0d78c0/scene_iphone_metadata.npz")
    print(dslr_data['images'])
    print(len(dslr_data['images']))
    print(dslr_data['trajectories'].shape)
    print(dslr_data['intrinsics'].shape)
    

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.pyopengl_platform.strip():
        os.environ['PYOPENGL_PLATFORM'] = args.pyopengl_platform
    process_scenes(args.scannetpp_dir, args.output_dir, args.target_resolution)
    # check_processed_data()