# I2P and L2W Pipelines: Point Cloud Registration & Localization

## Overview

SLAM3R's core reconstruction relies on two complementary neural models working in sequence:

1. **Image-to-Points (I2P)**: Predicts depth from a single/multi-view image → local point cloud
2. **Local-to-World (L2W)**: Registers local depth into global world coordinates → camera pose

Together they form a **sliding-window SLAM** system where:
- **I2P** produces dense per-frame depth estimates
- **L2W** anchors these depths into a consistent world map

---

## I2P Pipeline: Image-to-Points (Local Depth Prediction)

### Purpose
Convert raw images into **local** point clouds in **camera coordinate frame** with confidence estimates.

### Input
- **Current frame**: RGB 224×224 image
- **Context frames** (optional): Previous frames for temporal coherence
- **Intrinsics** (implicit): Assumed camera focal length ≈ 224 pixels (normalized frame)

### Model Architecture

#### Encoder Stage: `_encode_multiview()`
```python
# From recon_online_pipeline.py
res_shapes, res_feats, res_poses = model._encode_multiview(
    views,                      # Current + context views
    view_batchsize=10,          # Batch size for encoding
    normalize=False,
    silent=False
)
```

**What it does:**
- Extracts multi-scale visual features from input images using Vision Transformer backbone
- Produces embedding tensors for each view
- Outputs: `res_shapes` (feature maps), `res_feats` (encoded features), `res_poses` (if known)

#### Decoder Stage: Depth + Confidence Prediction
**Per-view depth regression:**
```
Input:  224×224×3 RGB frame
  ↓
Feature Extraction (CNN/ViT backbone)
  ↓
Depth Head (DPT decoder)
  ↓
Output: 
  - pts3d_local: 224×224×3 (X, Y, Z in camera frame)
  - confidence_i2p: 224×224 (uncertainty/confidence per pixel)
```

### Output Format

#### Points (`pts3d_local`)
- **Shape**: (1, 224, 224, 3)
- **Coordinate System**: Camera-centric (camera at origin)
- **Z-axis**: Depth (distance from camera)
- **X, Y**: Horizontal/vertical pixel coordinates × depth

#### Confidence (`confidence_i2p`)
- **Shape**: (1, 224, 224)
- **Range**: [0, ∞) (higher = more confident)
- **Threshold**: `conf_thres_i2p = 1.5` (filter low-confidence pixels)

### Point Addition in I2P

#### Step 1: Per-Frame I2P Inference
```python
def process_input_frame(frame, model_i2p, device):
    """Run I2P model on single frame"""
    with torch.no_grad():
        output = model_i2p(frame)  # Forward pass
    
    pts3d_local = output['pts3d']        # (1, 224, 224, 3)
    conf_i2p = output['confidence']      # (1, 224, 224)
    
    return pts3d_local, conf_i2p
```

#### Step 2: Store in Sliding Window
```python
# api/services/tracker_service.py
self.per_frame_res['i2p_pcds'].append(pts3d_local)    # Local depth
self.per_frame_res['i2p_confs'].append(conf_i2p)     # Local confidence
```

#### Step 3: Confidence Filtering (Pre-L2W)
```python
# Filter out low-confidence points before L2W registration
conf_mask = (conf_i2p > conf_thres_i2p)  # Boolean mask

# Only keep high-confidence local points
valid_local_points = pts3d_local[conf_mask]  # (N,) → (M,) where M ≤ N
```

#### Step 4: Feed to L2W as Initial Guess
```python
# Input to L2W model
# Uses both:
# - Current frame's I2P local depth
# - Previous frames' registered world depths (as context)
l2w_input = {
    'current_i2p': pts3d_local,           # Fresh local depth
    'prev_registered': prev_world_points, # Historical context
}
```

### Why I2P Matters
- **Monocular depth**: No need for stereo/multi-view (though it can use temporal context)
- **Dense predictions**: Every pixel gets a depth estimate
- **Confidence-aware**: Uncertain pixels filtered before registration
- **Fast inference**: Single forward pass per frame

---

## L2W Pipeline: Local-to-World (Registration & Localization)

### Purpose
**Register** I2P local depth into **global world coordinates** + **estimate camera pose** (localization).

### Input
- **Current frame's local depth**: `pts3d_local` from I2P (224×224×3)
- **Previous frames' world depth**: `pts3d_world` from past L2W outputs (context window)
- **Previous camera poses**: (4×4 matrices) for geometric constraints

### Model Architecture

#### Registration Module: `pointmap_local_recon()`
```python
def pointmap_local_recon(views, milestone, args, device):
    """
    Fuse current local depth into accumulated point map
    
    Args:
        views: List of recent views with pts3d_local
        milestone: Current buffer milestone (tracking which frames used)
    
    Returns:
        views with pts3d_world: Registered to world coordinates
    """
```

**What it does:**
1. **Accumulates** I2P depth from multiple frames in sliding window
2. **Aligns** all local frames to a **common reference frame** coordinate system
3. **Registers** into global world map using:
   - **Feature correspondence**: Pixel-level feature matching
   - **Photometric consistency**: RGB similarity between frames
   - **Geometric constraints**: Epipolar geometry + pose priors

#### Camera Pose Estimation: `estimate_camera_pose_from_correspondences()`
```python
def estimate_camera_pose_from_correspondences(
    current_world_points,      # Registered 3D points in world frame
    prev_world_points,         # Previous frame points (known poses)
    intrinsics,                # Camera K matrix
    device
):
    """
    Estimate current camera pose via point-to-point correspondence
    
    Returns:
        c2w: 4×4 camera-to-world transformation matrix
        success: Boolean confidence flag
    """
```

**Algorithm:**
```
1. For each pixel in current frame:
   - Find closest match in previous frame (feature similarity)
   
2. Build correspondence set:
   X_current ↔ X_world (2D→3D matches)
   
3. Run PnP (Perspective-n-Point) solver:
   - Estimates camera rotation (R) and translation (t)
   - Uses RANSAC for outlier rejection
   
4. Output: c2w = [R | t; 0 1]
                  4×4 transformation
```

### Output Format

#### World Points (`pts3d_world`)
- **Shape**: (1, 224, 224, 3)
- **Coordinate System**: **Global world frame** (consistent across all frames)
- **Meaning**: If frame i and frame j both register to world frame, their overlap is geometrically valid

#### L2W Confidence (`confidence_l2w`)
- **Shape**: (1, 224, 224)
- **Range**: [0, ∞)
- **Interpretation**: How well registered to world map (higher = more anchored)
- **Threshold**: `conf_thres_l2w = 12.0` (stricter than I2P)

#### Camera Pose (`c2w`)
- **Shape**: (4, 4)
- **Meaning**: Where is the camera in world coordinates?
  ```
  [R | t]   = camera rotation + translation
  [0 | 1]     (homogeneous coordinates)
  
  X_world = c2w @ X_camera  (transform point from camera → world)
  ```

### Point Addition in L2W

#### Step 1: Accumulate I2P Depth into Local Scene
```python
def initial_scene_for_accumulated_frames(views, initial_winsize, args):
    """
    Take first N frames of I2P local depths
    Align to common local reference frame
    
    Steps:
    1. Select best reference frame (highest mean confidence)
    2. For each frame i:
       - pts3d_local[i] is depth in camera_i's frame
       - pts3d_aligned[i] = Transform(pts3d_local[i], cam_i→ref)
    3. All frames now share reference frame coordinates
    """
    
    ref_id = select_best_reference(views)  # Pick anchor frame
    
    for i in range(initial_winsize):
        if i == ref_id:
            views[i]['pts3d_aligned'] = views[i]['pts3d_local']
        else:
            # Estimate pose from i → ref_id
            pose_i_to_ref = estimate_pose_from_pcd(
                views[i]['pts3d_local'],
                views[ref_id]['pts3d_local']
            )
            # Transform to reference frame
            views[i]['pts3d_aligned'] = transform_points(
                views[i]['pts3d_local'],
                pose_i_to_ref
            )
    
    return views
```

**Result:** All frames now in same local coordinate system.

#### Step 2: Register into Global World Map
```python
def pointmap_global_register(views, args, device):
    """
    Transform all aligned local points to global world coordinates
    Accumulate into single consistent map
    """
    
    if len(accumulated_world_points) == 0:
        # First frames: use their local frame as world frame
        world_origin = views[0]['pts3d_aligned']
        c2w = np.eye(4)  # Identity: world = first frame's local
    else:
        # Subsequent frames: register to existing map
        for view in views:
            # Find best transformation to existing world points
            c2w = solve_registration(
                view['pts3d_aligned'],        # Current
                accumulated_world_points,    # Target (world)
                intrinsics
            )
            view['c2w'] = c2w
    
    # Transform to world
    for view in views:
        view['pts3d_world'] = transform_points(
            view['pts3d_aligned'],
            view['c2w']
        )
    
    # Accumulate
    accumulated_world_points.append(
        np.concatenate([v['pts3d_world'] for v in views])
    )
```

#### Step 3: Estimate L2W Confidence
```python
def compute_l2w_confidence(
    pts3d_world,          # Registered points
    prev_pts3d_world,     # Previous registered points
    intrinsics
):
    """
    Confidence = how well correlated this frame is to world map
    
    Metrics:
    - Reprojection error: reproject world points to image, check consistency
    - Feature match count: how many pixels have good correspondences
    - Depth stability: variance of depth estimates
    """
    
    # Project registered points back to image
    reprojected = project_to_image(pts3d_world, intrinsics)
    
    # Check if reprojected matches original
    reprojection_error = ||reprojected - original_pixels||
    
    # Confidence inversely proportional to error
    confidence_l2w = 1.0 / (1.0 + reprojection_error)
```

#### Step 4: Store Registered Points & Pose in Buffer
```python
self.per_frame_res['l2w_pcds'].append(pts3d_world)     # World coordinates
self.per_frame_res['l2w_confs'].append(confidence_l2w) # Registration quality

# Store camera pose
self._traj_positions.append(c2w[:3, 3].tolist())  # Camera position in world
self._traj_forwards.append(c2w[:3, 2].tolist())   # Forward direction
```

### Localization in L2W

#### Camera Localization = Pose Estimation
```
Goal: Given depth in world + image features, where is camera?

Process:
1. Current frame I2P → local depth (224×224×3)
2. World map has accumulated points from all previous frames
3. Find correspondences between:
   - Current frame features (2D)
   - World map points (3D)
4. Solve PnP:
   - Input: 3D point correspondences + intrinsics
   - Output: R, t (camera rotation + translation)
   - Format: c2w = [R | t; 0 1]

Confidence Check:
- If few correspondences found → tracking loss → degradation
- If correspondence residuals large → pose unreliable
```

#### Example: Camera Position Recovery
```python
# Given:
# - World map: 1M points with known 3D positions
# - Current frame RGB + I2P depth
# - Feature descriptors (from encoder)

# Find matches
matches_2d_3d = match_features(
    current_frame_features,
    world_map_features
)  # Returns: [(pix_x, pix_y) ↔ (X, Y, Z)]

# Solve PnP
R, t, inliers = cv2.solvePnPRansac(
    objectPoints=world_3d_points[matches],
    imagePoints=image_2d_pixels[matches],
    cameraMatrix=intrinsics,
    distCoeffs=np.zeros(4)
)

# Camera pose
c2w = np.vstack([
    np.hstack([R, t.reshape(3,1)]),
    [0, 0, 0, 1]
])

# Camera position in world
cam_pos_world = -R.T @ t
```

---

## Complete I2P → L2W Processing Flow

### Per-Frame Sequence

```
Frame N received
    ↓
[I2P STAGE]
    ├─ Extract RGB features
    ├─ Predict local depth: pts3d_local (camera frame)
    ├─ Predict confidence: conf_i2p
    └─ Filter low-confidence pixels
    ↓
[L2W STAGE]
    ├─ Accumulate with previous N frames
    ├─ Align all to reference frame (local registration)
    ├─ Register aligned points to world map (global registration)
    ├─ Estimate camera pose (localization): c2w
    ├─ Compute L2W confidence
    └─ Update trajectory + world map
    ↓
[DEGRADATION CHECK]
    ├─ Track mean(conf_l2w)
    ├─ If 10+ consecutive frames < 0.5 → START NEW SEGMENT
    └─ Otherwise → continue
    ↓
[SAVE OUTPUT]
    ├─ Store: pts3d_world, conf_l2w, c2w
    ├─ Update: accumulated map, trajectory
    └─ Every N frames → flush to disk
```

### Buffer State During Processing

```
Sliding Window (size = 200):

Frame 1:  pts3d_local[1] → aligned[1] → world[1], c2w[1]  ✓
Frame 2:  pts3d_local[2] → aligned[2] → world[2], c2w[2]  ✓
...
Frame N:  pts3d_local[N] → aligned[N] → world[N], c2w[N]  ✓ (current)
Frame N+1: pts3d_local[N+1] [waiting for L2W]
Frame N+2: [not yet read]

World Map (accumulated):
  - Concatenation of all pts3d_world[1:N]
  - Used as target for registering Frame N+1
```

---

## Confidence Thresholds & Filtering

| Stage | Variable | Threshold | Action |
|-------|----------|-----------|--------|
| **I2P Output** | `conf_i2p` | 1.5 | Pixels below this discarded before L2W |
| **L2W Output** | `conf_l2w` | 12.0 | Stricter; final point cloud quality metric |
| **Final Map** | `conf_thres_res` | 3.0 | Filtering for visualization/PLY export |
| **Degradation** | `mean(conf_l2w)` per frame | < 0.5 (10+ frames) | Trigger segment split |

---

## Multi-Frame Coordination

### How Frames Reference Each Other

#### I2P is **frame-independent**
- Each frame processed separately
- Can use temporal context (optional, for smoothing)
- No dependency on previous frame poses

#### L2W is **frame-interdependent**
- Current frame depth aligned to reference frame (from initial window)
- Reference frame used as intermediate coordinate system
- Then all frames registered together to world map

```
Frame 1, 2, 3 (initial window):
    I2P: local[1], local[2], local[3]
    L2W:
        Ref = Frame 1
        Align 1 → Ref (identity)
        Align 2 → Ref (via pose estimation)
        Align 3 → Ref (via pose estimation)
        All now in Frame1's coordinate system
    
    Then register all to world:
        world[1] = Ref coords (world origin)
        world[2] = transform(aligned[2], c2w[2])
        world[3] = transform(aligned[3], c2w[3])
```

---

## Degradation & Recovery

### Degradation Triggers
```python
# Track confidence over window
conf_window = deque(maxlen=10)
conf_window.append(mean(conf_l2w))

# Degradation check
if all(c < 0.5 for c in conf_window):
    print("DEGRADED: Starting new segment")
    # Reset: new reference frame, new world origin
```

### Recovery via Loop Closure
```python
# New segment's appearance descriptor
new_desc = mean_and_std_of_depth(pts3d_world)  # 6-D signature

# Compare to old segments
for old_segment in segments:
    similarity = cosine_similarity(new_desc, old_segment.desc)
    if similarity > 0.80:
        # Found loop closure!
        rel_pose = estimate_relative_pose(new_segment, old_segment)
        # Stitch: transform new segment to old segment's world
```

---

## Key Differences: I2P vs L2W

| Aspect | I2P | L2W |
|--------|-----|-----|
| **Input** | Single RGB frame | I2P depth + world map context |
| **Coordinate Frame** | Camera-local | World-global |
| **Confidence Meaning** | Per-pixel depth uncertainty | Registration quality to world |
| **Output** | Dense (224×224) depth map | Registered points + camera pose |
| **Dependency** | Independent per frame | Interdependent (accumulative) |
| **Localization Role** | Provides initial depth | **Estimates camera pose** |
| **Filtering** | `conf_thres_i2p = 1.5` | `conf_thres_l2w = 12.0` (stricter) |

---

## Summary: How Points Are Added

### I2P: **Dense Depth Prediction**
1. Image → Neural network → Per-pixel depth + confidence
2. Stored as (224, 224, 3) point cloud in camera frame
3. Low-confidence pixels filtered out

### L2W: **Registration & Accumulation**
1. Take filtered I2P points from current frame
2. Align to reference frame (using pose from previous frames)
3. Register aligned points to global world map
4. Estimate current camera pose via PnP on correspondences
5. Transform points to world coordinates
6. Accumulate into single world point cloud

### Position Localization

**I2P**: No explicit localization (per-frame, independent)

**L2W**: **Localization = Camera Pose Estimation**
- Feature matching between current frame and world map
- PnP solver to find camera position + orientation
- Output: 4×4 c2w matrix describing camera location in world
- Confidence: Reprojection error + number of valid correspondences
- Degradation detection: Track confidence; if drops for 10+ frames → new segment

