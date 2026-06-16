# Mapping Service — API Reference

## Overview

A Dockerised service for incremental 3D map reconstruction from a single camera feed. New frames are ingested in real time and integrated into a shared map.

**Pipeline summary:**

- **Image-to-Point (I2P):** Depth estimation maps each frame into 3D space.
- **Local-to-World (L2W):** Camera localisation is performed over a sliding window of N frames.
- **Scale note:** The pipeline operates without explicit camera calibration. Reconstructed distances are internally consistent but not metric (i.e. not in real-world units).

---

## Base URL

All endpoints are relative to the service root (e.g. `http://localhost:8000`).

---

## Drones

Drones represent individual camera sources. Each drone maintains its own reconstruction state and output directory.

### `GET /api/drones`

Returns a list of all currently registered drones.

---

### `GET /api/drones/{drone_id}`

Returns the current status of a specific drone, including output directory, frame count, and processing state.

---

### `DELETE /api/drones/{drone_id}`

Removes a drone from the registry. Output files on disk are preserved.

---

## Frames

### `POST /api/process_source`

Triggers reconstruction from a server-side directory of frames. Use this when frames are already available on the server filesystem.

**Request body (JSON)**

| Parameter | Type | Description |
|---|---|---|
| `drone_id` | string | Identifier for the drone. |
| `source_path` | string | Absolute path to the frame directory on the server. |
| `map_id` | string _(optional)_ | Shared map ID for multi-drone stitching. |
| `frame_timeout_secs` | float | Per-frame processing timeout in seconds. Frames exceeding this are skipped. Default: `60.0`. |
| `keyframe_stride` | int | Step size between selected keyframes. Default: `1`. |
| `initial_winsize` | int | Window size used during initial localisation. Default: `2`. |
| `win_r` | int | Localisation window radius. Default: `3`. |
| `conf_thres_i2p` | float | Confidence threshold for I2P depth estimation. Default: `1.5`. |
| `num_scene_frame` | int | Number of frames used to initialise the scene. Default: `10`. |
| `max_num_register` | int | Maximum frames used per registration step. Default: `10`. |
| `conf_thres_l2w` | float | Confidence threshold for L2W localisation. Default: `12.0`. |
| `num_points_save` | int | Maximum number of points to retain in the saved point cloud. Default: `2000000`. |
| `save_frequency` | int | Interval (in frames) at which intermediate results are saved. Default: `3`. |
| `buffer_size` | int | Frame buffer capacity. Default: `100`. |
| `buffer_strategy` | string | Buffer eviction strategy (`"reservoir"` or other). Default: `"reservoir"`. |
| `seed` | int | Random seed for reproducibility. Default: `11`. |
| `device` | string | Compute device (`"cuda"` or `"cpu"`). Default: `"cuda"`. |

Additional boolean flags: `norm_input`, `save_each_frame`, `save_online`, `save_all_views`, `save_preds`, `save_for_eval`.

Adaptive keyframe parameters: `keyframe_adapt_min`, `keyframe_adapt_max`, `keyframe_adapt_stride`.

**Response fields**

| Field | Type | Description |
|---|---|---|
| `drone_id` | string | Echo of the drone identifier. |
| `save_dir` | string | Path to the output directory on the server. |
| `frame_count` | int | Total number of frames processed. |
| `last_frame` | int | Index of the final processed frame. |
| `position` | float[3] | Estimated camera position at the final frame (x, y, z). |
| `forward` | float[3] | Estimated camera forward vector at the final frame. |
| `valid` | bool | Whether localisation succeeded for the final frame. |
| `skipped_frames` | int[] | Indices of frames skipped due to timeout. |
| `map_id` | string | Map identifier, if provided. |
| `message` | string | Human-readable status summary. |

---

### `POST /api/upload_frames/{drone_id}`

Accepts uploaded frame images and runs reconstruction. Use this when frames originate from a client rather than the server filesystem.

**Request (multipart/form-data)**

| Field | Type | Description |
|---|---|---|
| `files` | file[] | One or more image files (`.jpg` or `.png`). |
| `map_id` | string _(optional)_ | Shared map ID for multi-drone stitching. |
| `frame_timeout_secs` | float _(optional)_ | Per-frame timeout in seconds. |
| _(pipeline params)_ | various _(optional)_ | Same parameters as `/api/process_source`. |

**Response:** Same structure as `/api/process_source`.

---

### `POST /api/cancel/{drone_id}`

Requests cancellation of an in-progress reconstruction job. The cancellation is asynchronous; the job may complete a current frame before halting.

---

### `POST /api/set_frame_timeout/{drone_id}`

Sets or clears the per-frame processing timeout for a drone. Pass `null` to disable the timeout entirely.

**Request body (JSON)**

| Field | Type | Description |
|---|---|---|
| `timeout_secs` | float \| null | Timeout in seconds, or `null` to disable. |

---

### `GET /api/skipped_frames/{drone_id}`

Returns the list of frame indices skipped due to timeout during the most recent reconstruction run.

---

## Point Cloud

### `GET /api/pointcloud/{drone_id}`

Downloads the latest point cloud for a drone as a `.ply` file.

---

### `GET /api/pointcloud/{drone_id}/trajectories`

Returns all trajectory frames for a drone: estimated camera positions and orientations across the reconstruction.

---

## Status & Health

### `GET /api/health`

Basic liveness check. Returns `200 OK` with service metadata when the service is running.

---

### `GET /api/status`

Returns a summary of all currently active drone maps and their processing state.

**`processing_status` values**

| Value | Meaning |
|---|---|
| `"idle"` | No active reconstruction job. |
| `"processing"` | Reconstruction is in progress. |
| `"cancelling"` | Cancellation has been requested; job is winding down. |

---

### `GET /api/info`

Returns API metadata and a full catalogue of available endpoints.