"""
SLAM3R API Testing Console — Gradio UI with per-frame streaming.

Frame upload
────────────
Use the "Reconstruction" tab.  Click the upload area and select one or
more frame images.  Frames are sorted by filename then sent to the API
in batches of exactly N (configurable via the "Frames per Request" slider).

Per-frame streaming
───────────────────
Each batch is sent as a separate multipart POST.  The generator yields
(log_text, depth_heatmap, last_json) after every batch so you can watch
the reconstruction progress in real time.
"""

import os
import mimetypes
import time
import tempfile
import requests
import gradio as gr
import numpy as np

API_DEFAULT = "http://localhost:8000"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(api_url, path, params=None):
    try:
        r = requests.get(f"{api_url}{path}", params=params or {}, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _delete(api_url, path):
    try:
        r = requests.delete(f"{api_url}{path}", timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _send_frames_batch(api_url: str, drone_id: str, file_paths: list, params: dict, map_id: str | None = None) -> dict:
    """Send a list of local file paths as a multipart POST."""
    url          = f"{api_url}/api/upload_frames/{drone_id}"
    open_handles = []
    try:
        file_tuples = []
        for src in file_paths:
            mime = mimetypes.guess_type(src)[0] or "image/jpeg"
            fh   = open(src, "rb")
            open_handles.append(fh)
            file_tuples.append(("files", (os.path.basename(src), fh, mime)))

        data = {**params}
        if map_id:
            data["map_id"] = map_id

        r = requests.post(url, files=file_tuples, data=data, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}
    finally:
        for fh in open_handles:
            fh.close()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint wrappers
# ─────────────────────────────────────────────────────────────────────────────

def api_get_health(api_url):  return _get(api_url or API_DEFAULT, "/api/health")
def api_get_info(api_url):    return _get(api_url or API_DEFAULT, "/api/info")
def api_get_status(api_url):  return _get(api_url or API_DEFAULT, "/api/status")
def api_list_drones(api_url): return _get(api_url or API_DEFAULT, "/api/drones")

def api_drone_status(api_url, drone_id):
    if not drone_id: return {"error": "Drone ID is required."}
    return _get(api_url or API_DEFAULT, f"/api/drones/{drone_id}")

def api_clear_drone(api_url, drone_id):
    if not drone_id: return {"error": "Drone ID is required."}
    return _delete(api_url or API_DEFAULT, f"/api/drones/{drone_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Frame upload helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sort_key(path: str) -> str:
    return os.path.basename(path).lower()


def filter_images(file_list: list) -> list:
    if not file_list:
        return []
    return sorted(
        [f for f in file_list if os.path.splitext(f)[1].lower() in IMAGE_EXTS],
        key=_sort_key,
    )


def preview_upload(file_list: list) -> str:
    imgs = filter_images(file_list)
    if not imgs:
        return "No image files found (supported: jpg, png, bmp, tiff, webp)."

    exts: dict = {}
    for f in imgs:
        ext = os.path.splitext(f)[1].lower()
        exts[ext] = exts.get(ext, 0) + 1
    ext_summary = ", ".join(f"{v}x {k}" for k, v in sorted(exts.items()))
    names = [os.path.basename(f) for f in imgs[:5]]
    tail  = f" ... (+{len(imgs) - 5} more)" if len(imgs) > 5 else ""
    return (
        f"{len(imgs)} images ready ({ext_summary})\n\n"
        f"First frames: {', '.join(names)}{tail}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sequential per-frame processing  (generator -> live UI updates)
# ─────────────────────────────────────────────────────────────────────────────

def process_frames_sequential(
    api_url, drone_id, uploaded_files, batch_size, map_id,
    keyframe_stride, initial_winsize, win_r,
    conf_thres_i2p, num_scene_frame, max_num_register,
    conf_thres_l2w, num_points_save, save_each_frame,
    progress=gr.Progress(track_tqdm=False),
):
    """
    Iterates over uploaded images, sending them batch_size at a time.
    If map_id is specified, all drones will be registered to the same map.
    Yields (log_text, summary_json) after every batch.
    """
    api_url = api_url or API_DEFAULT

    if not drone_id:
        yield "Drone ID is required.", {"error": "Drone ID is required."}
        return

    files = filter_images(uploaded_files or [])
    if not files:
        yield "No image files found in the upload.", {"error": "No images found."}
        return

    batch_size    = max(1, int(batch_size))
    total_batches = (len(files) + batch_size - 1) // batch_size

    params = dict(
        keyframe_stride=int(keyframe_stride),
        initial_winsize=int(initial_winsize),
        win_r=int(win_r),
        conf_thres_i2p=float(conf_thres_i2p),
        num_scene_frame=int(num_scene_frame),
        max_num_register=int(max_num_register),
        conf_thres_l2w=float(conf_thres_l2w),
        num_points_save=int(num_points_save),
        save_each_frame=str(save_each_frame).lower(),
    )

    log_lines:    list = []
    last_summary: dict = {}

    map_info = f" (map: {map_id})" if map_id else ""
    log_lines.append(
        f"Starting sequential processing of {len(files)} frames for drone '{drone_id}'{map_info} "
        f"in {total_batches} batches (batch_size={batch_size})."
    )
    yield "\n\n".join(log_lines), {}

    for batch_idx in range(total_batches):
        batch_files = files[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        names       = [os.path.basename(f) for f in batch_files]
        frame_range = f"{batch_idx * batch_size + 1}-{batch_idx * batch_size + len(batch_files)}"

        progress(batch_idx / total_batches, desc=f"Batch {batch_idx+1}/{total_batches} - frames {frame_range}")
        log_lines.append(
            f"Batch {batch_idx+1}/{total_batches} "
            f"(frames {frame_range}): {', '.join(names)}"
        )
        yield "\n\n".join(log_lines), last_summary

        t0      = time.time()
        resp    = _send_frames_batch(api_url, drone_id, batch_files, params, map_id=map_id or None)
        elapsed = time.time() - t0

        if "error" in resp:
            log_lines.append(f"  Error: {resp['error']}")
            yield "\n\n".join(log_lines), resp
            return

        last_summary = resp
        pts          = resp.get("num_points", "?")
        log_lines.append(f"  Done in {elapsed:.2f}s — points so far: {pts}")
        yield "\n\n".join(log_lines), last_summary

    progress(1.0, desc="Complete")
    log_lines.append(f"\nAll {len(files)} frames processed.")
    yield "\n\n".join(log_lines), last_summary


# ─────────────────────────────────────────────────────────────────────────────
# Point cloud
# ─────────────────────────────────────────────────────────────────────────────

def api_download_pointcloud(api_url, drone_id):
    if not drone_id:
        return None, {"error": "Drone ID is required."}
    try:
        # API always returns PLY format only
        r = requests.get(
            f"{(api_url or API_DEFAULT)}/api/pointcloud/{drone_id}",
            timeout=120,
        )
        r.raise_for_status()
        suffix = ".ply"
        tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(r.content); tmp.flush(); tmp.close()
        return tmp.name, {"message": f"Downloaded PLY for {drone_id}", "format": "ply", "bytes": len(r.content)}
    except Exception as e:
        return None, {"error": str(e)}


def api_fetch_pointcloud_and_trajectories(api_url, drone_id, confidence_threshold=None, max_display_pts=50000):
    import io
    base = api_url or API_DEFAULT

    try:
        r = requests.get(f"{base}/api/pointcloud/{drone_id}/trajectories", timeout=30)
        r.raise_for_status()
        trajectories = r.json()
    except Exception as e:
        return None, {"error": f"Failed to fetch trajectories: {e}"}

    try:
        # Fetch PLY file (API always returns PLY)
        r = requests.get(
            f"{base}/api/pointcloud/{drone_id}", timeout=120,
        )
        r.raise_for_status()
        # Load PLY using trimesh
        import trimesh
        buf = io.BytesIO(r.content)
        mesh = trimesh.load(buf, file_type='ply')
        points = np.asarray(mesh.vertices)
        # Extract vertex colors from PLY if available
        colors = mesh.visual.vertex_colors if hasattr(mesh.visual, 'vertex_colors') else None
        if colors is not None:
            colors = np.asarray(colors)
    except Exception as e:
        return None, {"error": f"Failed to fetch pointcloud: {e}"}

    try:
        import plotly.graph_objects as go

        fig     = go.Figure()
        # Downsample points for browser performance
        max_pts = int(max_display_pts) if max_display_pts else 50000
        if points.shape[0] > max_pts:
            idx  = np.random.choice(points.shape[0], max_pts, replace=False)
            pts  = points[idx]
            cols = colors[idx] if colors is not None else None
        else:
            pts  = points
            cols = colors

        marker_kw = dict(size=1, opacity=0.7, line=dict(width=0))
        if cols is not None and len(cols) > 0:
            # Handle colors from trimesh (already in 0-255 range)
            if cols.dtype == np.uint8:
                marker_kw["color"] = cols[:, :3].tolist() if cols.shape[1] >= 3 else cols.tolist()
            else:
                # Normalize if needed
                cols_normalized = (cols * 255).astype(np.uint8) if cols.max() <= 1.0 else cols.astype(np.uint8)
                marker_kw["color"] = cols_normalized[:, :3].tolist() if cols_normalized.shape[1] >= 3 else cols_normalized.tolist()

        # Remap so that X=right, Y=height, Z=forward (right-hand, Y-up convention).
        # Source data is assumed to be X=right, Y=forward, Z=up; we swap Y<->Z.
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 2], z=pts[:, 1],
            mode="markers", marker=marker_kw, name="pointcloud",
            hoverinfo="skip",
        ))

        # Extract trajectory frames from the new endpoint format
        frames = trajectories.get("frames", [])
        if frames:
            # Filter valid poses and extract positions/forwards, optionally by confidence
            valid_frames = [f for f in frames if f.get("valid")]
            if confidence_threshold is not None and confidence_threshold > 0:
                valid_frames = [f for f in valid_frames if f.get("confidence", 1.0) >= confidence_threshold]
            if valid_frames:
                pos = np.array([f["position"] for f in valid_frames])
                fwd = np.array([f["forward"] for f in valid_frames])
                
                # Plot trajectory as line with markers
                fig.add_trace(go.Scatter3d(
                    x=pos[:, 0], y=pos[:, 2], z=pos[:, 1],
                    mode="markers+lines",
                    marker=dict(size=4, color="red"),
                    line=dict(color="red", width=2),
                    name="trajectory",
                ))
                
                # Plot forward direction vectors
                if fwd.shape[0] > 0:
                    try:
                        fig.add_trace(go.Cone(
                            x=pos[:, 0], y=pos[:, 2], z=pos[:, 1],
                            u=fwd[:, 0], v=fwd[:, 2], w=fwd[:, 1],
                            sizemode="scaled", sizeref=1.0, anchor="tail",
                            colorscale=[[0, "orange"], [1, "orange"]], name="forward_vectors",
                        ))
                    except Exception:
                        pass

        axis_style = dict(
            showgrid=False,
            zeroline=False,
            showbackground=False,
            showspikes=False,
        )
        x_axis_style = dict(**axis_style, autorange="reversed")
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis=dict(title="X (right)", **x_axis_style),
                yaxis=dict(title="Y (height)", **axis_style),
                zaxis=dict(title="Z (forward)", **axis_style),
                camera=dict(
                    up=dict(x=0, y=1, z=0),
                    eye=dict(x=0, y=0.5, z=2),
                ),
            ),
            height=700,
            hovermode=False,
        )
        pts_displayed = pts.shape[0]
        pts_total = points.shape[0]
        traj_len = len([f for f in frames if f.get("valid")])
        return fig, {
            "message": "ok",
            "points_total": int(pts_total),
            "points_displayed": int(pts_displayed),
            "downsampled": pts_total != pts_displayed,
            "traj_len": traj_len,
        }
    except Exception as e:
        return None, {"error": f"Failed to build figure: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Shared pipeline parameter block
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline_params():
    with gr.Accordion("Pipeline Parameters", open=False):
        with gr.Row():
            ks  = gr.Slider(1, 20, value=1,  step=1, label="Keyframe Stride")
            iws = gr.Slider(2, 20, value=2,  step=1, label="Initial Window Size")
            wr  = gr.Slider(1, 10, value=3,  step=1, label="I2P Window Radius")
        with gr.Row():
            ci2p = gr.Number(value=1.5,              label="Conf Threshold I2P")
            nsf  = gr.Slider(1, 50, value=10, step=1, label="Num Scene Frames")
            mnr  = gr.Slider(1, 50, value=10, step=1, label="Max Frames to Register")
        with gr.Row():
            cl2w = gr.Number(value=12.0,              label="Conf Threshold L2W")
            nps  = gr.Number(value=2_000_000,         label="Num Points to Save")
            sef  = gr.Checkbox(value=True,            label="Save Each Frame")
    return ks, iws, wr, ci2p, nsf, mnr, cl2w, nps, sef


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def build_demo_ui():
    with gr.Blocks(
        title="SLAM3R API Testing Console",
        theme=gr.themes.Soft(primary_hue="orange", font="Inter"),
        css=".frame-log { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }",
    ) as demo:

        gr.Markdown(
            "# SLAM3R API Testing Console\n"
            "Upload one or more frame images for sequential per-frame processing.\n\n"
            "**Features:** FP16 mixed-precision acceleration | Multi-drone support | "
            "Loop closure detection | Segment-based trajectory stitching"
        )

        api_url = gr.Textbox(label="API Base URL", value=API_DEFAULT)

        # ── Health & Info ─────────────────────────────────────────────────
        with gr.Tab("Health & Info"):
            with gr.Row():
                health_btn = gr.Button("Health Check")
                info_btn   = gr.Button("Get API Info")
                status_btn = gr.Button("Get System Status")
            with gr.Row():
                health_out = gr.JSON(label="Health")
                info_out   = gr.JSON(label="Info")
                status_out = gr.JSON(label="Status")
            health_btn.click(api_get_health, api_url, health_out)
            info_btn.click(  api_get_info,   api_url, info_out)
            status_btn.click(api_get_status, api_url, status_out)

        # ── Drone Management ──────────────────────────────────────────────
        with gr.Tab("Drone Management"):
            drone_id_mgmt = gr.Textbox(label="Drone ID", placeholder="drone_1")
            with gr.Row():
                list_btn          = gr.Button("List Active Drones")
                drone_status_btn  = gr.Button("Get Drone Status")
                clear_btn         = gr.Button("Clear Drone", variant="stop")
            with gr.Row():
                list_out          = gr.JSON(label="Drone List")
                drone_status_out  = gr.JSON(label="Drone Status")
                clear_out         = gr.JSON(label="Clear Result")
            list_btn.click(        api_list_drones,  api_url,                  list_out)
            drone_status_btn.click(api_drone_status, [api_url, drone_id_mgmt], drone_status_out)
            clear_btn.click(       api_clear_drone,  [api_url, drone_id_mgmt], clear_out)

        # ── Reconstruction ────────────────────────────────────────────────
        with gr.Tab("Reconstruction"):
            gr.Markdown(
                "Select one or more frame images. "
                "Files are sorted by filename and sent to the API in batches of N frames. "
                "Processing progress and responses are shown below."
            )

            with gr.Row():
                recon_drone_id  = gr.Textbox(label="Drone ID", value="demo_drone", scale=2)
                recon_map_id    = gr.Textbox(label="Map ID (optional, for multi-drone stitching)", value="", scale=2)
                batch_size_ctrl = gr.Slider(
                    minimum=1, maximum=20, value=1, step=1,
                    label="Frames per Request",
                    scale=3,
                )

            frame_upload = gr.File(
                label="Frame Images — select one or more files",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
            )
            upload_info = gr.Markdown(value="")
            frame_upload.change(preview_upload, frame_upload, upload_info)

            rks, riws, rwr, rci2p, rnsf, rmnr, rcl2w, rnps, rsef = _pipeline_params()

            run_btn = gr.Button("Process Frames", variant="primary", size="lg")

            frame_log    = gr.Markdown(label="Progress Log", elem_classes=["frame-log"])
            recon_summary = gr.JSON(label="Last Batch Response")

            run_btn.click(
                process_frames_sequential,
                inputs=[
                    api_url, recon_drone_id, frame_upload,
                    batch_size_ctrl, recon_map_id,
                    rks, riws, rwr, rci2p, rnsf, rmnr, rcl2w, rnps, rsef,
                ],
                outputs=[frame_log, recon_summary],
            )

        # ── Point Cloud ───────────────────────────────────────────────────
        with gr.Tab("Point Cloud"):
            gr.Markdown("Download and visualize the latest point cloud for a drone.")
            
            # Download section
            with gr.Accordion("Download", open=True):
                with gr.Row():
                    pc_drone_id = gr.Textbox(label="Drone ID", value="demo_drone", scale=2)
                pc_btn = gr.Button("Download Point Cloud", variant="primary")
                with gr.Row():
                    pc_file = gr.File(label="Downloaded File")
                    pc_out  = gr.JSON(label="Response")
                pc_btn.click(
                    api_download_pointcloud,
                    inputs=[api_url, pc_drone_id],
                    outputs=[pc_file, pc_out],
                )
            
            # Visualization section
            with gr.Accordion("Visualization", open=True):
                with gr.Row():
                    viz_drone_id = gr.Textbox(label="Drone ID", value="demo_drone", scale=2)
                    viz_conf_threshold = gr.Number(label="Confidence Threshold", value=0.0, minimum=0, maximum=1, step=0.05, scale=1, info="Filter trajectory by confidence (0=show all)")
                with gr.Row():
                    viz_max_pts = gr.Slider(label="Max Points to Display", minimum=10000, maximum=200000, value=50000, step=10000, info="Lower = faster rendering")
                viz_btn  = gr.Button("Plot Pointcloud + Trajectory", variant="primary")
                viz_plot = gr.Plot()
                viz_out  = gr.JSON(label="Viz Info")
                viz_btn.click(
                    api_fetch_pointcloud_and_trajectories,
                    inputs=[api_url, viz_drone_id, viz_conf_threshold, viz_max_pts],
                    outputs=[viz_plot, viz_out],
                )

    return demo


if __name__ == "__main__":
    build_demo_ui().launch(debug=True, share=False)