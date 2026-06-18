"""
Convert a video to frames and run SLAM3R on them.
Usage:
    python video_to_slam3r.py --video path/to/video.mp4 --test_name my_scene
    python video_to_slam3r.py --video path/to/video.mp4 --test_name my_scene --fps 10
"""
import os
import sys
import argparse
import subprocess
import cv2

def extract_frames(video_path, out_dir, target_fps=10):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, round(src_fps / target_fps))
    
    print(f"Video: {src_fps:.1f} fps, {total} frames → saving every {interval} frame (~{target_fps} fps)")
    
    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame_{saved:06d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"Extracted {saved} frames to {out_dir}")
    return saved

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--test_name", required=True, help="Name for this run")
    parser.add_argument("--fps", type=float, default=10, help="Target FPS for extraction (default: 10)")
    parser.add_argument("--save_dir", default="results", help="Where to save results")
    # pass any extra recon.py args through
    args, extra = parser.parse_known_args()

    frames_dir = f"data/video_frames/{args.test_name}"
    n = extract_frames(args.video, frames_dir, args.fps)
    
    if n < 3:
        print("ERROR: fewer than 3 frames extracted, check your video path.")
        sys.exit(1)

    cmd = [
        sys.executable, "recon.py",
        "--img_dir", frames_dir,
        "--test_name", args.test_name,
        "--save_dir", args.save_dir,
    ] + extra

    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)