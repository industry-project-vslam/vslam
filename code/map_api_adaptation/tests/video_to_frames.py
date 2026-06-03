import argparse
import cv2
from pathlib import Path


def extract_frames(video_path: Path, output_dir: Path, skip: int, start: int = 0, end: int = None, prefix: str = 'frame'):
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = total_frames if end is None or end < 0 else min(end, total_frames)
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= start and frame_idx < end_frame and ((frame_idx - start) % skip == 0):
            out_path = output_dir / f"{prefix}_{saved:06d}.png"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1
        if end_frame is not None and frame_idx >= end_frame:
            break

    cap.release()
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract video frames with frame skipping')
    parser.add_argument('video', type=Path, help='Input video file path')
    parser.add_argument('output_dir', type=Path, help='Directory to save extracted frames')
    parser.add_argument('--skip', type=int, default=1, help='Save every Nth frame (default: 1)')
    parser.add_argument('--start', type=int, default=0, help='Start frame index (default: 0)')
    parser.add_argument('--end', type=int, default=-1, help='End frame index (exclusive, default: end of video)')
    parser.add_argument('--prefix', type=str, default='frame', help='Output frame filename prefix')

    args = parser.parse_args()
    saved_count = extract_frames(args.video, args.output_dir, args.skip, args.start, args.end, args.prefix)
    print(f"Saved {saved_count} frames to {args.output_dir}")
