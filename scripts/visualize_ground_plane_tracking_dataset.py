import argparse
import os
import time
from pathlib import Path

import colorhash
import cv2
import numpy as np
import pandas as pd

from ground_plane_tracking.data_loading import load_samples
from utils.projection import load_calibration

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent.parent / "DATA" / "MVBroTrack"


def _track_color(track_id, cache):
    tid = int(track_id)
    if tid not in cache:
        cache[tid] = colorhash.ColorHash(tid).rgb
    return cache[tid]


def _gt_by_frame(gt_df):
    """O(1) lookup per frame instead of filtering the full DataFrame each time."""
    return {int(f): g for f, g in gt_df.groupby("FrameId")}


def _image_tracks_by_frame(image_plane_tracks):
    """Per camera: frame_id -> DataFrame slice."""
    if image_plane_tracks is None:
        return None
    out = {}
    for cam_id, df in image_plane_tracks.items():
        out[cam_id] = {int(f): g for f, g in df.groupby("FrameId")}
    return out


def _get_base_sample_number(sample_number):
    """Extract base sample number from chunked id like '1_1' -> 1."""
    s = str(sample_number)
    return int(s.split('_')[0]) if '_' in s else int(s)


def visualize_samples(
    samples,
    calibs,
    cam_masks,
    start_sample_idx=0,
    start_frame=1,
):
    """
    Visualize the chunked samples in a 2x2 grid with bounding boxes drawn on each frame.
    Press 'n' to go to next sample, 'b' to toggle YOLO detections, 'v' to toggle image tracks (samples 1-3).

    Video frames are read sequentially (one seek per camera at chunk start). Random seeking every
    frame was the main reason playback used to feel slow.
    """
    playback_fps = 25.0
    frame_period = 1.0 / playback_fps

    cv2.namedWindow("Multi-View Visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-View Visualization", 1920, 1080)

    sample_idx = start_sample_idx % len(samples)
    total_samples = len(samples)
    color_cache = {}

    while True:
        sample = samples[sample_idx]
        video_paths = sample["video_paths"]
        max_frames = sample["max_valid_frame_for_sample"]
        sample_name = sample["sample_number"]
        base_sample_num = _get_base_sample_number(sample_name)
        has_image_tracks = sample.get("image_plane_tracks") is not None

        gt_by_frame = _gt_by_frame(sample["ground_truth_df"])
        tracks_by_frame = _image_tracks_by_frame(sample.get("image_plane_tracks"))

        # Open video captures for this sample (fixed camera order for stable 2x2 layout)
        captures = {}
        for cam_id in sorted(video_paths.keys(), key=lambda k: int(k)):
            path = video_paths[cam_id]
            if os.path.exists(path):
                captures[cam_id] = cv2.VideoCapture(path)
            else:
                print(f"Warning: Video file not found: {path}")

        if not captures:
            print("No valid video files found")
            sample_idx = (sample_idx + 1) % total_samples
            continue

        # One seek per camera: first video index for this chunk / start_frame (then sequential read only)
        start_f = max(1, min(int(start_frame), max_frames))
        video_base = sample["original_start_frame"] + start_f - 1
        for cam_id in sorted(captures.keys(), key=lambda k: int(k)):
            cap = captures[cam_id]
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_base)

        # Toggle for showing YOLO detections (press 'b'); start with YOLO OFF by default
        show_yolo_dets = False
        # Toggle for image-plane tracks (press 'v'); default ON when available
        show_image_tracks = has_image_tracks

        next_tick = time.perf_counter()

        for frame_idx in range(start_f, max_frames + 1):
            grid = np.zeros((1080 * 2, 1920 * 2, 3), dtype=np.uint8)
            frame_gt = gt_by_frame.get(frame_idx)
            if frame_gt is None:
                frame_gt = sample["ground_truth_df"].iloc[0:0]

            for i, cam_id in enumerate(sorted(captures.keys(), key=lambda k: int(k))):
                cap = captures[cam_id]
                ret, frame = cap.read()

                if not ret:
                    print(f"Failed to read frame {frame_idx} from camera {cam_id}")
                    continue

                cam_num = int(cam_id)

                if show_yolo_dets:
                    if cam_id in sample["detections"] and frame_idx in sample["detections"][cam_id]:
                        det_file_path = sample["detections"][cam_id][frame_idx]
                        if os.path.exists(det_file_path):
                            with open(det_file_path, "r") as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 6:
                                        x_center = float(parts[1])
                                        y_center = float(parts[2])
                                        width = float(parts[3])
                                        height = float(parts[4])
                                        conf = float(parts[5])

                                        x1 = int(x_center - width / 2)
                                        y1 = int(y_center - height / 2)
                                        x2 = int(x_center + width / 2)
                                        y2 = int(y_center + height / 2)

                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                        label = f"Conf: {conf:.2f}"
                                        cv2.putText(
                                            frame,
                                            label,
                                            (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            (0, 255, 0),
                                            1,
                                        )

                if show_image_tracks and tracks_by_frame and cam_id in tracks_by_frame:
                    frame_tracks = tracks_by_frame[cam_id].get(frame_idx)
                    if frame_tracks is not None:
                        for _, tr in frame_tracks.iterrows():
                            tid = int(tr["Id"])
                            x1, y1 = int(tr["X"]), int(tr["Y"])
                            x2, y2 = int(x1 + tr["Width"]), int(y1 + tr["Height"])
                            tc = _track_color(tid, color_cache)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), tc, 2)

                mask = cam_masks[cam_num]
                calib = calibs[cam_num]
                for _, row in frame_gt.iterrows():
                    x, y = int(row["X"]), int(row["Y"])
                    track_id = int(row["Id"])
                    tc = _track_color(track_id, color_cache)
                    proj_pt = np.array([[x / 100, y / 100, 0]], dtype=np.float32)
                    if mask[int(y), int(x)][0] > 0:
                        proj_pt, _ = cv2.projectPoints(
                            proj_pt,
                            calib["rvec"],
                            calib["tvec"],
                            calib["cameraMatrix"],
                            calib["distCoeffs"],
                        )

                        proj_x = int(proj_pt[0][0][0])
                        proj_y = int(proj_pt[0][0][1])

                        if 0 < proj_x < frame.shape[1] and 0 < proj_y < frame.shape[0]:
                            cv2.drawMarker(
                                frame,
                                (proj_x, proj_y),
                                (255, 255, 255),
                                markerType=cv2.MARKER_DIAMOND,
                                markerSize=20,
                                thickness=7,
                            )
                            cv2.circle(frame, (proj_x, proj_y), 6, tc, -1)

                row, col = i // 2, i % 2
                y_offset, x_offset = row * 1080, col * 1920
                try:
                    grid[y_offset : y_offset + frame.shape[0], x_offset : x_offset + frame.shape[1]] = frame
                except ValueError as e:
                    print(f"Error placing frame in grid: {e}")
                    print(
                        f"Frame shape: {frame.shape}, Grid section: {y_offset}:{y_offset+frame.shape[0]}, {x_offset}:{x_offset+frame.shape[1]}"
                    )

            y_line = grid.shape[0] - 110
            cv2.putText(
                grid,
                f"Sample {sample_idx+1}/{total_samples} ({sample_name})  [~{playback_fps:g} fps]",
                (50, y_line),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            if base_sample_num in (1, 2, 3):
                cv2.putText(
                    grid,
                    "This sample has ground truth image tracks (shown by default, toggle with 'v')",
                    (50, y_line + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 255),
                    2,
                )
            y_line += 70
            cv2.putText(
                grid,
                f"Frame: {frame_idx}/{max_frames}",
                (50, y_line),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            toggle_str = f"YOLO: {'ON' if show_yolo_dets else 'OFF'} (b) | Next: n"
            if has_image_tracks:
                toggle_str += f" | Image tracks: {'ON' if show_image_tracks else 'OFF'} (v)"
            cv2.putText(
                grid,
                toggle_str,
                (50, y_line + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Multi-View Visualization", grid)

            next_tick += frame_period
            delay_ms = max(1, int(round((next_tick - time.perf_counter()) * 1000)))

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                for cap in captures.values():
                    cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord("n"):
                for cap in captures.values():
                    cap.release()
                sample_idx = (sample_idx + 1) % total_samples
                next_tick = time.perf_counter()
                break
            elif key == ord("b"):
                show_yolo_dets = not show_yolo_dets
            elif key == ord("v") and has_image_tracks:
                show_image_tracks = not show_image_tracks

        else:
            # Finished all frames in this chunk without pressing 'n'
            for cap in captures.values():
                cap.release()
            sample_idx = (sample_idx + 1) % total_samples
            next_tick = time.perf_counter()
            continue

        # Reached via 'n' break — next_tick already reset
        continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize the Multi-View Ground-Plane Tracking dataset."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to the MVBroTrack root folder (relative to the current working directory or absolute).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help=(
            "First frame to play within each chunk (1-based, default: 1). "
            "Use e.g. 400 to skip the first part of a chunk with one seek instead of watching from frame 1."
        ),
    )
    args = parser.parse_args()

    # Convert provided path to an absolute path relative to the current working directory
    dataset_root_path = Path(args.dataset_root).resolve()
    if not dataset_root_path.is_dir():
        # raise SystemExit(f"Dataset root directory does not exist: {dataset_root_path}")
        print("Dataset root directory does not exist.")
        print(f"Checked absolute path: {dataset_root_path}")
        raise SystemExit(1)

    dataset_root = str(dataset_root_path)

    cam_nums = [9, 10, 11, 12]
    calibs = {cam: load_calibration(cam, calibration_root=os.path.join(dataset_root, 'Calibrations')) for cam in cam_nums}

    dets_string = 'yolo11x_dets'
    samples = load_samples(os.path.join(dataset_root, 'Multi-View-Tracking'), dets_string, 1)
    samples.sort(key=lambda x: x['sample_number'])

    # Informative prints about what this script visualizes
    print("Multi-View Ground-Plane Tracking visualizer")
    print(" - Dataset: 6 samples (~30 seconds each)")
    print(" - All samples have:")
    print("     * Ground-plane tracks (gt_groundplane_sample_*.txt)")
    print(f"     * YOLO detections folder: '{dets_string}' (per camera, per frame)")
    print(" - Samples 1, 2, 3 additionally have per-camera image-plane tracks in 'image_tracks/'")
    print(" - Keyboard controls:")
    print("     * 'b': toggle YOLO detections ON/OFF (starts OFF)")
    print("     * 'v': toggle image-plane tracks ON/OFF (only for samples 1–3)")
    print("     * 'n': go to next chunked sample")
    print("     * 'q': quit viewer")
    print("Playback pacing: ~25 fps (fixed)")
    print(f"Number of chunked samples loaded: {len(samples)}")

    cam_mask_root = os.path.join(dataset_root, 'Reprojection_Masks')
    cam_masks = {
        9: cv2.imread(os.path.join(cam_mask_root, 'cam_9_projected.jpg')),
        10: cv2.imread(os.path.join(cam_mask_root, 'cam_10_projected.jpg')),
        11: cv2.imread(os.path.join(cam_mask_root, 'cam_11_projected.jpg')),
        12: cv2.imread(os.path.join(cam_mask_root, 'cam_12_projected.jpg'))
    }

    start_idx = 0
    visualize_samples(
        samples,
        calibs,
        cam_masks,
        start_sample_idx=start_idx,
        start_frame=args.start_frame,
    )