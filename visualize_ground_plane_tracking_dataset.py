import cv2
import pandas as pd
import pickle
import os
import colorhash
import numpy as np
from tqdm import tqdm
import pickle
import os
import cv2
import numpy as np
import sys
import argparse
from pathlib import Path
from utils.projection import load_calibration
from ground_plane_tracking.data_loading import load_samples

def _get_base_sample_number(sample_number):
    """Extract base sample number from chunked id like '1_1' -> 1."""
    s = str(sample_number)
    return int(s.split('_')[0]) if '_' in s else int(s)


def visualize_samples(samples, calibs, cam_masks, start_sample_idx=0):
    """
    Visualize the chunked samples in a 2x2 grid with bounding boxes drawn on each frame.
    Press 'n' to go to next sample, 'b' to toggle YOLO detections, 'v' to toggle image tracks (samples 1-3).
    """
    cv2.namedWindow('Multi-View Visualization', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Multi-View Visualization', 1920, 1080)

    sample_idx = start_sample_idx % len(samples)
    total_samples = len(samples)

    while True:
        sample = samples[sample_idx]
        video_paths = sample['video_paths']
        gt_df = sample['ground_truth_df']
        max_frames = sample['max_valid_frame_for_sample']
        sample_name = sample['sample_number']
        base_sample_num = _get_base_sample_number(sample_name)
        has_image_tracks = sample.get('image_plane_tracks') is not None

        # Open video captures for this sample
        captures = {}
        for cam_id, path in video_paths.items():
            if os.path.exists(path):
                captures[cam_id] = cv2.VideoCapture(path)
                captures[cam_id].set(cv2.CAP_PROP_POS_FRAMES, sample['original_start_frame'])
            else:
                print(f"Warning: Video file not found: {path}")

        if not captures:
            print("No valid video files found")
            sample_idx = (sample_idx + 1) % total_samples
            continue

        # Toggle for showing YOLO detections (press 'b'); start with YOLO OFF by default
        show_yolo_dets = False
        # Toggle for image-plane tracks (press 'v'); default ON when available
        show_image_tracks = has_image_tracks

        # Process frames for this sample
        # for frame_idx in tqdm(range(1, max_frames + 1), desc="Processing frames"):
        for frame_idx in tqdm(range(int(max_frames/2), max_frames + 1), desc=f"Sample {sample_name}"):
            # Get ground truth for this frame
            frame_gt = gt_df[gt_df['FrameId'] == frame_idx]

            # Create a grid for the 4 cameras
            grid = np.zeros((1080*2, 1920*2, 3), dtype=np.uint8)

            # Process each camera
            for i, (cam_id, cap) in enumerate(captures.items()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, sample['original_start_frame'] + frame_idx - 1)

                # Set position in the grid
                row, col = i // 2, i % 2
                y_offset, x_offset = row * 1080, col * 1920

                # Read frame
                ret, frame = cap.read()

                if not ret:
                    print(f"Failed to read frame {frame_idx} from camera {cam_id}")
                    continue

                cam_num = int(cam_id)

                # Draw YOLO detections for this camera (can be toggled on/off with 'b')
                if show_yolo_dets:
                    if cam_id in sample['detections'] and frame_idx in sample['detections'][cam_id]:
                        det_file_path = sample['detections'][cam_id][frame_idx]
                        if os.path.exists(det_file_path):
                            # Read YOLO format detections (class x_center y_center width height confidence)
                            with open(det_file_path, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 6:  # Make sure we have all required values
                                        class_id = int(parts[0])
                                        x_center = float(parts[1]) #* frame.shape[1]
                                        y_center = float(parts[2]) #* frame.shape[0]
                                        width = float(parts[3]) #* frame.shape[1]
                                        height = float(parts[4]) #* frame.shape[0]
                                        conf = float(parts[5])

                                        # Calculate bounding box coordinates
                                        x1 = int(x_center - width/2)
                                        y1 = int(y_center - height/2)
                                        x2 = int(x_center + width/2)
                                        y2 = int(y_center + height/2)

                                        # Draw the bounding box
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                        # Draw class label and confidence
                                        label = f"Conf: {conf:.2f}"
                                        cv2.putText(frame, label, (x1, y1-10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw image-plane ground truth tracks (samples 1-3 only, toggle with 'v')
                if show_image_tracks and sample.get('image_plane_tracks') is not None and cam_id in sample['image_plane_tracks']:
                    frame_tracks = sample['image_plane_tracks'][cam_id][sample['image_plane_tracks'][cam_id]['FrameId'] == frame_idx]
                    for _, tr in frame_tracks.iterrows():
                        tid = int(tr['Id'])
                        # X, Y are top-left; Width, Height are bbox dimensions
                        x1, y1 = int(tr['X']), int(tr['Y'])
                        x2, y2 = int(x1 + tr['Width']), int(y1 + tr['Height'])
                        track_color = colorhash.ColorHash(tid).rgb
                        cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, 2)

                # Draw ground truth reprojections
                for _, row in frame_gt.iterrows():
                    x, y = int(row['X']), int(row['Y'])
                    track_id = int(row['Id'])
                    track_color = colorhash.ColorHash(track_id).rgb
                    proj_pt = np.array([[x/100, y/100, 0]], dtype=np.float32)
                    if cam_masks[cam_num][int(y), int(x)][0] > 0:
                        proj_pt, _ = cv2.projectPoints(proj_pt, calibs[cam_num]['rvec'], calibs[cam_num]['tvec'], calibs[cam_num]['cameraMatrix'], calibs[cam_num]['distCoeffs'])

                        proj_x = int(proj_pt[0][0][0])
                        proj_y = int(proj_pt[0][0][1])

                        if 0 < proj_x < frame.shape[1] and 0 < proj_y < frame.shape[0]:
                            cv2.drawMarker(frame, (proj_x, proj_y), (255,255,255), markerType=cv2.MARKER_DIAMOND, markerSize=20, thickness=7)
                            cv2.circle(frame, (proj_x, proj_y), 6, track_color, -1)
                try:
                    grid[y_offset:y_offset+frame.shape[0], x_offset:x_offset+frame.shape[1]] = frame
                except ValueError as e:
                    print(f"Error placing frame in grid: {e}")
                    print(f"Frame shape: {frame.shape}, Grid section: {y_offset}:{y_offset+frame.shape[0]}, {x_offset}:{x_offset+frame.shape[1]}")
            
            # Add overlay: sample, frame, toggles
            y_line = grid.shape[0] - 110
            cv2.putText(grid, f"Sample {sample_idx+1}/{total_samples} ({sample_name})", (50, y_line),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            if base_sample_num in (1, 2, 3):
                cv2.putText(grid, "This sample has ground truth image tracks (shown by default, toggle with 'v')",
                            (50, y_line + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
            y_line += 70
            cv2.putText(grid, f"Frame: {frame_idx}", (50, y_line),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            toggle_str = f"YOLO: {'ON' if show_yolo_dets else 'OFF'} (b) | Next: n"
            if has_image_tracks:
                toggle_str += f" | Image tracks: {'ON' if show_image_tracks else 'OFF'} (v)"
            cv2.putText(grid, toggle_str, (50, y_line + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the grid
            cv2.imshow('Multi-View Visualization', grid)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                for cap in captures.values():
                    cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                for cap in captures.values():
                    cap.release()
                sample_idx = (sample_idx + 1) % total_samples
                break  # break frame loop, re-enter outer loop with next sample
            elif key == ord('b'):
                show_yolo_dets = not show_yolo_dets
            elif key == ord('v') and has_image_tracks:
                show_image_tracks = not show_image_tracks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize the Multi-View Ground-Plane Tracking dataset."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="MVBroTrack",
        help="Path to the MVBroTrack root folder (relative to the current working directory or absolute).",
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
    print(f"Number of chunked samples loaded: {len(samples)}")

    cam_mask_root = os.path.join(dataset_root, 'Reprojection_Masks')
    cam_masks = {
        9: cv2.imread(os.path.join(cam_mask_root, 'cam_9_projected.jpg')),
        10: cv2.imread(os.path.join(cam_mask_root, 'cam_10_projected.jpg')),
        11: cv2.imread(os.path.join(cam_mask_root, 'cam_11_projected.jpg')),
        12: cv2.imread(os.path.join(cam_mask_root, 'cam_12_projected.jpg'))
    }

    start_idx = 0
    visualize_samples(samples, calibs, cam_masks, start_sample_idx=start_idx)