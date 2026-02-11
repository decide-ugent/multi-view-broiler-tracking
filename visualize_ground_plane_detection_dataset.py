import os
import cv2
import numpy as np
import argparse
import colorhash
from pathlib import Path
from ground_plane_detection.data_loading import load_samples
from utils.projection import load_calibration

# Global variables
show_bounding_boxes = True
confidence_threshold = 0.0
window_name = "Multi-View Detection Visualization"
current_sample = None  # Store reference to current sample
_current_calibs = None
_current_cam_masks = None

def update_confidence_threshold(val):
    global confidence_threshold, current_sample, _current_calibs, _current_cam_masks
    confidence_threshold = val / 100.0

    # Redraw the current sample with new threshold if available
    if current_sample is not None and _current_calibs is not None and _current_cam_masks is not None:
        visualize_sample(current_sample, _current_calibs, _current_cam_masks)

def _load_tracking_masks(mask_root, cam_nums):
    """Load reprojection masks from Tracking folder (cam_X_projected.jpg). Same masks as tracking viz."""
    cam_masks = {}
    for cam in cam_nums:
        path = os.path.join(mask_root, f'cam_{cam}_projected.jpg')
        if os.path.isfile(path):
            cam_masks[cam] = cv2.imread(path)
        else:
            cam_masks[cam] = None
    return cam_masks

def visualize_sample(sample, calibs, cam_masks):
    global show_bounding_boxes, confidence_threshold, current_sample
    current_sample = sample  # Store reference to current sample

    frames = {}
    for cam_id in sample['frame_set']:
        img_path = sample['frame_set'][cam_id]
        frames[cam_id] = cv2.imread(img_path)
        if frames[cam_id] is None:
            continue

        cam_num = int(cam_id)

        # Project gt_locs (ground-plane coords in cm) into this camera view
        gt_locs = sample.get('gt_locs', {})
        if gt_locs and cam_num in calibs and cam_masks.get(cam_num) is not None:
            mask = cam_masks[cam_num]
            h_mask, w_mask = mask.shape[:2]
            for did_str, xy in gt_locs.items():
                x, y = int(xy[0]), int(xy[1])
                if not (0 <= x < w_mask and 0 <= y < h_mask):
                    continue
                # Tracking masks: BGR image, [0] is B channel; visible where > 0
                if mask[int(y), int(x)][0] <= 0:
                    continue
                proj_pt = np.array([[x / 100, y / 100, 0]], dtype=np.float32)
                proj_pt, _ = cv2.projectPoints(
                    proj_pt, calibs[cam_num]['rvec'], calibs[cam_num]['tvec'],
                    calibs[cam_num]['cameraMatrix'], calibs[cam_num]['distCoeffs'])
                proj_x, proj_y = int(proj_pt[0][0][0]), int(proj_pt[0][0][1])
                h_img, w_img = frames[cam_id].shape[:2]
                if 0 < proj_x < w_img and 0 < proj_y < h_img:
                    detection_id = int(did_str) if did_str.isdigit() else hash(did_str) % 10000
                    detection_color = colorhash.ColorHash(detection_id).rgb
                    cv2.drawMarker(frames[cam_id], (proj_x, proj_y), (255, 255, 255),
                                  markerType=cv2.MARKER_DIAMOND, markerSize=16, thickness=5)
                    cv2.circle(frames[cam_id], (proj_x, proj_y), 6, detection_color, -1)

        yolo_path = sample['yolo_set'][cam_id]
        if os.path.exists(yolo_path) and show_bounding_boxes:
            with open(yolo_path, 'r') as f:
                detections = f.readlines()

            img_height, img_width = frames[cam_id].shape[:2]
            for det in detections:
                parts = det.strip().split()
                if len(parts) >= 6:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    confidence = float(parts[5])

                    # Only show detections above the confidence threshold
                    if confidence >= confidence_threshold:
                        x1 = int(x_center - width/2)
                        y1 = int(y_center - height/2)
                        x2 = int(x_center + width/2)
                        y2 = int(y_center + height/2)

                        cv2.rectangle(frames[cam_id], (x1, y1), (x2, y2), (0, 255, 0), 1)
                        # label = f"{confidence¨:.2f}"

                        # label = f"{round(confidence*100, 0)}%"
                        # cv2.putText(frames[cam_id], label, (x1, y1-10), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frames[cam_id], f"Camera {cam_id}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    top_row = np.hstack((frames.get(9, np.zeros((500, 500, 3), dtype=np.uint8)),
                         frames.get(10, np.zeros((500, 500, 3), dtype=np.uint8))))
    bottom_row = np.hstack((frames.get(11, np.zeros((500, 500, 3), dtype=np.uint8)),
                            frames.get(12, np.zeros((500, 500, 3), dtype=np.uint8))))

    combined_img = np.vstack((top_row, bottom_row))

    # Status text
    status_height = 100
    status_img = np.zeros((status_height, combined_img.shape[1], 3), dtype=np.uint8)
    bb_status = "ON" if show_bounding_boxes else "OFF"
    
    # Sample info
    sample_info = f"Sample {sample['sample_number_int']} / {len(samples)-1} - {sample['date']} - {sample['phase']} - age: {sample['age']}"
    cv2.putText(status_img, sample_info, 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Controls info
    gt_count = len(sample.get('gt_locs', {}))
    controls_info = f"Controls: 'b' - Toggle YOLO boxes ({bb_status}) | 'q' - Quit | Any key - Next | GT locations: {gt_count} projected"
    cv2.putText(status_img, controls_info, 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add confidence threshold info
    threshold_info = f"Confidence Threshold: {confidence_threshold:.2f}"
    cv2.putText(status_img, threshold_info, 
                (combined_img.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Combine with status bar
    display_img = np.vstack((status_img, combined_img))
    cv2.imshow(window_name, display_img)
    
    # Handle key presses
    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):  # Press 'q' to exit
            cv2.destroyAllWindows()
            return None
        elif key == ord('b'):  # Toggle bounding boxes
            show_bounding_boxes = not show_bounding_boxes
            return visualize_sample(sample, calibs, cam_masks)  # Re-render with new settings
        else:
            break  # Any other key continues to next sample
    
    return combined_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Multi-View Detection Dataset with YOLO11x detections.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="MVBroTrack",
        help="Path to the MVBroTrack root folder (relative to the current working directory or absolute).",
    )
    parser.add_argument(
        "--yolo_labels_folder",
        type=str,
        default="1920_11x",
        help="Folder name for YOLO labels inside the dataset.",
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

    yolo_labels_folder = args.yolo_labels_folder
    samples = load_samples(os.path.join(dataset_root, 'Multi-View-Detection'), yolo_labels_folder)
    samples.sort(key=lambda x: x['sample_number_int'])
    print(f"Total samples: {len(samples)}")

    cam_nums = [9, 10, 11, 12]
    calibs = {cam: load_calibration(cam, calibration_root=os.path.join(dataset_root, 'Calibrations')) for cam in cam_nums}
    cam_masks = _load_tracking_masks(os.path.join(dataset_root, 'Reprojection_Masks'), cam_nums)
    _current_calibs = calibs
    _current_cam_masks = cam_masks

    cv2.namedWindow(window_name)
    cv2.createTrackbar("Confidence Threshold", window_name, 0, 100, update_confidence_threshold)

    i = 0
    while i < len(samples):
        sample = samples[i]
        print(f"Visualizing sample {i+1}/{len(samples)}: {sample['sample_number_int']} / {len(samples)-1} - {sample['date']}")
        result = visualize_sample(sample, calibs, cam_masks)
        
        if result is None:  # User pressed 'q'
            break
        
        i += 1
        
        # if i == 0:  # Only run for the first sample (for testing)
        #     break

    cv2.destroyAllWindows()