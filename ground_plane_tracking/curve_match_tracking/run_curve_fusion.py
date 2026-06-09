import pickle
import json
import numpy as np
import pandas as pd
import cv2
import ground_plane_tracking.curve_match_tracking.geom_fusion_curves as geom_fusion_curves
from ground_plane_tracking.curve_match_tracking.curve_class import Curve
# import sys
# sys.path.insert(0, '../utils')
# from utils.metrics import calculate_metrics
from utils.projection import load_calibration


def calculate_curve_start_end_distance(curve):
    start_point = np.array(curve.get_curve()[0])
    end_point = np.array(curve.get_curve()[-1])
    return np.linalg.norm(end_point - start_point)

def calculate_curve_length(curve):
    total_length = 0
    points = curve.get_curve()
    for i in range(1, len(points)):
        total_length += np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
    return total_length

def shift_curve_towards_camera(curve, cam_location, shift_factor):
    shifted_points = []
    for point, distance in zip(curve.get_curve(), curve.list_of_distances):
        vector = np.array(point) - (cam_location[:2]*100)
        shifted_point = cam_location[:2]*100 + vector * (1 - shift_factor * distance / np.max(curve.list_of_distances))
        shifted_points.append(shifted_point)
    return np.array(shifted_points, dtype=np.int32)


def fuse_curves(initial_curves, calibs, scale_factor=1, length_threshold=200, shift_factor=0.09, R=30, R_eucl=30, avg_max_distance=400, max_gap=25):
    curves = []
    # cutoff_frame = 100
    for curve in initial_curves:
        # Filter points after cutoff frame

        # cutoff_indices = [i for i, frame in enumerate(curve.list_of_frames) if frame <= cutoff_frame]
        # if not cutoff_indices:  # Skip if no frames before cutoff
        #     continue
            
        # curve.list_of_points = [curve.list_of_points[i] for i in cutoff_indices]
        # curve.list_of_frames = [curve.list_of_frames[i] for i in cutoff_indices]
        # curve.list_of_boxes = [curve.list_of_boxes[i] for i in cutoff_indices]
        # curve.list_of_distances = [curve.list_of_distances[i] for i in cutoff_indices]

        # if len(curve.list_of_points) < 5:
        #     continue
        if curve.get_average_curve_distance() > avg_max_distance:
            continue

        # Filter out points where distance exceeds 450
        # valid_indices = [i for i, dist in enumerate(curve.list_of_distances) if dist <= 350]
        # if not valid_indices:  # Skip if no valid points remain
        #     continue
            
        # curve.list_of_points = [curve.list_of_points[i] for i in valid_indices]
        # curve.list_of_frames = [curve.list_of_frames[i] for i in valid_indices]
        # curve.list_of_boxes = [curve.list_of_boxes[i] for i in valid_indices]
        # curve.list_of_distances = [curve.list_of_distances[i] for i in valid_indices]

        # curve_length = calculate_curve_length(curve)
        curve_start_end_distance = calculate_curve_start_end_distance(curve)

        if curve_start_end_distance >= length_threshold:
            cam_location = calibs[curve.cam_id]['cam_location']
            shifted_curve = shift_curve_towards_camera(curve, cam_location, shift_factor)
            curves.append(Curve(shifted_curve, curve.list_of_frames, curve.list_of_boxes, curve.cam_id, curve.tracklet_id, curve.list_of_distances))
        else:
            curves.append(curve)
    
    # with open('outputs/preprocessed_curves/post_processed_curves_test.pkl', 'wb') as f:
    #     pickle.dump(curves, f)
    
    for curve in curves:
        curve.multiply_points(scale_factor)

    graph, temp_data = geom_fusion_curves.connect_detections(curves, R*scale_factor, R_eucl*scale_factor, max_gap)
    # temp_data = np.array(temp_data)
    # np.savetxt('temp_data_testing_for_vis_2.txt', temp_data, delimiter=',')
    fusion_results, unconnected_detections = geom_fusion_curves.fusion_multi_camera_detections(curves, graph)
    preds_data = geom_fusion_curves.generate_preds_df(fusion_results, unconnected_detections)
    return preds_data

if __name__ == '__main__':
    cam_nums = [9, 10, 11, 12]
    calibs = {cam: load_calibration(cam, calibration_root='../DATASET/Calibrations') for cam in cam_nums}

    # hits = [0, 15]
    # ages = [10, 15]
    hit = 15
    age = 15

    for sample_number in range(1, 4):
        with open(f'outputs/preprocessed_curves/curves_sample_{sample_number}_hits_{hit}_age_{age}_pp.pkl', 'rb') as f:
            initial_curves = pickle.load(f)

        preds_df = fuse_curves(initial_curves, calibs, scale_factor=4, length_threshold=80, shift_factor=0.09, R=30, R_eucl=30)
        preds_df.to_csv(f'outputs/fused_curves/preds_df_sample_{sample_number}_hits_{hit}_age_{age}_pp.csv', index=False)
    