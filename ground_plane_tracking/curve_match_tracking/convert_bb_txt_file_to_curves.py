import os
import numpy as np
import pandas as pd
# import sys
# sys.path.insert(0, '../utils')
from utils.projection import load_calibration, project_bb_to_ground_plane
from ground_plane_tracking.curve_match_tracking.curve_class import Curve
import pickle
import cv2
from tqdm import tqdm

def convert_bb_txt_file_to_curves(tracklets_per_cam, curve_save_path, calibs, min_dist, max_dist, pen_contour):
    curves = []
    for cam_num, track_data in tracklets_per_cam.items():
        track_data = np.array(track_data)
        unique_ids = np.unique(track_data[:, 1])

        for id_ in tqdm(unique_ids):
            id_data = track_data[track_data[:, 1] == id_]
            curve = []
            boxes = []
            frames = []
            distances = []

            for row in id_data:
                frame, id_, x1, y1, bb_width, bb_height, _, _, _ = row
                projected_points, distance = project_bb_to_ground_plane(x1, y1, x1 + bb_width, y1 + bb_height, calibs[cam_num], min_dist, max_dist)

                if cv2.pointPolygonTest(pen_contour, (projected_points[0][0], projected_points[0][1]), False) > 0:
                    curve.append((projected_points[0][0], projected_points[0][1]))
                    boxes.append((x1, y1, x1 + bb_width, y1 + bb_height))
                    frames.append(frame)
                    distances.append(distance)

            if len(curve) > 0:
                curve_obj = Curve(curve, frames, boxes, cam_num, id_, distances)
                curves.append(curve_obj)

    with open(curve_save_path, 'wb') as f:
        pickle.dump(curves, f)
    return curves

# if __name__ == '__main__':
#     # root_path = '../single-camera-detection-and-tracking/SORT_PREDICTIONS/MAX_AGE_10_MIN_HITS_0_IOU_THRESHOLD_0.7_BB_THRESHOLD_0.45'
#     # root_path = '../single-camera-detection-and-tracking/SORT_PREDICTIONS/MAX_AGE_10_MIN_HITS_0_IOU_THRESHOLD_0.7_BB_THRESHOLD_0.45_POST_PROCESSED'
#     # root_path = '../single-camera-detection-and-tracking/SORT_PREDICTIONS/MAX_AGE_15_MIN_HITS_15_IOU_THRESHOLD_0.7_BB_THRESHOLD_0.45'
#     root_path = '../single-camera-detection-and-tracking/SORT_PREDICTIONS/MAX_AGE_15_MIN_HITS_15_IOU_THRESHOLD_0.7_BB_THRESHOLD_0.45_POST_PROCESSED'
#     all_files = os.listdir(root_path)

#     for sample_number in range(1, 4):
#         curve_save_path = f'outputs/preprocessed_curves/curves_sample_{sample_number}_hits_15_age_15_pp.pkl'
#         convert_bb_txt_file_to_curves(root_path, sample_number, curve_save_path)
