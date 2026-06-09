import numpy as np
import colorhash
import csv
import os
import cv2 
from tqdm import tqdm
from sort.sort import *


def convert_points_to_boxes(points, bb_size=20):
    boxes = []
    for point in points:
        x, y = point
        x1 = x - bb_size / 2
        y1 = y - bb_size / 2
        x2 = x + bb_size / 2
        y2 = y + bb_size / 2
        
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes)

def track_ground_plane_sort(all_dets, start, stop, param_dict, debug=False):
    mot_tracker = Sort(max_age=param_dict['MAX_AGE'], min_hits=param_dict['MIN_HITS'], iou_threshold=param_dict['IOU_THRESHOLD']) 

    tracklets = []

    # for frame_index in range(len(all_dets)):
    for frame_index in range(start, stop):
        ground_plane_dets = all_dets[str(frame_index)]
        ground_plane_dets = np.array(ground_plane_dets)
        processed_boxes = convert_points_to_boxes(ground_plane_dets, bb_size=param_dict['BB_SIZE_radius_adjusted'])

        if len(processed_boxes) == 0:
            print('No boxes detected in frame', frame_index)
            processed_boxes = np.empty((0, 5))

        track_bbs_ids = np.empty((0, 5))
        track_bbs_ids = mot_tracker.update(processed_boxes)

        total_tracks_found = len(track_bbs_ids.tolist())
        # print(total_tracks_found, len(processed_boxes), 'unused_boxes', len(processed_boxes) - total_tracks_found)

        for j in range(0, total_tracks_found):
            coords = track_bbs_ids.tolist()[j]
            x1, y1, x2, y2, name_idx = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]), int(coords[4])


            # bb_height = y2 - y1
            # bb_width = x2 - x1
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            tracklets.append([frame_index, name_idx, x_center, y_center, 1, 1, -1, -1, -1])

    return tracklets
