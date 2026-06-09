import os
from tqdm import tqdm
import json
import pickle
import numpy as np
import pandas as pd
import timeit
from ground_plane_tracking.muSSP_tracking.muSSP_tracking_double_pass import muSSP_tracking
from ground_plane_tracking.curve_match_tracking.convert_bb_txt_file_to_curves import convert_bb_txt_file_to_curves
from ground_plane_tracking.curve_match_tracking.run_curve_fusion import fuse_curves
from single_camera_tracking.tracker_2D import run_2D_tracker

from ground_plane_detection.mv_detection_model import model
from ground_plane_tracking.muSSP_tracking.sort_ground_plane import track_ground_plane_sort

def run_tracker_on_sample(sample, param_dict, guid, calibs, method, root_output_dir, debug=False, skip_2d_guid=None, seconds_for_timing=None, timing_results_file='timing_results.txt'):
    os.makedirs(root_output_dir, exist_ok=True)
    sample_number = sample['sample_number']

    # Prepare tracker-specific inputs for this sample.
    if method == 'muSSP3D' or method == 'sort_3D': # Here we need to make sure that the ground plane detections per frame are prepared
        mussp_input = os.path.join(root_output_dir, f'muSSP_input/fused_per_frame_detections_sample_{sample_number}.json')
        if seconds_for_timing is not None:
            mussp_input = os.path.join(root_output_dir, f'muSSP_input/fused_per_frame_detections_sample_{sample_number}_seconds_{seconds_for_timing}.json')
        os.makedirs(os.path.dirname(mussp_input), exist_ok=True)
        if not os.path.exists(mussp_input):
            all_cam_nums = [9, 10, 11, 12]
            per_frame_fused_detections = {}

            start_time = timeit.default_timer()
            for frame_idx in tqdm(range(1, sample['max_valid_frame_for_sample']+1)):

                yolo_set = {}
                for cam_num in all_cam_nums:
                    yolo_set[cam_num] = sample['detections'][str(cam_num)][frame_idx]
                
                frame_sample = {
                    'yolo_set': yolo_set,
                }

                pred_locs, _, _ = model(frame_sample,
                                        cam_set=all_cam_nums, # for evaluating performance of specific camera sets
                                        calibs=calibs, 
                                        radius=sample['radius']*2,  
                                        bb_threshold=param_dict['bb_threshold'],
                                        option='distance', 
                                        remove_threshold=param_dict['remove_threshold'], 
                                        vis=False, 
                                        min_dist=param_dict['min_dist'], 
                                        max_dist=param_dict['max_dist'],
                                        save_image_path=None,
                                        need_normalization=False
                                        )
                per_frame_fused_detections[frame_idx] = pred_locs

            end_time = timeit.default_timer()
            time_taken = end_time - start_time
            with open(timing_results_file, 'a') as f:
                f.write(f"{sample_number},{seconds_for_timing},'mv_fusion_{method}',{time_taken}\n")

            with open(mussp_input, 'w') as f:
                json.dump(per_frame_fused_detections, f)

        with open(mussp_input, 'r') as f:
            data = json.load(f)

    elif method == 'sort_curve' or method == 'muSSP_curve':
        if skip_2d_guid is None:
            savefolder = os.path.join(root_output_dir, f'{method}_2D_output/{guid}_sample_{sample_number}') # currently not used
            # os.makedirs(savefolder, exist_ok=True)
            start_time = timeit.default_timer()
            tracklets_per_cam = run_2D_tracker(sample, param_dict, savefolder, method=method, debug=debug)
            end_time = timeit.default_timer()
            time_taken = end_time - start_time
            with open(timing_results_file, 'a') as f:
                f.write(f"{sample_number},{seconds_for_timing},'{method}_2D',{time_taken}\n")

            curve_save_path = os.path.join(root_output_dir, f'tracklet_curves/{method}/{guid}_sample_{sample_number}.pkl')
            os.makedirs(os.path.dirname(curve_save_path), exist_ok=True)

            pen_contour = np.array([[1120,  50], [1120 , 465], [ 240,  465], [ 240  , 50]])  # updated contour

            start_time = timeit.default_timer()
            curves = convert_bb_txt_file_to_curves(tracklets_per_cam,  curve_save_path, calibs, param_dict['min_dist']*10, param_dict['max_dist']*10, pen_contour)
            end_time = timeit.default_timer()
            time_taken = end_time - start_time
            with open(timing_results_file, 'a') as f:
                f.write(f"{sample_number},{seconds_for_timing},'{method}_2D_to_curves',{time_taken}\n")

        else:
            curve_save_path = os.path.join(root_output_dir, f'tracklet_curves/{method}/{skip_2d_guid}_sample_{sample_number}.pkl')
            with open(curve_save_path, 'rb') as f:
                curves = pickle.load(f)
    else:
        raise ValueError(f'Method {method} not supported')
    

    radius = sample['radius']
    if method == 'muSSP3D':
        start = 1
        stop = len(data)
        if debug:
            stop = 6
        param_dict['max_distance'] = param_dict['dist_multiplier'] * (radius*2)

        if debug:
            print(f"Running muSSP tracking with parameters: {param_dict}, start: {start}, stop: {stop}, sample_number: {sample_number}")
        start_time = timeit.default_timer()

        output_tracks = muSSP_tracking(data, start, stop, 
                                sigma_distance=param_dict['sigma_distance'], 
                                max_distance=param_dict['max_distance'], 
                                sigma_motion=param_dict['sigma_motion'],
                                max_jump_pass1=param_dict['max_jump_pass1'], 
                                max_jump_pass2=param_dict['max_jump_pass2'],
                                weight_source_sink=param_dict['weight_source_sink'],
                                length=param_dict['length'],
                                th_length=param_dict['th_length'],
                                verbose=debug
                                )
        end_time = timeit.default_timer()
        time_taken = end_time - start_time
        with open(timing_results_file, 'a') as f:
            f.write(f"{sample_number},{seconds_for_timing},'muSSP',{time_taken}\n")
    elif method == 'sort_3D':

        start = 1
        stop = len(data)
        if debug:
            stop = 6

        start_time = timeit.default_timer()
        param_dict['BB_SIZE_radius_adjusted'] = param_dict['BB_SIZE'] * radius
        output_tracks = track_ground_plane_sort(data, start, stop, param_dict, debug=debug)
        end_time = timeit.default_timer()
        time_taken = end_time - start_time
        with open(timing_results_file, 'a') as f:
            f.write(f"{sample_number},{seconds_for_timing},'sort_3D',{time_taken}\n")
    elif method == 'sort_curve' or method == 'muSSP_curve':
        start_time = timeit.default_timer()
        param_dict['R'] = param_dict['R_multiplier'] * (radius*2)
        param_dict['shift_factor'] = ((radius/2)/100) * param_dict['shift_factor_multiplier']
        if debug:
            print(f"Running {method} tracking with parameters: {param_dict}")
        output_tracks = fuse_curves(curves, calibs, scale_factor=1, 
                                length_threshold=param_dict['length_threshold'], 
                                shift_factor=param_dict['shift_factor'], 
                                R=param_dict['R'], 
                                R_eucl=param_dict['R_eucl'],
                                avg_max_distance=param_dict['avg_max_distance'],
                                max_gap=param_dict['max_gap'])
        end_time = timeit.default_timer()
        time_taken = end_time - start_time
        with open(timing_results_file, 'a') as f:
            f.write(f"{sample_number},{seconds_for_timing},'{method}_curve_fusion',{time_taken}\n")
    else:
        raise ValueError(f'Method {method} not supported')
    

    filename = f'{guid}_{sample_number}.txt'
    savepath = os.path.join(root_output_dir, f'output_tracks_{method}/{filename}')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    np.savetxt(savepath, output_tracks, delimiter=',')
    return pd.DataFrame(output_tracks, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'])
