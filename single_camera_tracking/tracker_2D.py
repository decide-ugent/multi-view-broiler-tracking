import numpy as np
import os 
from single_camera_tracking.muSSP_tracking_functions import muSSP_tracking
from single_camera_tracking.sort_tracking_functions import track_video_sort

def run_2D_tracker(sample, param_dict, savefolder, method='sort_curve', debug=False):
    tracklets_per_cam = {}
    for cam_num in sample['detections']:
        all_dets = sample['detections'][cam_num]

        if method == 'sort_curve':
            print(f'Running SORT on camera {cam_num}')
            tracklets = track_video_sort(all_dets, param_dict, debug=debug)
        elif method == 'muSSP_curve':
            if debug:
                stop = 15
            else:
                stop = len(all_dets)
            
            tracklets = muSSP_tracking(all_dets, start=1, stop=stop+1, 
                                    sigma_distance=param_dict['sigma_distance'], 
                                    max_distance=param_dict['max_distance'], 
                                    sigma_motion=param_dict['sigma_motion'],
                                    max_jump_pass1=param_dict['max_jump_pass1'], 
                                    max_jump_pass2=param_dict['max_jump_pass2'],
                                    weight_source_sink=param_dict['weight_source_sink'],
                                    length=param_dict['length'],
                                    th_length=param_dict['th_length'],
                                    sigma_box_size=param_dict['sigma_box_size'],
                                    BB_THRESHOLD=param_dict['BB_THRESHOLD'],
                                    verbose=True)
            
        tracklets_per_cam[int(cam_num)] = tracklets

    return tracklets_per_cam
