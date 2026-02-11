import os
import numpy as np
import pandas as pd
from datetime import datetime

def get_max_distance_eval(sample_number):
    sample_dates = {
        1: datetime(2023, 4, 22),
        2: datetime(2023, 4, 12),
        3: datetime(2023, 3, 29),
        4: datetime(2023, 4, 19),
        5: datetime(2023, 4, 8),
        6: datetime(2023, 3, 25),
    }
    first_day = datetime(2023, 3, 20)
    age = (sample_dates[sample_number] - first_day).days
    radius = ((0.3)*age) + 5
    max_distance = (radius / 17.5) * 16.5
    return max_distance, age

def sample_number_to_name(sample_number):
    name_dict = {
        1: 'finisher',
        2: 'grower',
        3: 'starter',
        4: 'finisher',
        5: 'grower',
        6: 'starter'
    }
    return name_dict[sample_number]

def load_samples(root_dataset_dir, selected_yolo_dets='yolo11x_dets', per_sample_split_amount=2):
    # per sample split amount -> every sample is 30 seconds long, spliting by 6 will result in 5 seconds per split

    samples = []
    # initial laoding of the samples of ~30 seconds
    for folder in os.listdir(root_dataset_dir):
        if folder.startswith('sample_'):
            max_valid_frame_for_sample = 1000000

            new_sample = {}
            new_sample['sample_number'] = int(folder.split('_')[-1])
            new_sample['phase'] = sample_number_to_name(new_sample['sample_number'])
            radius, age = get_max_distance_eval(new_sample['sample_number'])
            new_sample['radius'] = radius
            new_sample['age'] = age

            video_paths = {}

            for video_file in os.listdir(os.path.join(root_dataset_dir, folder, 'videos')):
                if video_file.endswith('.mp4'):
                    cam_num = video_file.split('_')[1]
                    video_paths[cam_num] = os.path.join(root_dataset_dir, folder, 'videos', video_file)
            
            new_sample['video_paths'] = video_paths

            yolo_detections_dir = os.path.join(root_dataset_dir, folder, selected_yolo_dets)
            yolo_detections_files = os.listdir(yolo_detections_dir)


            detections = {}

            for bb_folder in yolo_detections_files:
                cam_num = bb_folder.split('_')[1]

                all_cam_detections = os.listdir(os.path.join(yolo_detections_dir, bb_folder))
                all_cam_detections.sort()

                per_frame_cam_detections = {}
                for det_file in all_cam_detections:
                    # print(det_file)  000001.txt
                    frame_id = int(det_file.split('.')[0])
                    per_frame_cam_detections[frame_id] = os.path.join(yolo_detections_dir, bb_folder, det_file)

                detections[cam_num] = per_frame_cam_detections

                max_valid_frame_for_sample = min(max_valid_frame_for_sample, len(detections[cam_num]))
            
            new_sample['detections'] = detections

            ground_truth_file = 'gt_groundplane_sample_{}.txt'

            ground_truth_path = os.path.join(root_dataset_dir, folder, ground_truth_file.format(new_sample['sample_number']))
            gt_data = np.loadtxt(ground_truth_path, delimiter=',')
            new_sample['ground_truth_df'] = pd.DataFrame(gt_data, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'])

            # Load image-plane ground truth tracks for samples 1, 2, 3 (per-camera bbox tracks)
            image_plane_tracks = None
            if new_sample['sample_number'] in (1, 2, 3):
                image_tracks_dir = os.path.join(root_dataset_dir, folder, 'image_tracks')
                if os.path.isdir(image_tracks_dir):
                    image_plane_tracks = {}
                    for cam_id in video_paths.keys():
                        track_file = f"gt_df_sample_{new_sample['sample_number']}_cam_{cam_id}.txt"
                        track_path = os.path.join(image_tracks_dir, track_file)
                        if os.path.isfile(track_path):
                            track_data = np.loadtxt(track_path, delimiter=',')
                            image_plane_tracks[cam_id] = pd.DataFrame(
                                track_data,
                                columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility']
                            )
            new_sample['image_plane_tracks'] = image_plane_tracks
            # print("df length: ", new_sample['ground_truth_df']['FrameId'].nunique())
            max_valid_frame_for_sample = min(max_valid_frame_for_sample, new_sample['ground_truth_df']['FrameId'].nunique())
            # print("max_valid_frame_for_sample: ", max_valid_frame_for_sample)

            new_sample['max_valid_frame_for_sample'] = max_valid_frame_for_sample

            samples.append(new_sample)

    # now we split the samples into chuncks 

    chucked_samples = []
    for sample in samples:
        max_valid_frame_for_sample = sample['max_valid_frame_for_sample']
        chuck_size_in_frames = max_valid_frame_for_sample // per_sample_split_amount

        num_samples_already_chucked = 0

        for i in range(0, max_valid_frame_for_sample, chuck_size_in_frames):
            new_chucked_sample = {}
            new_chucked_sample['detections'] = {}
            for cam_num in sample['detections']:
                # Create new dictionary with renumbered frame IDs
                renumbered_detections = {}
                original_frames = sorted(sample['detections'][cam_num].keys())
                chunk_frames = original_frames[i:i+chuck_size_in_frames]
                
                for new_frame_id, original_frame_id in enumerate(chunk_frames, 1):
                    renumbered_detections[new_frame_id] = sample['detections'][cam_num][original_frame_id]
                
                new_chucked_sample['detections'][cam_num] = renumbered_detections
            
            # Filter ground truth for this chunk and renumber frames
            chunk_gt_df = sample['ground_truth_df'][(sample['ground_truth_df']['FrameId'] >= i+1) & 
                                                    (sample['ground_truth_df']['FrameId'] <= i+chuck_size_in_frames)]
            
            # Create a copy to avoid modifying the original
            chunk_gt_df = chunk_gt_df.copy()
            
            # Renumber frames to start at 1
            chunk_gt_df['FrameId'] = chunk_gt_df['FrameId'] - i
            
            new_chucked_sample['ground_truth_df'] = chunk_gt_df
            new_chucked_sample['max_valid_frame_for_sample'] = chuck_size_in_frames
            new_chucked_sample['video_paths'] = sample['video_paths']
            new_chucked_sample['original_start_frame'] = i
            new_chucked_sample['original_stop_frame'] = i+chuck_size_in_frames

            new_chucked_sample['radius'] = sample['radius']
            new_chucked_sample['age'] = sample['age']
            new_chucked_sample['phase'] = sample['phase']

            # Filter and renumber image_plane_tracks for this chunk (if present)
            if sample.get('image_plane_tracks') is not None:
                new_chucked_sample['image_plane_tracks'] = {}
                i_start = i
                i_stop = i + chuck_size_in_frames
                for cam_id, track_df in sample['image_plane_tracks'].items():
                    chunk_track = track_df[(track_df['FrameId'] >= i_start) & (track_df['FrameId'] < i_stop)].copy()
                    chunk_track['FrameId'] = chunk_track['FrameId'] - i_start + 1
                    new_chucked_sample['image_plane_tracks'][cam_id] = chunk_track
            else:
                new_chucked_sample['image_plane_tracks'] = None

            chucked_samples.append(new_chucked_sample)
            num_samples_already_chucked += 1
            new_chucked_sample['sample_number'] = str(sample['sample_number']) + '_' + str(num_samples_already_chucked)
            if num_samples_already_chucked == per_sample_split_amount:
                break


    return chucked_samples  # Changed from samples to chucked_samples