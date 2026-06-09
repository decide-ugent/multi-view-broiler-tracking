import os
import numpy as np
import json
import wandb
from tqdm import tqdm
from ground_plane_detection.mv_detection_model import run_one_sample
from ground_plane_detection.data_loading import create_dataset_splits, load_samples
from ground_plane_detection.mv_utils import get_reprojection_masks, average_results_per_cat, print_results_in_terminal_with_ci, print_results_in_terminal, get_n_best_params_per_fold
from utils.paths import DATASET_ROOT, GROUND_PLANE_SWEEP_RESULTS_DIR, MV_DETECTION_VIZ_DIR
from utils.projection import load_calibration
import argparse

def ground_plane_detection_on_sample_set(samples, calibs, mask_dict, param_dict, save_image_path, visualize=False, use_wandb=False):
    if use_wandb:
        wandb.init(project='Tuning_ground_plane_detection')
        wandb.config.update(param_dict)
        param_dict = wandb.config
    
    results = []
    for sample in tqdm(samples, desc='Processing samples'):
        result_dict = run_one_sample(sample, calibs, mask_dict, param_dict, save_image_path, visualize=visualize)
        results.append(result_dict)

    final_averaged_results = average_results_per_cat(results)

    if use_wandb:
        for cat, values in final_averaged_results.items():
            moda = values['moda']
            modp = values['modp']
            recall = values['recall']
            precision = values['precision']
            wandb.log({f'{cat}/moda': moda, f'{cat}/modp': modp, f'{cat}/recall': recall, f'{cat}/precision': precision})
        wandb.finish()
    
    return final_averaged_results

def parse_args():
    parser = argparse.ArgumentParser(description='Ground plane detection for multi-view dataset')
    
    # Dataset and general options
    parser.add_argument('--root_dataset', type=str, 
                        default=DATASET_ROOT,
                        help='Root directory of the dataset')
    parser.add_argument('--visualize', action='store_true', 
                        help='Enable visualization of results')
    parser.add_argument('--calibrations_folder', type=str, default=None,
                        help='Folder containing camera calibration files (default: {root_dataset}/Calibrations)')
    parser.add_argument('--save_image_path', type=str,
                        default=MV_DETECTION_VIZ_DIR,
                        help='Path to save visualization images')
    
    # Detection parameters
    parser.add_argument('--bb_threshold', type=float, default=0.4,
                        help='Bounding box confidence threshold')
    parser.add_argument('--min_dist', type=int, default=45,
                        help='Minimum distance threshold')
    parser.add_argument('--max_dist', type=int, default=500,
                        help='Maximum distance threshold')
    parser.add_argument('--option', type=str, default='distance',
                        help='Option for detection algorithm')
    parser.add_argument('--remove_threshold', type=int, default=440,
                        help='Threshold for removing detections')
    parser.add_argument('--cam_set', type=int, nargs='+', default=[9, 10, 11, 12],
                        help='Set of camera IDs to use')
    
    # Execution mode
    parser.add_argument('--action', type=str, choices=['test', 'tune'], default='test',
                        help='Action to perform: test across folds or tune on a single fold')
    parser.add_argument('--fold', type=str, default='fold_1',
                        help='Fold to use for tuning (only used if action is "tune")')
    parser.add_argument('--yolo_labels_folder', type=str, default='1920_11x',
                        help='Folder containing YOLO labels')

    
    args = parser.parse_args()
    if args.calibrations_folder is None:
        args.calibrations_folder = os.path.join(args.root_dataset, 'Calibrations')
    return args

if __name__ == '__main__':
    args = parse_args()

    # Build parameter dictionary from args
    param_dict = {
        'bb_threshold': args.bb_threshold,
        'min_dist': args.min_dist,
        'max_dist': args.max_dist,
        'option': args.option,
        'remove_threshold': args.remove_threshold,
        'cam_set': args.cam_set,
    }

    # Load calibrations
    cam_nums = [9, 10, 11, 12]
    calibs = {cam: load_calibration(cam, calibration_root=args.calibrations_folder) for cam in cam_nums}
    if args.visualize: os.makedirs(args.save_image_path, exist_ok=True)
    mask_dict = get_reprojection_masks(os.path.join(args.root_dataset, 'Reprojection_Masks'), cam_nums)


    print('Loading samples...   ', end='')
    samples = load_samples(os.path.join(args.root_dataset, 'Multi-View-Detection'), args.yolo_labels_folder)
    print(len(samples), 'samples loaded')
    folds, final_test_samples = create_dataset_splits(samples, n_folds=5, test_size=0.0, val_size=0.5, fold_test_size=0.5, n_bins=6)
    for fold, data in folds.items():
        print(fold, ': ', len(data['val_samples']), 'val samples, ', len(data['test_samples']), 'test samples')


    if args.action == 'test':
        results_dir = GROUND_PLANE_SWEEP_RESULTS_DIR
        new_optimized_params = os.path.isdir(results_dir)
        if not new_optimized_params:
            print(f'Optional sweep results not found at {results_dir}; using CLI parameters only.')
        avg_params = False

        top_params = {}
        if new_optimized_params:
            top_params = get_n_best_params_per_fold(1, results_dir, with_print=True)
        
        # calculate the average params across folds
        if avg_params:
            avg_params = {}
            for fold_name, params in top_params.items():
                for param_name, param_value in params.items():
                    if param_name not in avg_params:
                        avg_params[param_name] = []
                    avg_params[param_name].append(param_value)
            for param_name in avg_params:
                if param_name != 'bb_threshold':
                    avg_params[param_name] = int(np.mean(avg_params[param_name]))
                else:
                    avg_params[param_name] = round(np.mean(avg_params[param_name]), 2) 
            
            print(json.dumps(avg_params, indent=4))


        per_fold_results = {}
        for fold, data in folds.items():
            results = []

            if new_optimized_params: param_dict.update(top_params[fold])
            if avg_params: param_dict.update(avg_params)
            # param_dict['remove_threshold'] = 1000000   # For testing without remove threshold
            final_averaged_results = ground_plane_detection_on_sample_set(data['test_samples'], calibs, mask_dict, param_dict, args.save_image_path, visualize=args.visualize, use_wandb=False)

            # print_results_in_terminal(final_averaged_results)
            per_fold_results[fold] = final_averaged_results

        print_results_in_terminal_with_ci(per_fold_results, folds)

    elif args.action == 'tune':
        fold_data = folds[args.fold]
        final_averaged_results = ground_plane_detection_on_sample_set(fold_data['val_samples'], calibs, mask_dict, param_dict, args.save_image_path, visualize=args.visualize, use_wandb=True)
        print_results_in_terminal(final_averaged_results)