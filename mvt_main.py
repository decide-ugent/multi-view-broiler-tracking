"""Multi-view ground-plane tracking entry point and evaluation helpers.

Paths are defined in utils.paths (under DATA/ by default). Override with:
  MVBROTRACK_DATA_ROOT       (default: DATA)
  MVBROTRACK_DATASET_ROOT    (default: DATA/MVBroTrack)
  MVBROTRACK_OUTPUT_DIR      (default: DATA/INTERMEDIATE_RESULTS)
"""
import os
import uuid

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ground_plane_tracking.data_loading import load_samples as _load_samples_for_export
from ground_plane_tracking.trackers import run_tracker_on_sample as _run_tracker
from ground_plane_tracking.tracking_data_loading import load_samples
from utils.logging import log_summary_to_wandb, save_summary_to_file
from utils.metrics import calculate_ground_plane_metrics
from utils.paths import DATASET_ROOT, DEFAULT_RESULTS_CSV, FINAL_SUMMARIES_DIR, OUTPUT_DIR, timing_results_file
from utils.projection import load_calibration

_CONFIGS_ROOT = 'ground_plane_tracking/yml_configs'
_TUNED_ROOT = 'ground_plane_tracking/yml_tuned'

_METHOD_CONFIGS = {
    'muSSP_curve': 'config_muSSP_2D_curve.yaml',
    'muSSP3D':     'config_muSSP_3D.yaml',
    'sort_curve':  'config_sort_2D_curve.yaml',
    'sort_3D':     'config_sort_3D.yaml',
}


def default_config(method):
    if method not in _METHOD_CONFIGS:
        raise ValueError(f'No default config for method {method}')
    return os.path.join(_CONFIGS_ROOT, _METHOD_CONFIGS[method])


def tuned_params_path(method, fold_name):
    fname = f'muSSP_3D_{fold_name}.yaml' if method == 'muSSP3D' else f'{method}_{fold_name}.yaml'
    return os.path.join(_TUNED_ROOT, fname)


def produce_folds(samples):
    """5 stratified 50/50 val/test folds — matches paper protocol."""
    all_indices = np.arange(len(samples))
    folds = {}
    for i in range(5):
        name = f"fold_{i + 1}"
        train_idx, test_idx = train_test_split(
            all_indices,
            test_size=0.5,
            stratify=[samples[j]['phase'] for j in all_indices],
            random_state=42 + i,
        )
        folds[name] = {
            'val_samples':  [samples[j] for j in train_idx],
            'test_samples': [samples[j] for j in test_idx],
        }
        print(f"  {name}: {len(folds[name]['val_samples'])} val / {len(folds[name]['test_samples'])} test")
    return folds


def eval_samples(samples, params, guid, calibs, method, debug, skip_2d_guid, output_root, **tracking_kwargs):
    """Run tracker over a list of samples and return aggregated MOT metrics."""
    gts, preds, distances, names = [], [], [], []
    for sample in tqdm(samples):
        pred_df = _run_tracker(
            sample, params, guid, calibs, method,
            root_output_dir=output_root,
            debug=debug,
            skip_2d_guid=skip_2d_guid,
            **tracking_kwargs,
        )
        gts.append(sample['ground_truth_df'])
        preds.append(pred_df)
        distances.append(sample['radius'])
        names.append(sample['phase'])
    return calculate_ground_plane_metrics(gts, preds, names, distances)


def run_multi_fold_eval(
    samples,
    calibs,
    method,
    output_root,
    config_file,
    fold,
    *,
    tuning=False,
    use_wandb=False,
    debug=False,
    skip_2d_guid=None,
    summary_file='results.csv',
    wandb_project='Tuning_ground_plane_tracking',
    eval_set='val',
):
    """Run all folds or one fold.

    fold='all'    → all 5 folds, per-fold tuned YAML; split chosen by eval_set
                    (default val = paper half on modern sklearn; test = sklearn test half).
    fold='fold_X' → one fold, config_file, split chosen by eval_set.
    """
    import wandb

    if use_wandb:
        wandb.init(project=wandb_project)

    folds = produce_folds(samples)

    split_key = f'{eval_set}_samples'

    if fold == 'all':
        out_dir = FINAL_SUMMARIES_DIR
        os.makedirs(out_dir, exist_ok=True)
        for fold_name, fold_data in folds.items():
            guid = str(uuid.uuid4())
            with open(tuned_params_path(method, fold_name), 'r') as f:
                params = yaml.safe_load(f)
            print(f'fold {fold_name} | eval_set={eval_set} | GUID {guid}')
            # val = first half of stratified split (paper-repro on sklearn >= 1.0);
            # test = second half (sklearn's literal test_indices).
            summary = eval_samples(
                fold_data[split_key], params, guid, calibs, method, debug, skip_2d_guid, output_root,
            )
            save_summary_to_file(method, summary, guid, params,
                                 os.path.join(out_dir, f'{method}_{fold_name}_{eval_set}.csv'))
            if use_wandb:
                log_summary_to_wandb(method, summary, guid, params)
    else:
        with open(config_file, 'r') as f:
            params = yaml.safe_load(f)
        if tuning:
            params.update(wandb.config)
        guid = str(uuid.uuid4())
        print(f'Run GUID: {guid}')
        summary = eval_samples(
            folds[fold][split_key], params, guid, calibs, method, debug, skip_2d_guid, output_root,
        )
        save_summary_to_file(method, summary, guid, params, summary_file)
        if use_wandb:
            log_summary_to_wandb(method, summary, guid, params)

    if use_wandb:
        wandb.finish()


def export_single_sample(config_file, method, sample_number, *, debug=False, skip_2d_guid=None, use_wandb=False):
    """Export tracker predictions for one sample (pseudo-labelling).

    Uses MVBROTRACK_DATASET_ROOT and MVBROTRACK_OUTPUT_DIR env vars.
    """
    import wandb

    samples = _load_samples_for_export(os.path.join(DATASET_ROOT, 'Multi-View-Tracking'), 'yolo11x_dets', 3)
    cam_nums = [9, 10, 11, 12]
    calibs = {cam: load_calibration(cam, calibration_root=os.path.join(DATASET_ROOT, 'Calibrations')) for cam in cam_nums}

    prefix = f'{int(sample_number)}_'
    matches = [s for s in samples if str(s['sample_number']).startswith(prefix)] \
        or [s for s in samples if str(s['sample_number']) == str(sample_number)]
    if not matches:
        raise FileNotFoundError(f'No sample found for sample_number={sample_number}')
    sample = min(matches, key=lambda s: str(s['sample_number']))

    with open(config_file, 'r') as f:
        params = yaml.safe_load(f)
    guid = str(uuid.uuid4())

    if use_wandb:
        wandb.init(project='Tuning_ground_plane_tracking')

    pred_df = _run_tracker(
        sample, params, guid, calibs, method,
        root_output_dir=OUTPUT_DIR, debug=debug, skip_2d_guid=skip_2d_guid,
    )
    out_dir = os.path.join(OUTPUT_DIR, 'tracking_results_for_pseudo_labelling')
    os.makedirs(out_dir, exist_ok=True)
    pred_df.to_csv(os.path.join(out_dir, f'{guid}_{sample_number}.csv'))
    print(f'Saved results for GUID: {guid}')

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run ground plane tracking')
    parser.add_argument('--method', required=True,
                        choices=list(_METHOD_CONFIGS) + ['muSSP_3D'])
    parser.add_argument('--config', type=str, help='Config YAML; inferred from method if omitted')
    parser.add_argument('--summary', type=str, default=DEFAULT_RESULTS_CSV)
    parser.add_argument('--tune', action='store_true', help='Pull params from wandb (requires --fold fold_X)')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--skip_2d_guid', type=str, help='GUID for pre-computed 2D curves')
    parser.add_argument('--fold', type=str, default='all',
                        help='"all" runs all 5 folds with per-fold tuned YAML; '
                             '"fold_X" runs one fold with --config')
    parser.add_argument('--eval_set', type=str, default=None, choices=['val', 'test'],
                        help='Which stratified half to evaluate (default: val). '
                             'val = paper-repro half on sklearn>=1; test = sklearn test half.')
    parser.add_argument('--seconds_for_timing', type=int, default=None,
                        help='Clip length for timing benchmark: 5, 10, 15, or 30 s')
    parser.add_argument('--sample_number', type=int, default=None,
                        help='Export predictions for one sample only (pseudo-labelling)')
    args = parser.parse_args()

    if args.method == 'muSSP_3D':
        args.method = 'muSSP3D'
    if args.config is None:
        args.config = default_config(args.method)

    # Single-sample export: uses env-based dataset root, exits immediately
    if args.sample_number is not None:
        export_single_sample(
            args.config, args.method, args.sample_number,
            debug=args.debug, skip_2d_guid=args.skip_2d_guid, use_wandb=args.use_wandb,
        )
        raise SystemExit(0)


    _timing_splits = {30: 1, 15: 2, 10: 3, 5: 6}
    if args.seconds_for_timing is not None:
        if args.seconds_for_timing not in _timing_splits:
            raise ValueError(f'--seconds_for_timing must be one of {list(_timing_splits)}')
        split_size = _timing_splits[args.seconds_for_timing]
    else:
        split_size = 3

    samples = load_samples(os.path.join(DATASET_ROOT, 'Multi-View-Tracking'), 'yolo11x_dets', split_size)
    samples.sort(key=lambda x: x['sample_number'])
    print(f'Loaded {len(samples)} samples')

    cam_nums = [9, 10, 11, 12]
    calibs = {cam: load_calibration(cam, calibration_root=os.path.join(DATASET_ROOT, 'Calibrations'))
              for cam in cam_nums}

    if args.seconds_for_timing is not None:
        with open(args.config, 'r') as f:
            params = yaml.safe_load(f)
        guid = str(uuid.uuid4())
        timing_file = timing_results_file(args.seconds_for_timing, args.method)
        summary = eval_samples(
            samples, params, guid, calibs, args.method, args.debug, args.skip_2d_guid,
            OUTPUT_DIR, seconds_for_timing=args.seconds_for_timing, timing_results_file=timing_file,
        )
        with open(timing_file, 'a') as f:
            f.write(f"{summary['idf1']['OVERALL']},'final MOTA','{args.method}',{summary['mota']['OVERALL']}\n")
        save_summary_to_file(args.method, summary, guid, params, args.summary)
    else:
        # fold='all': test split, tuned YAML per fold (paper-comparable)
        # fold='fold_X': eval_set split with provided config (tuning / exploration)
        run_multi_fold_eval(
            samples, calibs, args.method, OUTPUT_DIR, args.config, args.fold,
            tuning=args.tune,
            use_wandb=args.use_wandb,
            debug=args.debug,
            skip_2d_guid=args.skip_2d_guid,
            summary_file=args.summary,
            wandb_project='Tuning_ground_plane_tracking',
            eval_set=args.eval_set or 'val',
        )
