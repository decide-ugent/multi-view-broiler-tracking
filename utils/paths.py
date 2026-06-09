"""Central paths for dataset downloads and generated outputs under DATA/."""
import os

DATA_ROOT = os.environ.get('MVBROTRACK_DATA_ROOT', 'DATA')


def data_path(*parts):
    return os.path.join(DATA_ROOT, *parts)


DATASET_ROOT = os.environ.get('MVBROTRACK_DATASET_ROOT', data_path('MVBroTrack'))
OUTPUT_DIR = os.environ.get('MVBROTRACK_OUTPUT_DIR', data_path('INTERMEDIATE_RESULTS'))
FINAL_SUMMARIES_DIR = data_path('final_summaries')
GENERATED_DATA_DIR = data_path('GENERATED_DATA')
DEFAULT_RESULTS_CSV = data_path('results.csv')
GROUND_PLANE_SWEEP_RESULTS_DIR = data_path('ground_plane_detection_param_sweeps_results_11x')
MV_DETECTION_VIZ_DIR = data_path('GENERATED_DATA', 'Multi-View-Detection', 'demonstration')


def timing_results_file(seconds, method):
    return data_path(f'timing_results_{seconds}_{method}.txt')


def ensure_data_dirs():
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GENERATED_DATA_DIR, exist_ok=True)
    os.makedirs(FINAL_SUMMARIES_DIR, exist_ok=True)
