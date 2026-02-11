import numpy as np
import os
import json
from datetime import datetime
from utils.data_loading import create_stratified_splits

def create_dataset_splits(samples, n_folds=5, test_size=0.1, val_size=0.6, fold_test_size=0.4, n_bins=10):
    """
    Create initial train/test split and then n_folds of train/val/test splits.
    
    First splits all samples into:
    - final_train (becomes all_samples for cross-validation)
    - final_test (held out completely)
    
    Then performs k-fold cross-validation on final_train to create:
    - K1_Train, K1_Val, K1_Test
    - K2_Train, K2_Val, K2_Test
    - etc.
    """
    np.random.seed(42)  # For reproducibility
    
    # Create stratified sampling function
    stratified_sampler = create_stratified_splits(samples, n_bins=n_bins)
    
    # First create final train/test split
    all_indices = set(range(len(samples)))
    if test_size > 0:
        final_test_indices = set(stratified_sampler(samples, test_size))
        final_train_indices = all_indices - final_test_indices
    else:
        final_test_indices = set()
        final_train_indices = all_indices
    
    # Store the final test set
    final_test_samples = [samples[i] for i in final_test_indices]
    
    # Create k-folds from final_train data
    folds = {}
    final_train_samples = [samples[i] for i in final_train_indices]
    
    for fold in range(n_folds):
        # Create a new stratified sampler for the final_train set
        fold_sampler = create_stratified_splits(final_train_samples, n_bins=n_bins)
        
        # Calculate test size from the total samples
        test_proportion = fold_test_size / (fold_test_size + val_size)

        # Split final_train into fold_test and fold_val
        fold_test_indices_local = set(fold_sampler(final_train_samples, test_proportion))
        fold_val_indices_local = set(range(len(final_train_samples))) - fold_test_indices_local
        
        # Get the actual samples for test and validation
        fold_test_samples = [final_train_samples[i] for i in fold_test_indices_local]
        fold_val_samples = [final_train_samples[i] for i in fold_val_indices_local]
        
        # Store only validation and test samples in each fold
        folds[f'fold_{fold+1}'] = {
            'val_samples': fold_val_samples,
            'test_samples': fold_test_samples
        }
    
    # Also return the final test set that's held out from all folds
    return folds, final_test_samples


def get_category_from_day(day): # only for multi-view-detection dataset
#      - Starter from 20/03/2023 to 29/03/2023
#      - Grower: from 30/03/2023 to 12/04/2023
#      - Finisher: from 13/04/2023 to 02/05/2023
    starterdate = datetime(2023, 3, 20).date()
    growerdate = datetime(2023, 3, 30).date()
    finisherdate = datetime(2023, 4, 13).date()
    phase = None
    if day < growerdate:
        phase = 'starter'
    elif day < finisherdate and day >= growerdate:
        phase = 'grower'
    else:
        phase = 'finisher'
    
    age = (day - starterdate).days
    return age, phase
    
def load_samples(root_dataset, yolo_labels_folder):
    labels_folder = os.path.join(root_dataset, 'labels')
    samples = []
    cam_numbers = [9, 10, 11, 12]
    for label_file in os.listdir(labels_folder):
        sample_number = label_file.split('_')[0]
        date_string = label_file.split('_')[1].split('.')[0]
        new_sample = {}
        new_sample['sample_number'] = sample_number
        new_sample['sample_number_int'] = int(sample_number)
        new_sample['date'] = date_string

        with open(os.path.join(labels_folder, label_file), 'r') as f:
            gt_locs = json.load(f)
        
        frame_set = {}
        yolo_set = {}
        for cam_number in cam_numbers:
            frame_set[cam_number] = os.path.join(root_dataset, 'images', f'{sample_number}_{date_string}_{cam_number:02d}.jpg')
            yolo_set[cam_number] = os.path.join(root_dataset, yolo_labels_folder, f'{sample_number}_{date_string}_{cam_number:02d}.txt')
        
        new_sample['frame_set'] = frame_set 
        new_sample['yolo_set'] = yolo_set
        new_sample['gt_locs'] = gt_locs
        new_sample['age'], new_sample['phase'] = get_category_from_day(datetime.strptime(date_string, '%Y-%m-%d').date())
        samples.append(new_sample)

    return samples