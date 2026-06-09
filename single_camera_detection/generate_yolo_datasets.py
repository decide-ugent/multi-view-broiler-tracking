import glob
import cv2
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from collections import defaultdict
import os
import datetime
import numpy as np
from tqdm import tqdm
import shutil
from utils.data_loading import create_stratified_splits


def get_age_and_phase(timestamp):
    starter_day = datetime.datetime(2022, 10, 24)
    grower_day = datetime.datetime(2022, 11, 4)
    finisher_day = datetime.datetime(2022, 11, 17)
    end_day = datetime.datetime(2022, 12, 6)

    age = (timestamp - starter_day).days
    if timestamp > starter_day and timestamp < grower_day:
        return 'starter', age
    elif timestamp > grower_day and timestamp < finisher_day:
        return 'grower', age
    elif timestamp > finisher_day and timestamp < end_day:
        return 'finisher', age
    else:
        raise ValueError(f"Timestamp {timestamp} is outside the known growth-phase date ranges")

def get_samples(root_dir, visualize=False, with_bb_area=False):
    samples = []

    images_root = os.path.join(root_dir, 'images')
    labels_root = os.path.join(root_dir, 'labels')

    all_images = os.listdir(images_root)
    all_images.sort()
    for im in tqdm(all_images):
        timestamp = im.split('_')[1]
        date = timestamp.split('.')[0]
        parsed_dt = datetime.datetime.strptime(date, '%Y%m%d%H%M%S')

        datapath = os.path.join(images_root, im)
        labelpath = os.path.join(labels_root, im.replace('.jpg', '.txt'))

        phase, age = get_age_and_phase(parsed_dt)

        if with_bb_area:
            image = cv2.imread(datapath)
            height, width, _ = image.shape
            bbs = []
            average_pixels_bb_area = []
            with open(labelpath, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    bb = line.split(' ')
                    bb = [float(x) for x in bb]
                    x_center, y_center, w, h = bb[1] * width, bb[2] * height, bb[3] * width, bb[4] * height
                    x_top_left = x_center - w / 2
                    y_top_left = y_center - h / 2
                    x_bottom_right = x_center + w / 2
                    y_bottom_right = y_center + h / 2
                    bbs.append((x_top_left, y_top_left, x_bottom_right, y_bottom_right))
                    if visualize:
                        cv2.rectangle(image, (int(x_top_left), int(y_top_left)), (int(x_bottom_right), int(y_bottom_right)), (0, 0, 255), 2)
                    average_pixels_bb_area.append(w * h)

            average_bb_area = sum(average_pixels_bb_area) / len(average_pixels_bb_area)
            # print(f'Average BB area: {average_bb_area} pixels')
        else:
            average_bb_area = 0

        if visualize:
            cv2.imshow('image', image)
            key = cv2.waitKey(1)
            if key == 27:
                break


        samples.append({
            'image': datapath,
            'label': labelpath,
            'date': parsed_dt,
            'age': age,
            'phase': phase,
            'average_bb_area': average_bb_area
        })

    if visualize:
        cv2.destroyAllWindows()
    return samples

def plot_age_phase_distribution(samples, number_of_bins, output_path):
    # Get all ages and determine range
    all_ages = [sample['age'] for sample in samples]
    min_age, max_age = min(all_ages), max(all_ages)
    
    # Create 10 bins
    bin_width = (max_age - min_age + 1) / number_of_bins  # +1 to include max_age
    bins = [min_age + i * bin_width for i in range(number_of_bins + 1)]  # 11 edges for 10 bins
    
    # Count samples in each bin by phase
    bin_phase_counts = defaultdict(lambda: defaultdict(int))
    
    for sample in samples:
        age = sample['age']
        phase = sample['phase']
        # Find which bin this age belongs to
        bin_index = min(int((age - min_age) / bin_width), number_of_bins - 1)  # Ensure it's within 0-9
        bin_phase_counts[bin_index][phase] += 1
    
    # Prepare data for stacked bar chart
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(number_of_bins)]
    
    # Create color map
    color_map = {
        'starter': 'rgb(255,127,14)', 
        'grower': 'rgb(44,160,44)',
        'finisher': 'rgb(31,119,180)'
    }
    
    # Create stacked bar chart
    fig = go.Figure()
    
    # Add bars for each phase
    for phase in ['starter', 'grower', 'finisher']:
        phase_counts = [bin_phase_counts[i][phase] for i in range(number_of_bins)]
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=phase_counts,
            name=phase.capitalize(),
            marker_color=color_map[phase]
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Age Distribution of Samples by Phase ({number_of_bins} bins)',
        xaxis_title='Age (days)',
        yaxis_title='Number of Samples',
        barmode='stack',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=bin_centers,
            ticktext=[f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(number_of_bins)]
        )
    )
    
    # Save plot
    fig.write_html(os.path.join(output_path, 'age_phase_distribution.html'))
    fig.write_image(os.path.join(output_path, 'age_phase_distribution.png'))


def plot_average_bb_area_distribution(samples, number_of_bins, output_path):
    """Plot the average bounding box area distribution over time."""
    # Calculate age bins
    ages = [sample['age'] for sample in samples]
    min_age, max_age = min(ages), max(ages)
    bins = np.linspace(min_age, max_age, number_of_bins + 1)
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(number_of_bins)]

    # Calculate average bb area per bin
    bin_areas = [[] for _ in range(number_of_bins)]
    bin_width = (max_age - min_age + 1) / number_of_bins  # +1 to include max_age
    
    for sample in samples:
        age = sample['age']
        average_bb_area = sample['average_bb_area']
        bin_index = min(int((age - min_age) / bin_width), number_of_bins - 1)
        bin_areas[bin_index].append(average_bb_area)
    # Calculate average area per bin
    avg_areas = [np.mean(areas) if areas else 0 for areas in bin_areas]

    # Create line plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=avg_areas,
        mode='lines+markers',
        name='Average BB Area',
        line=dict(color='rgb(31,119,180)', width=2),
        marker=dict(size=8)
    ))

    # Update layout
    fig.update_layout(
        title=f'Average Bounding Box Area vs Age ({number_of_bins} bins)',
        xaxis_title='Age (days)',
        yaxis_title='Average Bounding Box Area (pixels²)',
        xaxis=dict(
            tickmode='array',
            tickvals=bin_centers,
            ticktext=[f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(number_of_bins)]
        ),
        showlegend=False
    )

    # Save plot
    fig.write_html(os.path.join(output_path, 'average_bb_area_distribution.html'))
    fig.write_image(os.path.join(output_path, 'average_bb_area_distribution.png'))

def create_dataset_splits(samples, n_folds=5, test_size=0.2, val_size=0.2, fold_test_size=0.2):
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
    stratified_sampler = create_stratified_splits(samples, n_bins=10)
    
    # First create final train/test split
    all_indices = set(range(len(samples)))
    final_test_indices = set(stratified_sampler(samples, test_size))
    final_train_indices = all_indices - final_test_indices
    
    # Store the final test set
    final_test_samples = [samples[i] for i in final_test_indices]
    
    # Create k-folds from final_train data
    folds = {}
    final_train_samples = [samples[i] for i in final_train_indices]
    
    for fold in range(n_folds):
        # Create a new stratified sampler for the final_train set
        fold_sampler = create_stratified_splits(final_train_samples, n_bins=10)
        
        # Split final_train into fold_test and remaining samples
        fold_test_indices_local = set(fold_sampler(final_train_samples, fold_test_size))
        fold_train_val_indices_local = set(range(len(final_train_samples))) - fold_test_indices_local
        
        # Get the actual samples for fold_test
        fold_test_samples = [final_train_samples[i] for i in fold_test_indices_local]
        
        # Get remaining samples for train/val split
        fold_train_val_samples = [final_train_samples[i] for i in fold_train_val_indices_local]
        
        # Create another stratified sampler for train/val split
        train_val_sampler = create_stratified_splits(fold_train_val_samples, n_bins=10)
        
        # Calculate adjusted validation size from remaining samples
        # Adjust for the fact that we already removed fold_test_size
        val_size_adjusted = val_size / (1 - fold_test_size)
        
        # Split remaining samples into train and validation
        val_indices_local = set(train_val_sampler(fold_train_val_samples, val_size_adjusted))
        train_indices_local = set(range(len(fold_train_val_samples))) - val_indices_local
        
        # Get the actual samples
        fold_train_samples = [fold_train_val_samples[i] for i in train_indices_local]
        fold_val_samples = [fold_train_val_samples[i] for i in val_indices_local]
        
        folds[f'fold_{fold+1}'] = {
            'train_samples': fold_train_samples,
            'val_samples': fold_val_samples,
            'test_samples': fold_test_samples
        }
    
    # Also return the final test set that's held out from all folds
    return folds, final_test_samples
    
def plot_dataset_splits(dataset_splits, final_test_samples, output_path):
    
    # Create a 3x2 subplot figure
    fig = make_subplots(rows=3, cols=2, 
                        subplot_titles=["Final Test Set"] + [f"Fold {i+1}" for i in range(5)],
                        vertical_spacing=0.1)
    
    # Add final test set to the first subplot
    dates = [s['date'] for s in final_test_samples]
    ages = [s['age'] for s in final_test_samples]
    phases = [s['phase'] for s in final_test_samples]
    
    # Create color map for phases
    color_map = {
        'starter': 'rgb(255,127,14)', 
        'grower': 'rgb(44,160,44)',
        'finisher': 'rgb(31,119,180)'
    }
    
    # Plot final test set
    colors = [color_map[phase] for phase in phases]
    fig.add_trace(
        go.Scatter(x=dates, y=[0] * len(dates), mode='markers', 
                  marker=dict(color=colors, size=8),
                  name="Final Test", text=[f"Age: {age} days<br>Phase: {phase}" for age, phase in zip(ages, phases)]),
        row=1, col=1
    )
    
    # Define positions for each fold explicitly
    positions = [
        (1, 2),  # Fold 1: row 1, col 2
        (2, 1),  # Fold 2: row 2, col 1
        (2, 2),  # Fold 3: row 2, col 2
        (3, 1),  # Fold 4: row 3, col 1
        (3, 2)   # Fold 5: row 3, col 2
    ]
    
    # Plot each fold (train, val, test)
    for fold_idx, (fold_name, fold_data) in enumerate(dataset_splits.items()):
        # Get the position for this fold
        if fold_idx < len(positions):
            row, col = positions[fold_idx]
        else:
            # Fallback in case there are more folds than positions
            print(f"Warning: No position defined for {fold_name}")
            continue
            
        # Plot each split type as a separate line
        split_y_positions = {'train_samples': 0, 'val_samples': 1, 'test_samples': 2}
        
        for split_name, split_samples in fold_data.items():
            # Sort samples by date
            sorted_samples = sorted(split_samples, key=lambda x: x['date'])
            
            # Extract data for plotting
            dates = [s['date'] for s in sorted_samples]
            ages = [s['age'] for s in sorted_samples]
            phases = [s['phase'] for s in sorted_samples]
            
            # Create colors based on phases
            colors = [color_map[phase] for phase in phases]
            
            # Plot this split type as a separate line
            y_position = split_y_positions[split_name]
            
            fig.add_trace(
                go.Scatter(
                    x=dates, 
                    y=[y_position] * len(dates), 
                    mode='markers',
                    marker=dict(color=colors, size=8),
                    name=f"{fold_name}_{split_name}", 
                    text=[f"Age: {age} days<br>Phase: {phase}<br>Split: {split_name}" 
                          for age, phase in zip(ages, phases)]
                ),
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        title_text="Dataset Splits Across Time",
        showlegend=False
    )
    
    # Update x-axis titles and y-axis ticks
    for i in range(1, 7):
        row = (i-1) // 2 + 1
        col = (i-1) % 2 + 1
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        
        # For the final test plot, only show one y-tick
        if i == 1:
            fig.update_yaxes(
                title_text="Split Type", 
                tickvals=[0], 
                ticktext=["Test"],
                row=row, col=col
            )
        else:
            # For fold plots, show three y-ticks
            fig.update_yaxes(
                title_text="Split Type", 
                tickvals=[0, 1, 2], 
                ticktext=["Train", "Validation", "Test"],
                row=row, col=col
            )
    
    # Save as PNG and HTML
    fig.write_image(os.path.join(output_path, "dataset_splits_visualization.png"))
    fig.write_html(os.path.join(output_path, "dataset_splits_visualization.html"))
    

def create_yolo_datasets(dataset_splits, final_test_samples, output_base_dir):
    """
    Create YOLO dataset structure for each fold and the final test set.
    
    Args:
        dataset_splits: Dictionary containing the dataset splits for each fold
        final_test_samples: List of samples in the final test set
        output_base_dir: Directory where the datasets will be created
    """
    # Create the output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create a separate datasets directory for all YAML files
    datasets_dir = os.path.join(os.path.dirname(output_base_dir), 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Define class names (for the YAML file)
    class_names = ['broiler']
    
    # Create the final test dataset with the proper YOLO structure
    final_test_dir = os.path.join(output_base_dir, 'final_test')
    os.makedirs(os.path.join(final_test_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(final_test_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(final_test_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(final_test_dir, 'labels', 'test'), exist_ok=True)
    
    # Copy the final test samples to the train folder
    print(f"Creating final test dataset in {final_test_dir}")
    for sample in tqdm(final_test_samples):
        # Copy image
        image_filename = os.path.basename(sample['image'])
        shutil.copy2(sample['image'], os.path.join(final_test_dir, 'images', 'test', image_filename))
        # Copy label
        label_filename = os.path.basename(sample['label'])
        shutil.copy2(sample['label'], os.path.join(final_test_dir, 'labels', 'test', label_filename))
    
    # Now use all the samples in a random fold as the train set for the final test dataset
    # val_samples = dataset_splits['fold_1']['test_samples']
    all_train_samples = []
    for split_name, split_samples in dataset_splits['fold_1'].items():
        all_train_samples.extend(split_samples)

    for sample in tqdm(all_train_samples, desc="Creating train set for final_test"):
        # Copy image
        image_filename = os.path.basename(sample['image'])
        shutil.copy2(sample['image'], os.path.join(final_test_dir, 'images', 'train', image_filename))
        # Copy label
        label_filename = os.path.basename(sample['label'])
        shutil.copy2(sample['label'], os.path.join(final_test_dir, 'labels', 'train', label_filename))
    
    # Create YAML file for the final test dataset in the datasets directory
    final_test_yaml_path = os.path.join(datasets_dir, 'final_test.yaml')
    with open(final_test_yaml_path, 'w') as f:
        f.write(f"path: {final_test_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/test\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
    
    # Create datasets for each fold with proper YOLO structure
    for fold_name, fold_data in dataset_splits.items():
        fold_dir = os.path.join(output_base_dir, fold_name)
        
        # Create directories with proper YOLO structure
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(fold_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(fold_dir, 'labels', split), exist_ok=True)
        
        # Create additional phase-specific test directories
        for phase in ['starter', 'grower', 'finisher']:
            phase_dir = f'test_{phase}'
            os.makedirs(os.path.join(fold_dir, 'images', phase_dir), exist_ok=True)
            os.makedirs(os.path.join(fold_dir, 'labels', phase_dir), exist_ok=True)
        
        # Prepare phase-specific test samples
        phase_test_samples = {
            'starter': [],
            'grower': [],
            'finisher': []
        }
        
        # Group test samples by phase
        for sample in fold_data['test_samples']:
            phase = sample['phase']
            # print(phase)
            if phase in phase_test_samples:
                phase_test_samples[phase].append(sample)
        
        # Select up to 20 samples from each phase
        for phase, samples_list in phase_test_samples.items():
            # Shuffle samples to get a random selection
            np.random.shuffle(samples_list)
            # Take up to 20 samples
            selected_samples = samples_list[:20]
            
            # Copy the selected samples to the phase-specific test directory
            phase_dir = f'test_{phase}'
            for sample in selected_samples:
                # Copy image
                image_filename = os.path.basename(sample['image'])
                shutil.copy2(sample['image'], os.path.join(fold_dir, 'images', phase_dir, image_filename))
                
                # Copy label
                label_filename = os.path.basename(sample['label'])
                shutil.copy2(sample['label'], os.path.join(fold_dir, 'labels', phase_dir, label_filename))

        
        # Copy samples for each split
        print(f"Creating {fold_name} dataset in {fold_dir}")
        for split_name, split_samples in fold_data.items():
            # Map split_name to directory name
            split_dir = split_name.replace('_samples', '')
            
            # Copy samples
            for sample in tqdm(split_samples, desc=f"{fold_name} - {split_dir}"):
                # Copy image
                image_filename = os.path.basename(sample['image'])
                shutil.copy2(sample['image'], os.path.join(fold_dir, 'images', split_dir, image_filename))
                
                # Copy label
                label_filename = os.path.basename(sample['label'])
                shutil.copy2(sample['label'], os.path.join(fold_dir, 'labels', split_dir, label_filename))
        
        # Create YAML file for this fold in the datasets directory
        fold_yaml_path = os.path.join(datasets_dir, f'{fold_name}.yaml')
        with open(fold_yaml_path, 'w') as f:
            f.write(f"path: {fold_dir}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n")
            f.write(f"test: images/test\n")
            f.write(f"nc: {len(class_names)}\n")
            f.write(f"names: {class_names}\n")
    
    # Create phase-specific YAML files for each fold
    for fold_name, fold_data in dataset_splits.items():
        fold_dir = os.path.join(output_base_dir, fold_name)
        # Create a YAML file for each phase (starter, grower, finisher)
        for phase in ['starter', 'grower', 'finisher']:
            phase_yaml_path = os.path.join(datasets_dir, f'{fold_name}_{phase}.yaml')
            with open(phase_yaml_path, 'w') as f:
                f.write(f"path: {fold_dir}\n")
                f.write(f"train: images/train\n")
                f.write(f"val: images/val\n")
                f.write(f"test: images/test_{phase}\n")
                f.write(f"nc: {len(class_names)}\n")
                f.write(f"names: {class_names}\n")

    
    print(f"Created YOLO datasets in {output_base_dir}")
    print(f"Created all YAML config files in {datasets_dir}")
    print(f"- 1 final test dataset (train: final test samples, val: test samples from fold 1)")
    print(f"- {len(dataset_splits)} fold datasets (each with train/val/test splits)")

def generate_yolo_datasets(root_dataset_path, output_datasets_path):
    os.makedirs(output_datasets_path, exist_ok=True)
    output_path = os.path.join(output_datasets_path, 'dataset_figures')
    os.makedirs(output_path, exist_ok=True)

    yolo_output_dir = os.path.join(output_datasets_path, 'yolo_datasets')

    print("Gathering samples...")
    samples = get_samples(root_dataset_path, visualize=False, with_bb_area=True)

    print("Plotting age and phase distribution...")
    plot_age_phase_distribution(samples, 16, output_path)
    print("Plotting average bb area distribution...")
    plot_average_bb_area_distribution(samples, 16, output_path)

    # Create dataset splits - each fold has train/val/test + a separate final test set
    dataset_splits, final_test_samples = create_dataset_splits(
        samples, n_folds=5, test_size=0.05, val_size=0.1, fold_test_size=0.1
    )

    print("Plotting dataset splits...")
    plot_dataset_splits(dataset_splits, final_test_samples, output_path)

    # Print basic statistics
    print(f"\nDataset split statistics:")
    print(f"Final test: {len(final_test_samples)} samples")
    
    for fold, split in dataset_splits.items():
        print(f"\n{fold}:")
        for split_name, split_samples in split.items():
            print(f"{split_name}: {len(split_samples)} samples")
            
    # Create YOLO datasets
    create_yolo_datasets(dataset_splits, final_test_samples, yolo_output_dir)


