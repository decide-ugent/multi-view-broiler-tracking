import cv2
import numpy as np
import os
from scipy import stats
import json
from datetime import datetime
import pandas as pd
import glob
import plotly.graph_objects as go

from ground_plane_detection.data_loading import create_dataset_splits, get_category_from_day, load_samples


def calculate_dt(age):
    radius = ((0.3)*age) + 5
    radius = (radius / 17.5) * 0.165
    return radius * 100


def pythonify(json_data):
    correctedDict = {}

    for key, value in json_data.items():
        if isinstance(value, list):
            value = [pythonify(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, dict):
            value = pythonify(value)
        try:
            key = int(key)
        except (ValueError, TypeError):
            pass
        correctedDict[key] = value

    return correctedDict

def get_combined_view_6_frames(frames):
    if len(frames) != 6:
        # add black frames with the same dimensions until we have 6 frames
        while len(frames) < 6:
            frames.append(np.zeros((1080, 1920, 3), dtype=np.uint8))

    horizontal_stack = False
    if horizontal_stack:
        width = int((1920/2)*3)
        height = 1080
        combined_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        quadwidth = int(width/3)
        quadheight = int(height/2)
        combined_image[:quadheight, :quadwidth] = cv2.resize(frames[0], (quadwidth, quadheight))
        combined_image[:quadheight, quadwidth:2*quadwidth] = cv2.resize(frames[1], (quadwidth, quadheight))
        combined_image[quadheight:, :quadwidth] = cv2.resize(frames[3], (quadwidth, quadheight))
        combined_image[quadheight:, quadwidth:2*quadwidth] = cv2.resize(frames[2], (quadwidth, quadheight))
        combined_image[:quadheight, 2*quadwidth:] = cv2.resize(frames[4], (quadwidth, quadheight))
        combined_image[quadheight:, 2*quadwidth:] = cv2.resize(frames[5], (quadwidth, quadheight))
    else:
        width = 1920*2
        height = int(1080*3)
        combined_image = np.zeros((height, width, 3), dtype=np.uint8)
        quadheight = int(height/3)
        quadwidth = int(width/2)
        combined_image[:quadheight, :quadwidth] = cv2.resize(frames[0], (quadwidth, quadheight))
        combined_image[:quadheight, quadwidth:] = cv2.resize(frames[1], (quadwidth, quadheight))
        combined_image[quadheight:2*quadheight, :quadwidth] = cv2.resize(frames[2], (quadwidth, quadheight))
        combined_image[quadheight:2*quadheight, quadwidth:] = cv2.resize(frames[3], (quadwidth, quadheight))

        combined_image[2*quadheight:, :quadwidth] = cv2.resize(frames[4], (quadwidth, quadheight))
        combined_image[2*quadheight:, quadwidth:] = cv2.resize(frames[5], (quadwidth, quadheight))

    return combined_image

def update_topdown_visualization(topdownframe, topdownframe_fused, fusion_result, cam_nums, x, y):
    colors = [(255, 0, 0), (0, 0, 255), (0, 200, 0),(0, 255, 255) ]

    for j in range(len(fusion_result)):
        cam = int(fusion_result[j][2])
        cam_num_index = cam_nums.index(cam)
        color = colors[cam_num_index]
        cv2.circle(topdownframe, (int(fusion_result[j][0]), int(fusion_result[j][1])), 5, color, -1)

    cv2.circle(topdownframe_fused, (int(x), int(y)), 5, (255,0,0), -1)

    return topdownframe, topdownframe_fused

def update_topdown_visualization_with_unconnected(topdownframe, topdownframe_fused, unconnected_dets, cam_nums, i):
    colors = [(255, 0, 0), (0, 0, 255), (0, 200, 0),(0, 255, 255) ]

    cam = int(unconnected_dets[i][2])
    cam_num_index = cam_nums.index(cam)
    color = colors[cam_num_index]
    cv2.circle(topdownframe, (int(unconnected_dets[i][0]), int(unconnected_dets[i][1])), 5, color, -1)
    cv2.circle(topdownframe_fused, (int(unconnected_dets[i][0]), int(unconnected_dets[i][1])), 5, (0, 255, 0), -1)
    return topdownframe, topdownframe_fused

def visualize_one_view(img, bbs_unshaped, points_unshaped, projected_points, cam_num, cam_nums, cam_num_index):
    pen_total_area_height = 500
    pen_total_area_width = 1189
    # colors =  [(0, 0, 255), (0, 255, 0),(255, 0, 0),  (255, 255, 255)]
    colors =  [(0, 255, 0), (0, 255, 0),(0, 255, 0),  (0, 255, 0)]

    for cp in points_unshaped:
        # cv2.circle(img, (int(cp[0]), int(cp[1])), 5,  (0,255,0), -1)
        # cv2.circle(img, (int(cp[0]), int(cp[1])), 3, colors[cam_num_index], -1)
        cv2.drawMarker(img,  (int(cp[0]), int(cp[1])), (0, 0, 255), markerType=cv2.MARKER_TRIANGLE_DOWN, markerSize=15, thickness=2, line_type=cv2.LINE_AA)

    for bb in bbs_unshaped:
        # cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), colors[cam_num_index], 2)
        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0,0,255), 2)
        # cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0,255,0), 2)

    # if projected_points is not None:
    #     topdown_small_corner = np.zeros((pen_total_area_height, pen_total_area_width, 3), dtype=np.uint8)
    #     for pp in projected_points:
    #         cam_num_index = cam_nums.index(cam_num)
    #         cv2.circle(topdown_small_corner, (int(pp[0]), int(pp[1])), 5, colors[cam_num_index], 5)

    #     # resize the topdown image to a small resolution and overlay it on top of the img
    #     # topdown_small_corner = cv2.resize(topdown_small_corner, (0,0), fx=0.3, fy=0.3)
    #     # topleft
    #     # img[0:topdown_small_corner.shape[0], 0:topdown_small_corner.shape[1]] = topdown_small_corner
    #     # bottomleft

    #     img[img.shape[0]-topdown_small_corner.shape[0]:img.shape[0], 0:topdown_small_corner.shape[1]] = topdown_small_corner
    
    return img


def get_bbs_from_txt(ann_path, width, height, th, CLASS_LIST=['broiler'], class_index_to_select=0, with_conf=False, need_normalization=True):
    bbs = []
    if os.path.exists(ann_path):
        with open(ann_path) as fp:
            for idx, line in enumerate(fp):
                obj = line
                class_name, class_index, xmin, ymin, xmax, ymax, conf = get_txt_object_data(obj, width, height, CLASS_LIST, need_normalization=need_normalization)
                if with_conf:
                    bbs.append([xmin, ymin, xmax, ymax, conf])
                else:
                    if float(conf) > th and class_index == class_index_to_select:
                        bbs.append([xmin, ymin, xmax, ymax])
        
    return bbs


def get_txt_object_data(obj, img_width, img_height, CLASS_LIST, need_normalization=True):
    try:
        classId, centerX, centerY, bbox_width, bbox_height, conf = obj.split()
    except ValueError:
        classId, centerX, centerY, bbox_width, bbox_height = obj.split()
        conf = 1.0

    bbox_width = float(bbox_width)
    bbox_height  = float(bbox_height)
    centerX = float(centerX)
    centerY = float(centerY)

    class_index = int(float(classId))
    class_name = CLASS_LIST[class_index]

    # print(centerX, centerY, bbox_width, bbox_height)
    if need_normalization:
        xmin = int(img_width * centerX - img_width * bbox_width/2.0)
        xmax = int(img_width * centerX + img_width * bbox_width/2.0)
        ymin = int(img_height * centerY - img_height * bbox_height/2.0)
        ymax = int(img_height * centerY + img_height * bbox_height/2.0)
    else:
        xmin = int(centerX - bbox_width/2.0)
        xmax = int(centerX + bbox_width/2.0)
        ymin = int(centerY - bbox_height/2.0)
        ymax = int(centerY + bbox_height/2.0)
    return [class_name, class_index, xmin, ymin, xmax, ymax, conf]

def get_reprojection_masks(rootpath, cam_nums):
    """Load reprojection masks (cam_X_projected.jpg). Visible where blue channel > 0."""
    mask_dict = {}
    for cam in cam_nums:
        path = os.path.join(rootpath, f'cam_{cam}_projected.jpg')
        mask_im = cv2.imread(path)
        if mask_im is None:
            raise FileNotFoundError(f'Reprojection mask not found: {path}')
        mask_dict[cam] = mask_im[:, :, 0]
    return mask_dict

def print_results_in_terminal_with_ci(per_fold_results, folds):
    # After processing all folds, calculate confidence intervals
    print("\n95% Confidence Intervals Across Folds:")
    print("Cat\tstarter\t\tgrower\t\tfinisher\ttotal")
    
    print("MODA", end="\t")
    for cat in ["starter", "grower", "finisher", "total"]:
        values = np.array([per_fold_results[fold][cat]["moda"] for fold in folds.keys() 
                          if cat in per_fold_results[fold]])
        mean = np.mean(values)
        if len(values) > 1:
            ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
            ci_range = f"{(ci[1] - ci[0])/2:.1f}"
        else:
            ci_range = "N/A"
        print(f"{mean:.1f} ±{ci_range}", end="\t")
    print("")
    
    print("MODP", end="\t")
    for cat in ["starter", "grower", "finisher", "total"]:
        values = np.array([per_fold_results[fold][cat]["modp"] for fold in folds.keys() 
                          if cat in per_fold_results[fold]])
        mean = np.mean(values)
        if len(values) > 1:
            ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
            ci_range = f"{(ci[1] - ci[0])/2:.1f}"
        else:
            ci_range = "N/A"
        print(f"{mean:.1f} ±{ci_range}", end="\t")
    print("")
    
    print("Prec", end="\t")
    for cat in ["starter", "grower", "finisher", "total"]:
        values = np.array([per_fold_results[fold][cat]["precision"] for fold in folds.keys() 
                          if cat in per_fold_results[fold]])
        mean = np.mean(values)
        if len(values) > 1:
            ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
            ci_range = f"{(ci[1] - ci[0])/2:.1f}"
        else:
            ci_range = "N/A"
        print(f"{mean:.1f} ±{ci_range}", end="\t")
    print("")
    
    print("Rec", end="\t")
    for cat in ["starter", "grower", "finisher", "total"]:
        values = np.array([per_fold_results[fold][cat]["recall"] for fold in folds.keys() 
                          if cat in per_fold_results[fold]])
        mean = np.mean(values)
        if len(values) > 1:
            ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
            ci_range = f"{(ci[1] - ci[0])/2:.1f}"
        else:
            ci_range = "N/A"
        print(f"{mean:.1f} ±{ci_range}", end="\t")
    print("")

    # Print total column as a row for easy copy to LaTeX table
    print("\nTotal row for LaTeX:")
    
    # Get values for total column
    total_values = {}
    metrics = ["moda", "modp", "precision", "recall"]
    metric_names = ["MODA", "MODP", "Prec", "Rec"]
    
    for metric in metrics:
        values = np.array([per_fold_results[fold]["total"][metric] for fold in folds.keys() 
                          if "total" in per_fold_results[fold]])
        mean = np.mean(values)
        if len(values) > 1:
            ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
            ci_range = f"{(ci[1] - ci[0])/2:.1f}"
        else:
            ci_range = "N/A"
        total_values[metric] = (mean, ci_range)
    
    # Print in LaTeX format
    latex_row = ""
    for i, metric in enumerate(metrics):
        mean, ci_range = total_values[metric]
        if i > 0:
            latex_row += " & "
        latex_row += f"{mean:.1f} ±{ci_range}"
    
    print(latex_row)




def print_results_in_terminal(final_averaged_results):
    print('Category', end=' & ')
    for key, value in final_averaged_results.items():
        print(key, end=' & ')
    print('')

    print('MODA', end=' & ')
    for key, value in final_averaged_results.items():
        print(round(value['moda'], 1), end=' & ')
    
    print('')
    print('MODP', end=' & ')
    for key, value in final_averaged_results.items():
        print(round(value['modp'], 1), end=' & ')

    print('')
    print('Precision', end=' & ')
    for key, value in final_averaged_results.items():
        print(round(value['precision'], 1), end=' & ')
    
    print('')
    print('Recall', end=' & ')
    for key, value in final_averaged_results.items():
        print(round(value['recall'], 1), end=' & ')

    print('')

def save_tps_fps_fns(gt_locs, pred_locs, distances, dt):
    false_positives = []
    true_positives = []
    tp_ids = []
    false_negatives = [] # missed detections
    connected_gts = []

    for i in range(len(gt_locs)):
        x, y = gt_locs[i]
        distance = distances[0][i]
        if distance < dt:
            for j in range(len(pred_locs)):
                x_pred, y_pred = pred_locs[j]
                if np.linalg.norm(np.array([x, y]) - np.array([x_pred, y_pred])) == distance:
                    tp_ids.append([j, distance])
                    connected_gts.append(i)

    for index, dist in tp_ids:
        x, y = pred_locs[index]
        true_positives.append([x, y, dist])

    # now get all the other pred locs and count them as false positives
    for i in range(len(pred_locs)):
        if i not in [x[0] for x in tp_ids]:
            x, y = pred_locs[i]
            false_positives.append([x, y, np.inf])

    for i in range(len(gt_locs)):
        if i not in connected_gts:
            x, y = gt_locs[i]
            false_negatives.append([x, y, np.inf])
    
    return true_positives, false_positives, false_negatives

def average_results(data):
    moda = []
    modp = []
    precision = []
    recall = []
    for i in range(len(data)):
        moda.append(data[i]['moda'])
        modp.append(data[i]['modp'])
        precision.append(data[i]['precision'])
        recall.append(data[i]['recall'])
    
    return {
        'moda': sum(moda) / len(moda),
        'modp': sum(modp) / len(modp),
        'precision': sum(precision) / len(precision),
        'recall': sum(recall) / len(recall)
    }

def average_results_per_cat(data):
    results = {
        'starter': {
            'moda': [],
            'modp': [],
            'precision': [],
            'recall': [],
            'num_samples': 0
        },
        'grower': {
            'moda': [],
            'modp': [],
            'precision': [],
            'recall': [],
            'num_samples': 0
        },
        'finisher': {
            'moda': [],
            'modp': [],
            'precision': [],
            'recall': [],
            'num_samples': 0
        }
    }

    for i in range(len(data)):
        results[data[i]['category']]['moda'].append(data[i]['moda'])
        results[data[i]['category']]['modp'].append(data[i]['modp'])
        results[data[i]['category']]['precision'].append(data[i]['precision'])
        results[data[i]['category']]['recall'].append(data[i]['recall'])
        results[data[i]['category']]['num_samples'] += 1

    # now average all the values per list 
    for cat in results:
        results[cat]['moda'] = sum(results[cat]['moda']) / len(results[cat]['moda'])
        results[cat]['modp'] = sum(results[cat]['modp']) / len(results[cat]['modp'])
        results[cat]['precision'] = sum(results[cat]['precision']) / len(results[cat]['precision'])
        results[cat]['recall'] = sum(results[cat]['recall']) / len(results[cat]['recall'])

    
    # add the total results as well 
    results['total'] = average_results(data)
    results['total']['num_samples'] = len(data)

    return  results




def save_params_to_json(params_dict, filename):
    """
    Save only essential parameters to a JSON file, converting distances to integers
    """
    formatted_dict = {}
    for fold, data in params_dict.items():
        # For single best params format
        formatted_dict['fold_' + str(fold)] = {
            'bb_threshold': float(data['bb_threshold']),
            'min_dist': int(data['min_dist']),
            'max_dist': int(data['max_dist']),
            'remove_threshold': int(data['remove_threshold'])
        }
    
    # Save to JSON file with nice formatting
    with open(filename, 'w') as f:
        json.dump(formatted_dict, f, indent=4)
    print(f"\nParameters saved to {filename}")


def get_n_best_params_per_fold(n, results_dir, with_print=False):
    # Path to the parameter sweep results
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    # Dictionary to store best parameters for each fold
    best_params_per_fold = {}
    
    # Process each CSV file separately
    for csv_file in csv_files:
        # Extract fold name from filename
        fold_name = os.path.basename(csv_file).split('.')[1].split('_')[-2]
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Sort by MODA score in descending order
        df = df.sort_values('total/moda', ascending=False)
        
        # Take the top n results for this fold
        top_n_results = df.head(n)
        
        # Store the parameters and calculate statistics for this fold
        params_df = top_n_results[['bb_threshold', 'min_dist', 'max_dist', 'remove_threshold', 'total/moda']]
        
        stats = {
            'params': params_df.to_dict('records'),
            'avg_moda': params_df['total/moda'].mean(),
            'std': {
                'bb_threshold': params_df['bb_threshold'].std(),
                'min_dist': params_df['min_dist'].std(),
                'max_dist': params_df['max_dist'].std(),
                'remove_threshold': params_df['remove_threshold'].std(),
                'moda': params_df['total/moda'].std()
            }
        }
        best_params_per_fold[int(fold_name)] = stats
    
    # Sort the dictionary by fold number
    best_params_per_fold = dict(sorted(best_params_per_fold.items()))

    
    # Print the results for each fold
    if with_print:
        print("\nTop parameters per fold:")
        print("=" * 80)

    averaged_params = {}
    for fold, results in best_params_per_fold.items():
        averaged_params['fold_' + str(fold)] = {
            'bb_threshold': np.mean([p['bb_threshold'] for p in results['params']]),
            'min_dist': np.mean([p['min_dist'] for p in results['params']]),
            'max_dist': np.mean([p['max_dist'] for p in results['params']]),
            'remove_threshold': np.mean([p['remove_threshold'] for p in results['params']])
        }
        if with_print:
            print(f"\nFold: {fold}")
            print(f"Average MODA for top {n} runs: {results['avg_moda']:.4f} ± {results['std']['moda']:.4f}")
            print("\nParameter ranges:")
            print(f"  bb_threshold: {averaged_params['fold_' + str(fold)]['bb_threshold']:.4f} ± {results['std']['bb_threshold']:.4f}")
            print(f"  min_dist: {averaged_params['fold_' + str(fold)]['min_dist']:.2f} ± {results['std']['min_dist']:.2f}")
            print(f"  max_dist: {averaged_params['fold_' + str(fold)]['max_dist']:.2f} ± {results['std']['max_dist']:.2f}")
            print(f"  remove_threshold: {averaged_params['fold_' + str(fold)]['remove_threshold']:.2f} ± {results['std']['remove_threshold']:.2f}")
        
            # print("\nIndividual runs:")
            # for i, params in enumerate(results['params'], 1):
            #     print(f"\nRun {i}:")
            #     print(f"  MODA: {params['total/moda']:.4f}")
            #     print(f"  bb_threshold: {params['bb_threshold']:.4f}")
            #     print(f"  min_dist: {params['min_dist']:.2f}")
            #     print(f"  max_dist: {params['max_dist']:.2f}")
            #     print(f"  remove_threshold: {params['remove_threshold']:.2f}")
            print("-" * 80)
    
    # Save parameters to JSON
    # save_params_to_json(averaged_params, f"top_params_per_fold.json")
    if with_print:
        print(averaged_params)
    return averaged_params



def make_bar_plot_of_params_across_folds(n, results_dir):
    # Path to the parameter sweep results
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    # Dictionary to store best parameters for each fold
    best_params_per_fold = {}
    
    # Process each CSV file separately
    for csv_file in csv_files:
        # Extract fold name from filename
        fold_name = os.path.basename(csv_file).split('.')[1].split('_')[-2]
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Sort by MODA score in descending order
        df = df.sort_values('total/moda', ascending=False)
        
        # Take the top n results for this fold
        top_n_results = df.head(n)
        
        # Store the parameters and calculate statistics for this fold
        params_df = top_n_results[['bb_threshold', 'min_dist', 'max_dist', 'remove_threshold', 'total/moda']]
        
        stats = {
            'params': params_df.to_dict('records'),
            'avg_moda': params_df['total/moda'].mean(),
            'std': {
                'bb_threshold': params_df['bb_threshold'].std(),
                'min_dist': params_df['min_dist'].std(),
                'max_dist': params_df['max_dist'].std(),
                'remove_threshold': params_df['remove_threshold'].std(),
                'moda': params_df['total/moda'].std()
            }
        }
        best_params_per_fold[int(fold_name)] = stats
    
    # Sort the dictionary by fold number
    best_params_per_fold = dict(sorted(best_params_per_fold.items()))

    # Extract fold names and parameter names
    fold_names = list(best_params_per_fold.keys())
    param_names = ['bb_threshold', 'min_dist', 'max_dist', 'remove_threshold']
    
    # Colors for parameters
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Create separate plot for each parameter
    for param_idx, param in enumerate(param_names):
        # Create a new figure for this parameter
        fig = go.Figure()
        
        # Prepare data for plotting
        x_values = fold_names
        y_values = []
        error_y = []
        
        for fold_num in fold_names:
            # Get the mean value for this parameter in this fold
            param_values = [p[param] for p in best_params_per_fold[fold_num]['params']]
            y_values.append(np.mean(param_values))
            
            # Get the standard deviation for error bars
            error_y.append(best_params_per_fold[fold_num]['std'][param])
        
        # Add bar plot trace
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=y_values,
                name=param,
                error_y=dict(
                    type='data',
                    array=error_y,
                    visible=True
                ),
                marker_color=colors[param_idx]
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'{param} Values Across Folds with Confidence Intervals',
            xaxis=dict(
                title='Fold',
                tickmode='array',
                tickvals=x_values,
                ticktext=[f"Fold {f}" for f in x_values]
            ),
            yaxis=dict(
                title=f'{param} Value'
            ),
            width=900,
            height=500
        )
        
        # Save the figure with parameter name in the filename
        output_filename = f"param_{param}_across_folds.png"
        fig.write_image(output_filename)
        print(f"Saved plot for {param} to {output_filename}")