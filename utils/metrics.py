import motmetrics as mm
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cdist


def get_accumulator(gt_df, preds_df, max_distance_threshold, auto_id=False):
    acc = mm.MOTAccumulator(auto_id=auto_id)

    for frame in sorted(set(gt_df['FrameId']).intersection(set(preds_df['FrameId']))):
        gt_dets = gt_df[gt_df['FrameId'] == frame][['X', 'Y']].values
        pred_dets = preds_df[preds_df['FrameId'] == frame][['X', 'Y']].values
        
        # Compute Euclidean distances between ground truth and MUSSP detections
        distances = cdist(gt_dets, pred_dets, metric='euclidean')
        
        # Set a maximum distance threshold (adjust as needed)
        distances[distances > max_distance_threshold] = np.inf
        
        gt_ids = gt_df[gt_df['FrameId'] == frame]['Id'].values
        pred_ids = preds_df[preds_df['FrameId'] == frame]['Id'].values
        
        # Update the accumulator and get the matching results
        acc.update(gt_ids, pred_ids, distances, frameid=frame)
    
    return acc

def calculate_ground_plane_metrics(gts, preds, names, distances):
    accs = []
    for gt, pred, _, distance in zip(gts, preds, names, distances):
        accs.append(get_accumulator(gt, pred, distance))

    motchallenge_metric_names = {
        'idf1': 'IDF1',
        'idp': 'IDP',
        'idr': 'IDR',
        'recall': 'Rcll',
        'precision': 'Prcn',
        'num_unique_objects': 'GT',
        'mostly_tracked': 'MT',
        'partially_tracked': 'PT',
        'mostly_lost': 'ML',
        'num_false_positives': 'FP',
        'num_misses': 'FN',
        'num_switches': 'IDs',
        'num_fragmentations': 'FM',
        'mota': 'MOTA',
        'motp': 'MOTP',
        'num_transfer': 'IDt',
        'num_ascend': 'IDa',
        'num_migrate': 'IDm',
        "num_frames": 'FRs'
    }

    motchallenge_metrics = [
        "idf1",
        "idp",
        "idr",
        "recall",
        "precision",
        "num_unique_objects",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "num_false_positives",
        "num_misses",
        "num_switches",
        "num_fragmentations",
        "mota",
        "motp",
        "num_transfer",
        "num_ascend",
        "num_migrate",
        "num_frames"
    ]
    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=motchallenge_metrics,
        names=names,
        generate_overall=True
    )

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=motchallenge_metric_names
    )
    print(strsummary)
    return summary

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
    return max_distance

def calculate_metrics(gt_df, preds_df, max_distance_threshold):
    acc = get_accumulator(gt_df, preds_df, max_distance_threshold)
    # acc.mot_events.to_csv('acc_mot_events.csv')
    # Compute metrics
    mh = mm.metrics.create()

    summary = mh.compute_many(
    # [acc, acc.events.loc[0:1]],
        # [acc],
        [acc],
        metrics=mm.metrics.motchallenge_metrics,
        names=['acc'],
        generate_overall=False
    )

    # summary.to_csv(savepath)
    # summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'num_switches', 
    #                                    'idp', 'idr', 'idf1', 'precision', 'recall', 'mostly_tracked', 'mostly_lost'])
    
    # Normalize MOTP
    summary['motp'] = summary['motp'] / max_distance_threshold

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    return summary # Return both summary and frame_matches

