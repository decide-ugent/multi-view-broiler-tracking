import os
from pathlib import Path

import numpy as np
import pandas as pd

FINAL_SUMMARIES_DIR = Path(__file__).resolve().parent.parent / "DATA" / "final_summaries"

# Paper table column order and LaTeX display labels.
METHOD_ORDER = ['sort_3D', 'muSSP3D', 'sort_curve', 'muSSP_curve']
METHOD_DISPLAY_NAMES = {
    'sort_3D': 'sort-3D',
    'muSSP3D': 'muSSP3D',
    'sort_curve': 'sort-curve',
    'muSSP_curve': 'muSSP-curve',
}


def parse_summary_filename(filename):
    """Parse `{method}_fold_{N}_{eval_set}.csv` into method, fold, and eval_set."""
    stem = filename.removesuffix('.csv')
    method, rest = stem.split('_fold_', 1)
    fold_str, eval_set = rest.rsplit('_', 1)
    return method, int(fold_str), eval_set


if __name__ == '__main__':

    all_metrics_to_compare = ['idf1', 'idp', 'idr', 'recall', 'precision',
       'num_unique_objects', 'mostly_tracked', 'partially_tracked',
       'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches',
       'num_fragmentations', 'mota', 'motp', 'num_transfer', 'num_ascend',
       'num_migrate', 'num_frames', 'val_mota']

    list_of_dicts = []

    all_files = os.listdir(FINAL_SUMMARIES_DIR)
    for file in all_files:
        if file.endswith('.csv'):
            method, fold, _eval_set = parse_summary_filename(file)

            df = pd.read_csv(FINAL_SUMMARIES_DIR / file)
            # print(df.head())
            # Get the row where the Unnamed: 0 column value is equal to OVERALL
            overall_row = df[df['Unnamed: 0'] == 'OVERALL']
            # print(overall_row.columns)
            # print(overall_row)

            # Get the metrics from the overall row
            # metrics = overall_row[all_metrics_to_compare]

            new_dict = {}

            # num_unique_objects
            # create three new metrics: MTp, PTp, MLp
            new_dict['MTp'] = overall_row['mostly_tracked'].values[0] / overall_row['num_unique_objects'].values[0]
            new_dict['PTp'] = overall_row['partially_tracked'].values[0] / overall_row['num_unique_objects'].values[0]
            new_dict['MLp'] = overall_row['mostly_lost'].values[0] / overall_row['num_unique_objects'].values[0]

            for metric in all_metrics_to_compare:
                new_dict[metric] = overall_row[metric].values[0]
            
            new_dict['method'] = method
            new_dict['fold'] = fold

            list_of_dicts.append(new_dict)

    df = pd.DataFrame(list_of_dicts)
    # sort by method and fold
    df = df.sort_values(by=['method', 'fold'])


    unique_methods = df['method'].unique()

    final_df = []

    all_metrics_to_compare.append('MTp')
    all_metrics_to_compare.append('PTp')
    all_metrics_to_compare.append('MLp')

    for method in unique_methods:
        df_method = df[df['method'] == method]
        # Calculate the mean and 95% CI for each metric.
        new_row = {}
        for metric in all_metrics_to_compare:
            mean = df_method[metric].mean()
            # Calculate standard error
            std_error = df_method[metric].std() / np.sqrt(len(df_method))
            # Calculate 95% confidence interval
            confidence_interval = 1.96 * std_error
            
            new_row[metric] = f"{mean:.2f} ± {confidence_interval:.2f}"

        new_row['method'] = method
        final_df.append(new_row)
    
    final_df = pd.DataFrame(final_df)
    final_df = final_df.sort_values(by=['method'])

    # Create a mapping for prettier metric names in the table
    metric_name_mapping = {
        'idf1': 'IDF1',
        'recall': 'Rcll',
        'precision': 'Prcn',
        # 'mostly_tracked': 'MT',
        # 'partially_tracked': 'PT',
        # 'mostly_lost': 'ML',
        'MTp': 'MT',
        'PTp': 'PT',
        'MLp': 'ML',
        # 'num_false_positives': 'FP',
        # 'num_misses': 'FN',
        'num_switches': 'IDs',
        'num_fragmentations': 'FM',
        'mota': 'MOTA',
        'motp': 'MOTP',
        'num_transfer': 'IDt',
        'num_ascend': 'IDa',
        'num_migrate': 'IDm',
        # 'num_frames': 'FRs'
    }

    # Select only the metrics we want to show
    metrics_to_show = list(metric_name_mapping.keys())
    
    # Create LaTeX table
    latex_table = "\\begin{table}\n\\centering\n\\begin{tabular}{l|" + "c" * len(final_df) + "}\n"
    
    # Add header with method names
    method_names = [METHOD_DISPLAY_NAMES[m] for m in METHOD_ORDER]

    header = "Metric & " + " & ".join(method_names) + " \\\\\n\\hline\n"
    latex_table += header

    # Add data rows (one for each metric)
    for metric in metrics_to_show:
        metric_name = metric_name_mapping[metric]
        values = []
        for method_key in METHOD_ORDER:
            rows = final_df[final_df['method'] == method_key]
            if rows.empty:
                available = sorted(final_df['method'].unique())
                raise ValueError(
                    f"No summary rows found for method '{method_key}'. "
                    f"Available methods: {available}"
                )
            values.append(rows[metric].iloc[0])
        latex_table += f"{metric_name} & " + " & ".join(values) + " \\\\\n"
    
    # Close table
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Tracking results comparing different methods. Values are presented as mean ± 95\\% confidence interval. †Higher is better for IDF1, lower is better for MOTP.}\n"
    latex_table += "\\label{tab:tracking_results}\n"
    latex_table += "\\end{table}"
    
    # Save to file
    print(latex_table)
    # with open('tracking_results_table.tex', 'w') as f:
    #     f.write(latex_table)