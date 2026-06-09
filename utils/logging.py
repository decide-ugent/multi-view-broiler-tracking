import os
import pandas as pd
import wandb


method_dict = {
    'muSSP_curve': 1,
    'muSSP3D': 2,
    'muSSP_3D': 2,
    'sort_curve': 3,
    'sort_3D': 4,
}

def save_summary_to_file(method, summary, guid, params, savefile):
    summary['guid'] = guid
    summary['method'] = method
    summary['method_id'] = method_dict[method]

    for key, value in params.items():
        summary[key] = value
    if os.path.exists(savefile):
        summary.to_csv(savefile, mode='a')
    else:
        summary.to_csv(savefile)

def log_summary_to_wandb(method, summary, guid, params):
    final_summary = {
        'guid': guid,
        'method': method,
        'method_id': method_dict[method]
    }
    for key, value in params.items():
        final_summary[key] = value

    for index, row in summary.iterrows():
        for key, value in row.items():
            final_summary[index + '_' + key] = value

    # print(json.dumps(final_summary, indent=4))
    wandb.log(final_summary)

