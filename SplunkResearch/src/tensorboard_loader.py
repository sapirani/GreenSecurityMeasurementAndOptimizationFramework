import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.plugins.hparams import plugin_data_pb2
import pandas as pd
import os
import glob

def load_tensorboard_data(log_dir, phases=["train"], tags=None):
    print(f"Loading data from: {log_dir}")
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={
        event_accumulator.TENSORS: 0,  # Load all tensors
        event_accumulator.SCALARS: 0,  # Load all scalars
    })
    ea.Reload()
    
    print(f"Available tags: {ea.Tags()}")
    
    data = {}
    for phase in phases:
        phase_data = {}
        
        all_metrics = [
            f"{phase}/distribution_val",
            f"{phase}/distribution_reward",
            f"{phase}/alert_reward",
            f"{phase}/duration_reward",
            f"{phase}/total_reward",
            f"{phase}/alert_val",
            f"{phase}/duration_val",
            f"{phase}/duration_gap",
            f"{phase}/p_values",
            f"{phase}/t_values",
            f"{phase}/degrees_of_freedom",
            f"{phase}/no_agent_alert_val",
            f"{phase}/no_agent_duration_val",
            f"{phase}/rules_duration",
            f"{phase}/rules_alerts",
            f"{phase}/rules_std_duration",
            f"{phase}/rules_durations_gap",
            f"{phase}/episodic_policy"
        ]
        
        metrics_to_load = tags if tags else all_metrics
        
        for metric in metrics_to_load:
            if metric in ea.Tags()['scalars']:
                events = ea.Scalars(metric)
                phase_data[metric] = pd.DataFrame([(e.wall_time, e.step, e.value) for e in events],
                                                  columns=['wall_time', 'step', 'value'])
            elif metric in ea.Tags()['tensors']:
                events = ea.Tensors(metric)
                phase_data[metric] = pd.DataFrame([(e.wall_time, e.step, tf.make_ndarray(e.tensor_proto)) for e in events],
                                                  columns=['wall_time', 'step', 'value'])
        
        data[phase] = phase_data
    
    # Load hyperparameters
    hparams = load_hparams(ea)
    if hparams:
        data['hparams'] = hparams
    else:
        print("No hparams found in this run.")
    
    return data

def load_hparams(ea):
    data = ea._plugin_to_tag_to_content["hparams"]["_hparams_/session_start_info"]
    hparam_data = plugin_data_pb2.HParamsPluginData.FromString(data).session_start_info.hparams
    hparam_dict = {key: hparam_data[key].ListFields()[0][1] for key in hparam_data.keys()}
    return hparam_dict

def load_data_from_multiple_dirs(base_dir, tags=None):
    all_data = {}
    log_dirs = glob.glob(os.path.join(base_dir, "*"))
    
    for log_dir in log_dirs:
        if os.path.isdir(log_dir):
            run_name = os.path.basename(log_dir)
            if run_name in {"train_20241014_230426_1", "train_20241105_202344_1","train_20241106_190929_1"}:
                continue
            print(f"\nProcessing run: {run_name}")
            run_data = load_tensorboard_data(log_dir, tags=tags)
            all_data[run_name] = run_data
    
    return all_data

def save_data_to_csv(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for run_name, run_data in data.items():
        run_dir = os.path.join(output_dir, run_name)

        os.makedirs(run_dir, exist_ok=True)
        for phase, phase_data in run_data.items():
            if phase == 'hparams':
                # Save hyperparameters to a separate CSV
                hparams_df = pd.DataFrame([phase_data])
                hparams_df.to_csv(os.path.join(run_dir, 'hparams.csv'), index=False)
                print(f"Saved {run_name} hyperparameters to {os.path.join(run_dir, 'hparams.csv')}")
            else:
                phase_dir = os.path.join(run_dir, phase)
                os.makedirs(phase_dir, exist_ok=True)
                for metric, df in phase_data.items():
                    filename = os.path.join(phase_dir, f"{metric.split('/')[-1]}.csv")
                    df.to_csv(filename, index=False)
                    print(f"Saved {run_name} - {phase} - {metric} data to {filename}")

def aggregate_data(all_data, metric_to_aggregate, hparams_to_include):
    aggregated_data = []
    for run_name, run_data in all_data.items():
        if 'train' in run_data and metric_to_aggregate in run_data['train']:
            df = run_data['train'][metric_to_aggregate].copy()
            df['run_name'] = run_name
            
            for hparam_to_include in hparams_to_include:
                # Add the specified hparam value if it exists
                if 'hparams' in run_data and hparam_to_include in run_data['hparams']:
                    df[hparam_to_include] = run_data['hparams'][hparam_to_include]
                else:
                    df[hparam_to_include] = None
                
            aggregated_data.append(df)
    
    return pd.concat(aggregated_data, ignore_index=True)

# Example usage
# Example usage
base_dir = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/best_experiments/tensorboard'
output_dir = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/best_experiments/processed_tesnorboard'
tags_to_load =  ["train/p_values", "train/duration_gap", "train/duration"]
all_data = load_data_from_multiple_dirs(base_dir, tags=tags_to_load)
save_data_to_csv(all_data, output_dir)

for metric in tags_to_load:
    aggregated_data = aggregate_data(all_data, metric, ["env_name"])
    aggregated_data.to_csv(os.path.join(output_dir, f"{metric.split('/')[-1]}.csv"), index=False)
