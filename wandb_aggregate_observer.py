#!/usr/bin/env python

"""
W&B "Lite" Project Aggregator

This script fetches runs from a "heavy" W&B project, aggregates them based on
a grouping key, performs downsampling and aggregation (mean, std, max) over
intervals, and logs the results to a new "lite" project.

It is designed to be run incrementally, processing only groups that have
been updated since the last run.

*** MODIFIED VERSION ***
This version buffers all log data in memory first, then sorts it by step
and logs it all at once. This fixes the "Steps must be monotonically
increasing" warning.
"""


import os
import math
import wandb
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import sys
from tqdm import tqdm
import time

# --- 1. CONFIGURATION ---
# (Fill these in)

ENV = "Hopper" #  Ant  HalfCheetah  Hopper  Walker2d  Humanoid

# Your source project (the "heavy" one)
SOURCE_PROJECT = f"spo_alpaca/StreamX_OptDesign_Observe_{ENV}-v5"

# Your destination project (the "lite" one)
DEST_PROJECT = "StreamX_OptDesign_Observe_lite"

# Config key to group runs by (e.g., "exp_name", "run_name")
GROUP_CONFIG_KEY = "run_name"

# Logging interval (T): Aggregate data into bins of this many steps
LOGGING_INTERVAL_T = 10000

# --- FIX 2 (Part A): Re-added SPECIAL_MAX_METRIC config ---
# Special metric to calculate the max over the interval (as well as mean)
SPECIAL_MAX_METRIC = "_episode/return"

# Only process groups where at least one run has been updated
# in the last X hours. Set to 0 to process all groups (that aren't
# already up-to-date in the destination).
PROCESS_LAST_X_HOURS = 24_000 

# The config key in your *source* runs that holds the seed number
# This is used to build the 'seeds_detail' string
SOURCE_SEED_CONFIG_KEY = "seed"


# Define metrics to forward-fill (to make them continuous).
# Set to [] to disable all forward-filling (all plots will be sparse).
# User requested "_episode/length" if any.
METRICS_TO_FORWARD_FILL = ["_episode/length"]

SAMPLING_INTERVAL_CONFIG = [
    {
        'interval': 1,
        'plots': [
            'observer/v(s)',
            'observer/abs_delta',
            'observer/delta',
            'observer/td_error',
            'network/observer_w_norm'
        ]
    },
    {
        'interval': 50000,
        'plots': [
            # 'other_sparse_metric'
        ]
    }
]


METRICS_TO_IGNORE = [
    'critic/*',
    'critic_prediction/*',
    'policy/*',
    'rewards/*',
    'system*',
    'observer/norm1_eligibility_trace',
    'observer/clipped_step_size',
    'observer/M',
    'observer/delta_bar',
]

## end of config
#----------------------------------------------------
def parse_iso_timestamp(ts_string):
    """Safely parse W&B's ISO timestamp string."""
    try:
        # Handle both "Z" and "+00:00" UTC formats
        if ts_string.endswith('Z'):
            return datetime.strptime(ts_string, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        else:
            return datetime.fromisoformat(ts_string)
    except Exception:
        return None

# --- MODIFICATION START ---
# This function is now more robust and guarantees a datetime return.
def get_run_update_time_as_datetime(run):
    """Robustly get a run's update time, falling back to summary timestamp."""
    ts = None
    
    # 1. Try updated_at
    try:
        if run.updated_at:
            ts = parse_iso_timestamp(run.updated_at)
        if ts:
            return ts
    except AttributeError:
        pass # Fall through

    # 2. Try summary _timestamp
    try:
        if '_timestamp' in run.summary:
            ts_seconds = run.summary['_timestamp']
            return datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
    except Exception:
        pass # Fall through
    
    # 3. Try created_at
    try:
        print(f"  - WARNING: Could not parse update_at or summary._timestamp for run {run.name}. Using created_at.")
        if run.created_at:
            ts = parse_iso_timestamp(run.created_at)
        if ts:
            return ts
    except Exception:
        pass # Fall through

    # 4. Final fallback
    print(f"  - CRITICAL WARNING: Could not find ANY valid timestamp for run {run.name}. Using a very old date.")
    return datetime.now(timezone.utc) - timedelta(days=365)
# --- MODIFICATION END ---


def main():
    print(f"W&B Aggregator Script Initialized.")
    
    # 1. --- Initialize API ---
    api = wandb.Api(timeout=19) # Set timeout
    
    # 2. --- Build Cache of Existing 'Lite' Runs ---
    print(f"Fetching existing 'lite' runs from {DEST_PROJECT} to check timestamps...")
    dest_runs_cache = {}
    try:
        dest_runs = api.runs(DEST_PROJECT)
        for run in dest_runs:
            if GROUP_CONFIG_KEY in run.config:
                group_name = run.config[GROUP_CONFIG_KEY]
                run_time = get_run_update_time_as_datetime(run)
                
                # Store the most recent run for this group name
                if group_name not in dest_runs_cache or run_time > dest_runs_cache[group_name]['time']:
                    dest_runs_cache[group_name] = {'time': run_time, 'id': run.id}
        print(f"Found {len(dest_runs_cache)} existing processed groups in {DEST_PROJECT}.")
    except Exception as e:
        print(f"  - Warning: Could not fetch destination project. Assuming all runs are new. Error: {e}")

    # 3. --- Fetch and Group All Source Runs ---
    print(f"Fetching all runs from {SOURCE_PROJECT}...")
    try:
        source_runs = api.runs(SOURCE_PROJECT)
    except Exception as e:
        print(f"!!! FATAL: Could not fetch runs from source project {SOURCE_PROJECT}. Error: {e}")
        return

    grouped_runs = defaultdict(list)
    for run in source_runs:
        try:
            group_name = run.config[GROUP_CONFIG_KEY]
            grouped_runs[group_name].append(run)
        except KeyError:
            pass # Skip runs missing the group key
    
    print(f"Found {len(source_runs)} total runs, grouped into {len(grouped_runs)} groups.")

    # 4. --- Process Each Group ---
    
    # Create a reverse lookup for "sample" metrics
    # e.g., {'policy_w_norm': 1, 'other_metric': 50000}
    metric_to_sample_interval = {}
    all_sampled_metrics = set()
    for config_item in SAMPLING_INTERVAL_CONFIG:
        interval = config_item['interval']
        for plot_name in config_item['plots']:
            metric_to_sample_interval[plot_name] = interval
            all_sampled_metrics.add(plot_name)

    # Convert hours to a timedelta
    time_threshold = datetime.now(timezone.utc) - timedelta(hours=PROCESS_LAST_X_HOURS)
    
    group_pbar = tqdm(list(grouped_runs.items()), desc="Processing Groups", unit="group")
    for group_name, run_list in group_pbar:
        
        group_pbar.set_description(f"Group: {group_name[:30]}...")
        
        # 4.1. --- Check Timestamps for Incremental Processing ---
        
        # Get the last update time *from all runs in this source group*
        try:
            run_update_times = [get_run_update_time_as_datetime(run) for run in run_list if run]
            if not run_update_times:
                print(f"  - Skipping (No valid runs): Group {group_name} has no processable runs.")
                continue
            
            latest_source_update = max(run_update_times)
            
            # Check if *any* run in the group is new enough to process
            if latest_source_update < time_threshold:
                print(f"  - Skipping (Too Old): Group's latest update ({latest_source_update.date()}) is older than {PROCESS_LAST_X_HOURS} hours.")
                continue
                
        except Exception as e:
            print(f"  - WARNING: Error checking timestamp for group {group_name}. Processing anyway. Error: {e}")
            latest_source_update = datetime.now(timezone.utc)

        # Check against the 'lite' project cache
        if group_name in dest_runs_cache:
            lite_run_time = dest_runs_cache[group_name]['time']
            
            if lite_run_time >= latest_source_update:
                print(f"  - Skipping (Up-to-date): 'Lite' run from {lite_run_time.date()} is newer than source ({latest_source_update.date()}).")
                continue
            else:
                print(f"  - Processing (Stale): 'Lite' run is from {lite_run_time.date()}, but source was updated on {latest_source_update.date()}. Deleting old run.")
                try:
                    # Delete the old 'lite' run
                    old_lite_run_id = dest_runs_cache[group_name]['id']
                    run_to_delete = api.run(f"{DEST_PROJECT}/{old_lite_run_id}")
                    run_to_delete.delete()
                except Exception as e:
                    print(f"  - WARNING: Could not delete old 'lite' run {old_lite_run_id}. May result in duplicate. Error: {e}")
        else:
            print(f"  - Processing (New): No 'lite' run found for this group.")

        # 4.2. --- Fetch Data for All Seeds in This Group ---
        print("    Fetching data for all seeds...")
        all_seed_data = {} # {seed_num: {metric: pd.Series, ...}, ...}
        all_metrics_in_group = set()
        seed_pbar = tqdm(run_list, desc="  Fetching seed data", leave=False)
        
        seed_numbers = []
        
        for run in seed_pbar:
            try:
                seed = run.config.get('seed', 'unknown_seed')
                seed_numbers.append(str(seed))
                
                # Fetch all history for this run
                history_df = run.history(pandas=True)
                
                # Drop rows where _step is NaN (can corrupt data)
                history_df = history_df.dropna(subset=['_step'])
                
                if history_df.empty:
                    continue
                
                # Set _step as index
                history_df = history_df.set_index('_step')
                
                all_seed_data[seed] = {}
                for col in history_df.columns:
                    
                    # --- FIX 1: This is the main bug fix ---
                    # We skip internal wandb metrics, but *allow* _episode/*
                    if col.startswith('gradients/') or col.startswith('parameters/'):
                        continue
                    if col.startswith('_') and not col.startswith('_episode/'):
                        # This skips _runtime, _timestamp, etc.
                        # but allows _episode/return, _episode/length
                        continue
                    # --- End of FIX 1 ---
                    
                    
                    # Check against METRICS_TO_IGNORE with wildcard support
                    is_ignored = False
                    for pattern in METRICS_TO_IGNORE:
                        if pattern.endswith('*'):
                            # Wildcard match (e.g., 'critic/*')
                            if col.startswith(pattern[:-1]):
                                is_ignored = True
                                break
                        else:
                            # Exact match
                            if col == pattern:
                                is_ignored = True
                                break
                    
                    if is_ignored:
                        continue # Skip this metric
                    
                    # Convert to numeric, coercing errors (like strings) to NaN
                    metric_series = pd.to_numeric(history_df[col], errors='coerce')
                    
                    # Drop NaN values to keep series light
                    metric_series = metric_series.dropna()
                    
                    if not metric_series.empty:
                        all_seed_data[seed][col] = metric_series
                        all_metrics_in_group.add(col)

            except Exception as e:
                print(f"    - ERROR: Could not process run {run.name} (Seed: {seed}). Error: {e}")
                continue
        
        if not all_seed_data:
            print("    - Skipping group: No valid seed data found.")
            continue
            
        print(f"    Fetched data for {len(all_seed_data)}/{len(run_list)} seeds.")
        
        # 4.3. --- Prepare for Logging ---
        print(f"    Processing bins and logging to {DEST_PROJECT}...")
        
        # Get representative config from the first run
        base_config = run_list[0].config
        # Add new seed info
        base_config['seeds'] = len(all_seed_data)
        base_config['seeds_detail'] = "_".join(sorted(list(set(seed_numbers))))
        if 'seed' in base_config:
            del base_config['seed'] # Remove single-run seed

        # Suppress wandb.init messages
        os.environ["WANDB_SILENT"] = "true"
        
        with wandb.init(project=DEST_PROJECT, name=group_name, config=base_config) as new_run:
            
            # *** X-AXIS FIX ***
            # Declare custom step and make all metrics use it
            new_run.define_metric("_step", hidden=True) # declare the step metric
            new_run.define_metric("*", step_metric="_step") # use `_step` for all metrics

            # --- MODIFICATION: Create log buffer ---
            # We will add all data to this dict, keyed by step,
            # and then log it all at the end in sorted order.
            log_buffer = defaultdict(dict)
            print("    Data will be buffered and logged at the end to ensure correct step order.")


            # --- 4.4.1: Process "Sampled" Metrics ---
            
            # This is Phase 1: We loop over the sampling config
            # We process *only* these metrics and *only* at their specified interval.
            
            # Keep track of last-logged values for forward-filling
            last_valid_values = {}
            
            for config_item in SAMPLING_INTERVAL_CONFIG:
                sample_interval = config_item['interval']
                metrics_to_sample = config_item['plots']
                
                # Find the max step *for this specific subset of metrics*
                max_step_for_interval = 0
                for seed_data in all_seed_data.values():
                    for metric_name, metric_series in seed_data.items():
                        if metric_name in metrics_to_sample:
                            if not metric_series.empty:
                                max_step_for_interval = max(max_step_for_interval, metric_series.index.max())
                
                if max_step_for_interval == 0:
                    continue # No data for any of these metrics
                
                num_bins = math.ceil(max_step_for_interval / sample_interval)
                
                
                # --- MODIFICATION START: Split logic for interval 1 vs. interval > 1 ---
                
                if sample_interval == 1:
                    
                    # --- OPTIMIZATION: Split metrics into ffill and sparse ---
                    # We MUST process ffill_metrics by visiting every step (slow)
                    # We CAN process sparse_metrics by visiting only steps with data (fast)
                    
                    ffill_metrics = [m for m in metrics_to_sample if m in METRICS_TO_FORWARD_FILL and m in all_metrics_in_group]
                    sparse_metrics = [m for m in metrics_to_sample if m not in METRICS_TO_FORWARD_FILL and m in all_metrics_in_group]

                    # --- BLOCK 1: Process F-Fill Metrics (Slow but necessary) ---
                    if ffill_metrics:
                        bin_pbar = tqdm(range(1, num_bins + 1), desc=f"    Processing {sample_interval}-step (f-fill)", leave=False, unit="bin")
                        for log_step in bin_pbar:
                            log_data = {} 
                            for metric in ffill_metrics:
                                seed_last_values = []
                                for seed_data in all_seed_data.values():
                                    if metric in seed_data:
                                        val = seed_data[metric].get(log_step)
                                        if val is not None:
                                            seed_last_values.append(val)
                                
                                if seed_last_values:
                                    mean_val = np.mean(seed_last_values)
                                    log_data[metric] = mean_val
                                    last_valid_values[metric] = mean_val
                                elif metric in last_valid_values: # Forward-fill logic
                                    log_data[metric] = last_valid_values[metric]

                            if log_data:
                                log_data["_step"] = int(log_step)
                                # --- MODIFICATION: Buffer instead of logging ---
                                log_buffer[int(log_step)].update(log_data)

                    # --- BLOCK 2: Process Sparse Metrics (FAST) ---
                    if sparse_metrics:
                        # 1. Find all steps that have *any* data for *any* sparse metric
                        print("    Pre-calculating sparse steps...")
                        all_active_steps = set()
                        for metric in sparse_metrics:
                            for seed_data in all_seed_data.values():
                                if metric in seed_data:
                                    all_active_steps.update(seed_data[metric].index)
                        
                        sorted_steps = sorted(list(all_active_steps))
                        print(f"    Found {len(sorted_steps)} active steps for {len(sparse_metrics)} sparse metrics.")

                        # 2. Loop *only* over those steps
                        bin_pbar_sparse = tqdm(sorted_steps, desc=f"    Processing {sample_interval}-step (sparse)", leave=False, unit="bin")
                        for log_step in bin_pbar_sparse:
                            if log_step > max_step_for_interval:
                                continue # Should not happen, but a good safeguard
                            
                            log_data = {} 
                            for metric in sparse_metrics:
                                seed_last_values = []
                                for seed_data in all_seed_data.values():
                                    if metric in seed_data:
                                        val = seed_data[metric].get(log_step)
                                        if val is not None:
                                            seed_last_values.append(val)
                                
                                if seed_last_values:
                                    mean_val = np.mean(seed_last_values)
                                    log_data[metric] = mean_val
                                    # No 'else' needed, as we don't forward-fill sparse metrics
                            
                            if log_data:
                                log_data["_step"] = int(log_step)
                                # --- MODIFICATION: Buffer instead of logging ---
                                log_buffer[int(log_step)].update(log_data)

                            
                else:
                    # --- ORIGINAL PATH for N-step (dense) data (interval > 1) ---
                    bin_pbar = tqdm(range(1, num_bins + 1), desc=f"    Processing {sample_interval}-step bins", leave=False, unit="bin")
                    for i in bin_pbar:
                        end_step = i * sample_interval
                        start_step = end_step - sample_interval + 1
                        log_step = end_step
                        
                        log_data = {} # Don't add _step yet

                        for metric in metrics_to_sample:
                            if metric not in all_metrics_in_group:
                                continue
                                
                            # "Sample-last" logic
                            seed_last_values = []
                            for seed_data in all_seed_data.values():
                                if metric in seed_data:
                                    # Get all values in this bin (original slow slice)
                                    bin_data = seed_data[metric][(seed_data[metric].index >= start_step) & (seed_data[metric].index <= end_step)]
                                    if not bin_data.empty:
                                        # Get the *last* value
                                        seed_last_values.append(bin_data.iloc[-1])
                            
                            if seed_last_values:
                                # If we found new data, log its mean and "remember" it
                                mean_val = np.mean(seed_last_values)
                                log_data[metric] = mean_val
                                last_valid_values[metric] = mean_val
                            
                            elif metric in METRICS_TO_FORWARD_FILL and metric in last_valid_values:
                                # If bin is empty BUT we forward-fill, use the remembered value
                                log_data[metric] = last_valid_values[metric]

                        # Log the aggregated data point for this bin
                        if log_data: # Only log if we have *any* data
                            log_data["_step"] = int(log_step)
                            # --- MODIFICATION: Buffer instead of logging ---
                            log_buffer[int(log_step)].update(log_data)
                
                # --- MODIFICATION END ---
            
            # --- 4.4.2: Process "Default Interval-Averaged" Metrics ---
            
            # This is Phase 2: We process all *remaining* metrics
            
            default_metrics = all_metrics_in_group - all_sampled_metrics
            if not default_metrics:
                print("    All metrics were sampled. No default processing needed.")
                # --- MODIFICATION: We still need to log the buffer! ---
                # The 'continue' was moved to after the buffer log.
            
            else: # Only run this block if there ARE default metrics
                # Find max step for all *default* metrics
                max_step_default = 0
                for seed_data in all_seed_data.values():
                    for metric_name, metric_series in seed_data.items():
                        if metric_name in default_metrics:
                            max_step_default = max(max_step_default, metric_series.index.max())

                if max_step_default == 0:
                    print("    No data found for default metrics.")
                    # --- MODIFICATION: We still need to log the buffer! ---
                    # The 'continue' was moved to after the buffer log.

                else: # Only run this if we have default data
                    num_bins = math.ceil(max_step_default / LOGGING_INTERVAL_T)
                    
                    bin_pbar_default = tqdm(range(1, num_bins + 1), desc=f"    Processing {LOGGING_INTERVAL_T}-step bins", leave=False, unit="bin")
                    for i in bin_pbar_default:
                        end_step = i * LOGGING_INTERVAL_T
                        start_step = end_step - LOGGING_INTERVAL_T + 1
                        log_step = end_step

                        log_data = {} # Don't add _step yet

                        for metric in default_metrics:
                            # "Interval-average" logic
                            seed_means = []
                            
                            # --- FIX 2 (Part B): Re-added logic for special max metric ---
                            all_values_in_bin = []

                            for seed_data in all_seed_data.values():
                                if metric in seed_data:
                                    # Get all values in this bin
                                    bin_data = seed_data[metric][(seed_data[metric].index >= start_step) & (seed_data[metric].index <= end_step)]
                                    if not bin_data.empty:
                                        # Get the *mean* of the bin
                                        seed_means.append(bin_data.mean())
                                        # --- FIX 2 (Part C): Collect all values for max calc ---
                                        if metric == SPECIAL_MAX_METRIC:
                                            all_values_in_bin.extend(bin_data.values)

                            if seed_means:
                                # If we found new data, log its mean and "remember" it
                                mean_val = np.mean(seed_means)
                                log_data[metric] = mean_val
                                last_valid_values[metric] = mean_val
                                
                                # --- FIX 2 (Part D): Calculate and log the max ---
                                if metric == SPECIAL_MAX_METRIC and all_values_in_bin:
                                    max_val = np.max(all_values_in_bin)
                                    log_data[f"{metric}_max"] = max_val
                                    last_valid_values[f"{metric}_max"] = max_val
                            
                            elif metric in METRICS_TO_FORWARD_FILL and metric in last_valid_values:
                                # If bin is empty BUT we forward-fill, use the remembered value
                                log_data[metric] = last_valid_values[metric]
                                # --- FIX 2 (Part E): Forward-fill the max value too ---
                                if metric == SPECIAL_MAX_METRIC and f"{metric}_max" in last_valid_values:
                                    log_data[f"{metric}_max"] = last_valid_values[f"{metric}_max"]

                        # Log the aggregated data point for this bin
                        if log_data: # Only log if we have *any* data
                            # *** X-AXIS FIX ***
                            # Set both the custom _step and W&B's internal "Step"
                            log_data["_step"] = int(log_step)
                            # --- MODIFICATION: Buffer instead of logging ---
                            log_buffer[int(log_step)].update(log_data)

            # --- 4.5. Log All Buffered Data ---
            print(f"\n    All data processing complete. Found {len(log_buffer)} unique steps to log.")
            print("    Sorting and logging all buffered data to W&B...")
            
            if not log_buffer:
                print("    No data to log for this group.")
            else:
                try:
                    # Sort the buffer by the step (the dictionary key)
                    sorted_steps = sorted(log_buffer.keys())
                    
                    log_pbar = tqdm(sorted_steps, desc="    Logging to W&B", leave=False, unit="step")
                    for step in log_pbar:
                        data_point = log_buffer[step]
                        
                        # The 'step' kwarg is what ensures monotonicity
                        new_run.log(data_point, step=step)
                        
                    print(f"\n    Successfully logged {len(sorted_steps)} data points.")
                
                except Exception as e:
                    print(f"\n    !!! ERROR during final sorted log: {e}")
                    print("    Data for this group may be incomplete.")
            
            # --- End of Buffer Logging ---

        # Re-enable wandb.init messages for other scripts
        os.environ["WANDB_SILENT"] = "false"
        print("    Group processing complete.")
        
        # 4.5. --- Clear W&B Cache ---
        # print("    Clearing local W&B cache...")
        # try:
        #     api.cache.clear()
        # except Exception as e:
        #     print(f"    WARNING: Could not clear W&B cache. Error: {e}")
            
    print("\n--- Aggregation script finished! ---")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total script runtime: {end_time - start_time:.2f} seconds")