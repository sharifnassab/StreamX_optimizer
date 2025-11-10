#!/usr/bin/env python

"""
W&B "Lite" Project Aggregator

This script fetches runs from a "heavy" W&B project, aggregates them based on
a grouping key, performs downsampling and aggregation (mean, std, max) over
intervals, and logs the results to a new "lite" project.

It is designed to be run incrementally, processing only groups that have
been updated since the last run.
"""


import wandb
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import sys
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# (Fill these in)

ENV = "Humanoid" #  Ant  HalfCheetah  Hopper  Walker2d  Humanoid

# Your source project (the "heavy" one)
SOURCE_PROJECT = f"spo_alpaca/StreamX_OptDesign_policy_5m_{ENV}-v5"

# Your destination project (the "lite" one)
DEST_PROJECT = "StreamX_OptDesign_policy_5m_lite"

# Config key to group runs by (e.g., "exp_name", "run_name")
GROUP_CONFIG_KEY = "run_name"

# Logging interval (T): Aggregate data into bins of this many steps
LOGGING_INTERVAL_T = 10000

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


# --- 2. SETUP ---
api = wandb.Api(timeout=19) # Increase timeout for large queries
METRIC_KEYS_TO_SKIP = ["_step", "_runtime", "_timestamp"]

# --- 3. HELPER FUNCTIONS ---

def parse_iso_timestamp(ts_string):
    """Converts W&B ISO timestamp string to a timezone-aware datetime object."""
    try:
        # Handle the common case with fractional seconds
        return datetime.fromisoformat(ts_string.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        # Handle cases without fractional seconds or invalid types
        if isinstance(ts_string, str):
            return datetime.fromisoformat(ts_string)
        raise

def get_run_update_time_as_datetime(run):
    """
    Robustly gets the last update time of a run as a datetime object.
    Falls back from updated_at -> summary[_timestamp] -> created_at.
    """
    try:
        # 1. Try 'updated_at' attribute
        if run.updated_at:
            return parse_iso_timestamp(run.updated_at)
    except AttributeError:
        pass # Attribute doesn't exist, try next method
    
    try:
        # 2. Try 'summary["_timestamp"]'
        if "_timestamp" in run.summary:
            ts_float = run.summary["_timestamp"]
            return datetime.fromtimestamp(ts_float, tz=timezone.utc)
    except Exception:
        pass # Summary might be broken, try next method

    try:
        # 3. Fallback to 'created_at'
        if run.created_at:
            print(f"  - WARNING: Could not find update time for run {run.name}. "
                  f"Falling back to create time. This run may be processed unnecessarily.", file=sys.stderr)
            return parse_iso_timestamp(run.created_at)
    except AttributeError:
        pass # This really shouldn't happen

    # 4. If all else fails, return oldest possible time
    print(f"  - ERROR: Could not determine any timestamp for run {run.name}. "
          "Skipping timestamp checks for this run.", file=sys.stderr)
    return datetime.fromtimestamp(0, tz=timezone.utc)


def main():
    """Main aggregation logic."""
    
    # --- 4.1. Get Destination Runs (for incremental checks) ---
    print(f"Fetching existing 'lite' runs from {DEST_PROJECT} to check timestamps...")
    dest_runs_cache = {} # {group_name: (update_datetime, run_id)}
    try:
        dest_runs = api.runs(DEST_PROJECT)
        for run in dest_runs:
            if GROUP_CONFIG_KEY in run.config:
                group_name = run.config[GROUP_CONFIG_KEY]
                run_time = get_run_update_time_as_datetime(run)
                dest_runs_cache[group_name] = (run_time, run.id)
    except Exception as e:
        print(f"Warning: Could not fetch destination project. Will process all groups. Error: {e}", file=sys.stderr)

    # --- 4.2. Get Source Runs and Group Them ---
    print(f"Fetching all runs from {SOURCE_PROJECT}...")
    try:
        source_runs = api.runs(SOURCE_PROJECT)
    except Exception as e:
        print(f"FATAL: Could not fetch runs from {SOURCE_PROJECT}. Error: {e}", file=sys.stderr)
        return

    grouped_runs = defaultdict(list)
    for run in source_runs:
        if GROUP_CONFIG_KEY in run.config:
            group_name = run.config[GROUP_CONFIG_KEY]
            grouped_runs[group_name].append(run)
        else:
            print(f"Skipping run {run.name} - missing config key '{GROUP_CONFIG_KEY}'", file=sys.stderr)

    print(f"Found {len(source_runs)} total runs, grouped into {len(grouped_runs)} groups.")

    # Time threshold for 'PROCESS_LAST_X_HOURS'
    time_threshold = datetime.now(timezone.utc) - timedelta(hours=PROCESS_LAST_X_HOURS)

    # --- 4.3. Iterate Over Each Group ---
    group_count = len(grouped_runs)
    for i, (group_name, run_list) in enumerate(grouped_runs.items()):
        
        print(f"\n--- Group {i+1}/{group_count}: {group_name} ({len(run_list)} seeds) ---")

        # --- 4.4. Incremental Processing Logic ---
        
        # 4.4.1. Check source run update times
        try:
            run_update_times = [get_run_update_time_as_datetime(run) for run in run_list if run]
            if not run_update_times:
                print("  Skipping: No valid runs found for this group.")
                continue
            latest_source_update = max(run_update_times)
        except Exception as e:
            print(f"  ERROR checking source run times: {e}. Skipping group.")
            continue

        # 4.4.2. Check against destination project
        if group_name in dest_runs_cache:
            dest_update_time, dest_run_id = dest_runs_cache[group_name]
            
            # Check 1: Is it already up-to-date?
            if dest_update_time >= latest_source_update:
                print("  Skipping (Up-to-date): 'Lite' run is newer than all source runs.")
                continue
            
            # Check 2: Is it too old to bother *re-processing*?
            # This only applies if it's an *existing* run.
            if PROCESS_LAST_X_HOURS > 0 and latest_source_update < time_threshold:
                print(f"  Skipping (Stale): All source runs are older than {PROCESS_LAST_X_HOURS} hours. Not re-processing.")
                continue
                
            # If not skipped, it's outdated and needs processing
            print(f"  Processing (Outdated): 'Lite' run is older. Deleting old run {dest_run_id}...")
            try:
                old_run_to_delete = api.run(f"{DEST_PROJECT}/{dest_run_id}")
                old_run_to_delete.delete()
                print("  Old run deleted.")
            except Exception as e:
                print(f"  WARNING: Could not delete old run {dest_run_id}. {e}")
        
        else:
            # It's a new group. Always process it the first time.
            print("  Processing (New): No 'lite' run found for this group.")

        # --- 4.4.3. Fetch all data for the group ---
        print("  Fetching data for all seeds...")
        all_seeds_data = [] # List of DataFrames, one per seed
        all_metric_keys = set()
        processed_seed_values = [] # For the new 'seeds_detail' config
        
        for j, run in enumerate(tqdm(run_list, desc="  Fetching seed data", leave=False)):
            try:
                # We only need the history, not all files
                history_df = run.history(pandas=True)
                if history_df.empty:
                    print(f"    - WARNING: No history for run {run.name} (seed {j+1}). Skipping seed.")
                    continue
                
                # Ensure _step is the index
                if "_step" in history_df.columns:
                    history_df = history_df.set_index("_step").sort_index()
                else:
                    print(f"    - WARNING: Run {run.name} is missing '_step' column. Skipping seed.")
                    continue
                
                all_metric_keys.update(history_df.columns)
                all_seeds_data.append(history_df)
                
                # Store the seed value from config for 'seeds_detail'
                if SOURCE_SEED_CONFIG_KEY in run.config:
                    processed_seed_values.append(str(run.config[SOURCE_SEED_CONFIG_KEY]))
                else:
                    # Fallback if key isn't present
                    processed_seed_values.append(f"idx_{j}")
                
            except Exception as e:
                print(f"    - FAILED to fetch history for run {run.name}. Error: {e}")
        
        if not all_seeds_data:
            print("  No valid seed data found for this group. Skipping.")
            continue
            
        print(f"  Fetched data for {len(all_seeds_data)}/{len(run_list)} seeds.")
        
        # Filter out keys we don't want to process
        metric_keys_to_process = [k for k in all_metric_keys if k not in METRIC_KEYS_TO_SKIP]
        
        # Find the max step across all seeds
        max_step = 0
        for df in all_seeds_data:
            if not df.empty:
                try:
                    max_step = max(max_step, df.index.max())
                except TypeError: # Handle cases where index might be non-numeric
                    print(f"  WARNING: Non-numeric step index found. Skipping dataframe.")
        
        if max_step == 0 or pd.isna(max_step):
            print("  Max step is 0 or invalid. No data to process. Skipping.")
            continue

        # --- 4.4.4. Bin, Aggregate, and Log to New Run ---
        print(f"  Processing bins and logging to {DEST_PROJECT}...")
        
        # --- Create the new 'lite' config ---
        base_config = run_list[0].config.copy()
        
        # Remove original seed key if it exists
        if SOURCE_SEED_CONFIG_KEY in base_config:
            del base_config[SOURCE_SEED_CONFIG_KEY]
            
        # Add new seed count and detail args
        base_config['seeds'] = len(all_seeds_data) # Num successfully processed
        base_config['seeds_detail'] = "_".join(sorted(list(set(processed_seed_values)))) # Use set for uniqueness
        
        with wandb.init(
            project=DEST_PROJECT, 
            name=group_name, 
            config=base_config, 
            settings=wandb.Settings(silent=True) # Suppress wandb logging
        ) as new_run:
            
            # *** X-AXIS FIX ***
            # Define _step as the default x-axis for ALL metrics
            new_run.define_metric("_step", hidden=True)          # declare the step metric
            new_run.define_metric("*", step_metric="_step")
            
            # Calculate total number of bins needed
            num_bins = int(np.ceil(max_step / LOGGING_INTERVAL_T))
            if num_bins == 0:
                print("  No bins to process. Skipping.")
                continue

            # "Memory" for forward-fill
            last_valid_values = {}
            max_metric_name = f"{SPECIAL_MAX_METRIC}_max" # Helper variable

            # Iterate through bins, where i is 1, 2, 3,...
            for i in tqdm(range(1, num_bins + 1), desc="  Processing bins", leave=False):
                # Bin i=1 -> end=T, start=1
                # Bin i=2 -> end=2T, start=T+1
                end_step = i * LOGGING_INTERVAL_T
                start_step = end_step - LOGGING_INTERVAL_T + 1
                
                # Set the x-axis step value to i*T
                log_step = end_step
                
                # Get all data points in this bin
                log_data = {"_step": log_step}
                
                for metric in metric_keys_to_process:
                    seed_means = [] # Store the mean for each seed
                    all_points_in_bin = [] # For max calculation
                    
                    for seed_df in all_seeds_data:
                        if metric in seed_df.columns:
                            # Get data for this seed within the bin
                            bin_data = seed_df[
                                (seed_df.index >= start_step) & 
                                (seed_df.index <= end_step)
                            ][metric].dropna()
                            
                            if not bin_data.empty:
                                seed_means.append(bin_data.mean())
                                if metric == SPECIAL_MAX_METRIC:
                                    all_points_in_bin.extend(bin_data.values)
                    
                    # --- Handle Mean Value ---
                    if seed_means:
                        # Calculate, log, and "remember" the new value
                        new_val = np.mean(seed_means)
                        log_data[metric] = new_val
                        last_valid_values[metric] = new_val
                    elif metric in METRICS_TO_FORWARD_FILL and metric in last_valid_values:
                        # No data, but it's a metric we should forward-fill
                        log_data[metric] = last_valid_values[metric]
                    # --- Else (no data, not in fill list) ---
                    # Do nothing. The key is not added to log_data. This is the fix for policy_w_norm.
                    
                    # --- Handle Max Value (only for the special metric) ---
                    if metric == SPECIAL_MAX_METRIC:
                        if all_points_in_bin:
                            # Calculate, log, and "remember" the new max
                            new_max_val = np.max(all_points_in_bin)
                            log_data[max_metric_name] = new_max_val
                            last_valid_values[max_metric_name] = new_max_val
                        elif max_metric_name in METRICS_TO_FORWARD_FILL and max_metric_name in last_valid_values:
                            # Forward-fill if bin is empty
                            log_data[max_metric_name] = last_valid_values[max_metric_name]
                        # --- Else (no data, not in fill list) ---
                        # Do nothing. The key is not added to log_data.

                # Log the aggregated data point for this bin
                # We log if we have *any* data (even just forward-filled data)
                if len(log_data) > 1:
                    log_data["_step"] = int(log_step)               # e.g., 10000, 20000, ...
                    new_run.log(log_data, step=int(log_step)) 
                #if len(log_data) > 1:
                #    new_run.log(log_data)

        print("  Group processing complete.")
        
        # Removed the 'api.cache.clear()' block
        
    print("\nAll groups processed.")

# --- 5. RUN SCRIPT ---
if __name__ == "__main__":
    main()