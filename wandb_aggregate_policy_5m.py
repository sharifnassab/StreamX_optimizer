#!/usr/bin/env python

"""
wandb_aggregate_policy_5m.py
W&B "Lite" Project Aggregator (Multi-Environment & Multiprocessing)

This script fetches runs from multiple "heavy" W&B environment-specific 
projects, aggregates them based on a common "base" group name, performs 
downsampling and aggregation, and logs the results to a single new 
"lite" project.

It uses multiprocessing to process groups in parallel.
"""


import wandb
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import sys
from tqdm import tqdm
import multiprocessing

# --- 1. CONFIGURATION ---

# Number of parallel processes to use for processing groups
NUM_PROCESSES = 5

# List of environments to aggregate
ENVS = ["Ant", "HalfCheetah", "Hopper", "Walker2d", "Humanoid"]

# Your source project template (the "heavy" one)
SOURCE_PROJECT_TEMPLATE = "spo_alpaca/StreamX_OptDesign_policy_5m_{ENV}-v5"

# Your single destination project (the "lite" one)
DEST_PROJECT = "StreamX_OptDesign_policy_5m_lite2"

# Config key in source runs to group by (e.g., "run_name")
GROUP_CONFIG_KEY = "run_name"

# This is the prefix format to remove from the GROUP_CONFIG_KEY
# (Using 5 underscores, which matched your logs)
RUN_NAME_PREFIX_TEMPLATE = "{ENV}-v5____"

# Logging interval (T): Aggregate data into bins of this many steps
LOGGING_INTERVAL_T = 50_000

# Special metric to calculate the max over the interval (as well as mean)
SPECIAL_MAX_METRIC = "_episode/return"

# Only process groups where at least one run has been updated
PROCESS_LAST_X_HOURS = 24_000 

# The config key in your *source* runs that holds the seed number
SOURCE_SEED_CONFIG_KEY = "seed"

# Config keys to remove from the "lite" run's config
CONFIG_KEYS_TO_REMOVE_FROM_LITE_RUN = [
    SOURCE_SEED_CONFIG_KEY, 
    "env", 
    "env_name", 
    "environment"
]

# Define metrics to forward-fill (to make them continuous).
METRICS_TO_FORWARD_FILL = ["_episode/length"]


# --- 2. SETUP ---
# (Global API removed, will be instantiated in main/workers)
METRIC_KEYS_TO_SKIP = ["_step", "_runtime", "_timestamp"]

# --- 3. HELPER FUNCTIONS ---

def get_env_and_base_name(run_name, env_list):
    """
    Parses a run name (e.g., "Walker2d-v5_____-exp_1") to find the
    environment and the base group name.
    
    Returns: (env_name, base_group_name)
    """
    for env in env_list:
        prefix = RUN_NAME_PREFIX_TEMPLATE.format(ENV=env)
        if run_name.startswith(prefix):
            base_name = run_name[len(prefix):]
            if base_name:
                return env, base_name
    
    print(f"  - WARNING: Could not parse env prefix from run name '{run_name}'. "
          f"Using full name as base group.", file=sys.stderr)
    return None, run_name


def parse_iso_timestamp(ts_string):
    """Converts W&B ISO timestamp string to a timezone-aware datetime object."""
    try:
        return datetime.fromisoformat(ts_string.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        if isinstance(ts_string, str):
            return datetime.fromisoformat(ts_string)
        raise

def get_run_update_time_as_datetime(run):
    """
    Robustly gets the last update time of a run as a datetime object.
    (Used only in main thread for dest_runs_cache)
    """
    try:
        if run.updated_at:
            return parse_iso_timestamp(run.updated_at)
    except AttributeError:
        pass 
    try:
        if "_timestamp" in run.summary:
            ts_float = run.summary["_timestamp"]
            return datetime.fromtimestamp(ts_float, tz=timezone.utc)
    except Exception:
        pass 
    try:
        if run.created_at:
            print(f"  - WARNING: Could not find update time for run {run.name}. "
                  f"Falling back to create time.", file=sys.stderr)
            return parse_iso_timestamp(run.created_at)
    except AttributeError:
        pass 
    print(f"  - ERROR: Could not determine any timestamp for run {run.name}. "
          "Skipping timestamp checks.", file=sys.stderr)
    return datetime.fromtimestamp(0, tz=timezone.utc)


def process_group(args):
    """
    This is the worker function that processes a single group.
    It runs in a separate process.
    """
    # 1. Unpack arguments
    (
        base_group_name, 
        env_runs_data_dict, 
        dest_runs_cache_entry, 
        time_threshold, 
        group_index_tuple
    ) = args
    
    group_index, group_count = group_index_tuple
    
    # 2. Each worker needs its own wandb.Api instance
    local_api = wandb.Api(timeout=40)

    num_envs_in_group = len(env_runs_data_dict)
    total_seeds_in_group = sum(len(runs) for runs in env_runs_data_dict.values())
    
    # Using print statements here is okay, but output might be interleaved.
    print(f"\n--- Group {group_index}/{group_count}: {base_group_name} ({num_envs_in_group} envs, {total_seeds_in_group} total seeds) ---")

    # --- 4.4. Incremental Processing Logic ---
    
    # 4.4.1. Check source run update times
    all_runs_in_group_data = [run_data for runs_list in env_runs_data_dict.values() for run_data in runs_list]
    run_update_times = []
    
    # Re-implementing get_run_update_time_as_datetime logic from serializable data
    for (run_path, run_config, updated_at, summary_ts, created_at) in all_runs_in_group_data:
        try:
            if updated_at: # This safely handles None
                run_update_times.append(parse_iso_timestamp(updated_at))
                continue
        except Exception: pass
        try:
            if summary_ts: # This safely handles None
                run_update_times.append(datetime.fromtimestamp(summary_ts, tz=timezone.utc))
                continue
        except Exception: pass
        try:
            if created_at: # This safely handles None
                print(f"  - WARNING: Falling back to create time for run {run_path}.", file=sys.stderr)
                run_update_times.append(parse_iso_timestamp(created_at))
                continue
        except Exception: pass
        run_update_times.append(datetime.fromtimestamp(0, tz=timezone.utc))

    if not run_update_times:
        print("  Skipping: No valid runs found for this group.")
        return
    latest_source_update = max(run_update_times)

    # 4.4.2. Check against destination project
    if dest_runs_cache_entry is not None:
        dest_update_time, dest_run_id = dest_runs_cache_entry
        
        if dest_update_time >= latest_source_update:
            print("  Skipping (Up-to-date): 'Lite' run is newer than all source runs.")
            return
        
        if PROCESS_LAST_X_HOURS > 0 and latest_source_update < time_threshold:
            print(f"  Skipping (Stale): All source runs are older than {PROCESS_LAST_X_HOURS} hours. Not re-processing.")
            return
            
        print(f"  Processing (Outdated): 'Lite' run is older. Deleting old run {dest_run_id}...")
        try:
            old_run_to_delete = local_api.run(f"{DEST_PROJECT}/{dest_run_id}")
            old_run_to_delete.delete()
            print("  Old run deleted.")
        except Exception as e:
            print(f"  WARNING: Could not delete old run {dest_run_id}. {e}")
    
    else:
        print("  Processing (New): No 'lite' run found for this group.")

    # --- 4.4.3. Fetch all data for the group ---
    print("  Fetching data for all envs and seeds...")
    
    all_envs_data = {}
    all_metric_keys_set = set()
    processed_seed_details = {}
    max_step = 0
    
    for env_name, run_data_list in env_runs_data_dict.items():
        print(f"    Fetching for env: {env_name} ({len(run_data_list)} seeds)")
        
        all_seeds_data_for_this_env = []
        all_metric_keys_for_this_env = set()
        processed_seed_values_for_this_env = []

        # This loop re-fetches the run object from its path
        for j, (run_path, run_config, _, _, _) in enumerate(tqdm(run_data_list, desc=f"    {env_name} seeds", leave=False)):
            try:
                # *** RE-FETCH RUN OBJECT (now with string path) ***
                run = local_api.run(run_path) 
                
                history_df = run.history(pandas=True)
                if history_df.empty:
                    print(f"    - WARNING: No history for run {run.name} (env {env_name}). Skipping seed.")
                    continue
                
                if "_step" in history_df.columns:
                    history_df = history_df.set_index("_step").sort_index()
                else:
                    print(f"    - WARNING: Run {run.name} is missing '_step' column. Skipping seed.")
                    continue
                
                all_metric_keys_for_this_env.update(history_df.columns)
                all_seeds_data_for_this_env.append(history_df)
                
                if not history_df.empty:
                    try:
                        max_step = max(max_step, history_df.index.max())
                    except TypeError:
                        print(f"  WARNING: Non-numeric step index found in {run.name}.")

                if SOURCE_SEED_CONFIG_KEY in run_config:
                    processed_seed_values_for_this_env.append(str(run_config[SOURCE_SEED_CONFIG_KEY]))
                else:
                    processed_seed_values_for_this_env.append(f"idx_{j}")
                
            except Exception as e:
                print(f"    - FAILED to fetch history for run {run_path}. Error: {e}")
        
        if all_seeds_data_for_this_env:
            all_envs_data[env_name] = all_seeds_data_for_this_env
            all_metric_keys_set.update(all_metric_keys_for_this_env)
            processed_seed_details[env_name] = "_".join(sorted(list(set(processed_seed_values_for_this_env))))

    if not all_envs_data:
        print("  No valid seed data found for this group. Skipping.")
        return
        
    print(f"  Fetched data for {len(all_envs_data)} environments.")
    
    #
    # *** THIS IS THE CORRECTED LINE ***
    # Filter out system metrics
    #
    metric_keys_to_process = [k for k in all_metric_keys_set if k not in METRIC_KEYS_TO_SKIP and not k.startswith("system/")]

    
    if max_step == 0 or pd.isna(max_step):
        print("  Max step is 0 or invalid. No data to process. Skipping.")
        return

    # --- 4.4.4. Bin, Aggregate, and Log to New Run ---
    print(f"  Processing bins and logging to {DEST_PROJECT}...")
    
    # --- Create the new 'lite' config ---
    # Use config from the first run's data tuple
    base_config = all_runs_in_group_data[0][1].copy() 
    
    keys_to_remove = set(CONFIG_KEYS_TO_REMOVE_FROM_LITE_RUN)
    keys_to_remove.add(GROUP_CONFIG_KEY)
    
    for key in keys_to_remove:
        if key in base_config:
            del base_config[key]
    
    base_config[GROUP_CONFIG_KEY] = base_group_name
        
    for env, dfs in all_envs_data.items():
        base_config[f'seeds_{env}'] = len(dfs)
    
    for env, detail_str in processed_seed_details.items():
        base_config[f'seed_detail_{env}'] = detail_str
        
    base_config['environments'] = sorted(list(all_envs_data.keys()))
    
    with wandb.init(
        project=DEST_PROJECT, 
        name=base_group_name, 
        config=base_config, 
        settings=wandb.Settings(silent=True)
    ) as new_run:
        
        new_run.define_metric("_step", hidden=True)
        new_run.define_metric("*", step_metric="_step")
        
        num_bins = int(np.ceil(max_step / LOGGING_INTERVAL_T))
        if num_bins == 0:
            print("  No bins to process. Skipping.")
            return

        last_valid_values = {}
        max_metric_name = f"{SPECIAL_MAX_METRIC}_max"

        for i in tqdm(range(1, num_bins + 1), desc="  Processing bins", leave=False):
            end_step = i * LOGGING_INTERVAL_T
            start_step = end_step - LOGGING_INTERVAL_T + 1
            log_step = end_step
            
            log_data = {"_step": log_step}
            
            for metric in metric_keys_to_process:
                new_metric_section = metric.replace("/", "__")
                
                for env_name, all_seeds_data in all_envs_data.items():
                    seed_means = []
                    all_points_in_bin = []
                    
                    for seed_df in all_seeds_data:
                        if metric in seed_df.columns:
                            bin_data = seed_df[
                                (seed_df.index >= start_step) & 
                                (seed_df.index <= end_step)
                            ][metric].dropna()
                            
                            if not bin_data.empty:
                                seed_means.append(bin_data.mean())
                                if metric == SPECIAL_MAX_METRIC:
                                    all_points_in_bin.extend(bin_data.values)
                    
                    new_metric_key = f"{new_metric_section}/{env_name}"
                    
                    if seed_means:
                        new_val = np.mean(seed_means)
                        log_data[new_metric_key] = new_val
                        last_valid_values[new_metric_key] = new_val
                    elif metric in METRICS_TO_FORWARD_FILL and new_metric_key in last_valid_values:
                        log_data[new_metric_key] = last_valid_values[new_metric_key]
                    
                    if metric == SPECIAL_MAX_METRIC:
                        max_metric_section = max_metric_name.replace("/", "__")
                        new_max_metric_key = f"{max_metric_section}/{env_name}"
                        
                        if all_points_in_bin:
                            new_max_val = np.max(all_points_in_bin)
                            log_data[new_max_metric_key] = new_max_val
                            last_valid_values[new_max_metric_key] = new_max_val
                        elif max_metric_name in METRICS_TO_FORWARD_FILL and new_max_metric_key in last_valid_values:
                            log_data[new_max_metric_key] = last_valid_values[new_max_metric_key]

            if len(log_data) > 1:
                log_data["_step"] = int(log_step)
                new_run.log(log_data, step=int(log_step)) 

    print(f"  Group processing complete for {base_group_name}.")


def main():
    """Main aggregation logic."""
    
    # Instantiate API in main process for setup
    api = wandb.Api(timeout=40) 
    
    # --- 4.1. Get Destination Runs (for incremental checks) ---
    print(f"Fetching existing 'lite' runs from {DEST_PROJECT} to check timestamps...")
    dest_runs_cache = {} # {base_group_name: (update_datetime, run_id)}
    try:
        dest_runs = api.runs(DEST_PROJECT)
        for run in dest_runs:
            if GROUP_CONFIG_KEY in run.config:
                base_group_name = run.config[GROUP_CONFIG_KEY]
                run_time = get_run_update_time_as_datetime(run)
                dest_runs_cache[base_group_name] = (run_time, run.id)
    except Exception as e:
        print(f"Warning: Could not fetch destination project. Will process all groups. Error: {e}", file=sys.stderr)

    # --- 4.2. Get Source Runs and Group Them ---
    print("Fetching all source runs from all environments...")
    
    # Store serializable data, not wandb.Run objects
    all_source_runs_data = []
    for env in ENVS:
        project_name = SOURCE_PROJECT_TEMPLATE.format(ENV=env)
        print(f"Fetching from {project_name}...")
        try:
            source_runs = api.runs(project_name)
            for run in source_runs:
                #
                # *** THIS IS THE CORRECTED BLOCK ***
                #
                string_path = "/".join(run.path) # Convert list to "entity/project/run_id" string
                
                all_source_runs_data.append((
                    string_path,                    # [0] <-- Now a STRING
                    run.config,                     # [1]
                    getattr(run, 'updated_at', None),  # [2]
                    run.summary.get("_timestamp"),  # [3]
                    getattr(run, 'created_at', None),  # [4]
                    env                             # [5]
                ))
            print(f"  Found {len(source_runs)} runs.")
        except Exception as e:
            print(f"FATAL: Could not fetch runs from {project_name}. Error: {e}", file=sys.stderr)
            # Continue to next env even if one fails
            # return # Uncomment this to stop on failure
    
    print("\nGrouping all runs by base name...")
    # grouped_runs_data = {base_group_name: {env_name: [run_data_tuple, ...]}}
    grouped_runs_data = defaultdict(lambda: defaultdict(list))
    
    for (run_path, run_config, updated_at, summary_ts, created_at, project_env) in all_source_runs_data:
        if GROUP_CONFIG_KEY not in run_config:
            print(f"Skipping run {run_path} - missing config key '{GROUP_CONFIG_KEY}'", file=sys.stderr)
            continue
            
        full_run_name = run_config[GROUP_CONFIG_KEY]
        env_from_name, base_group_name = get_env_and_base_name(full_run_name, ENVS)
        
        env_to_use = env_from_name if env_from_name else project_env
        
        grouped_runs_data[base_group_name][env_to_use].append(
            (run_path, run_config, updated_at, summary_ts, created_at)
        )

    print(f"Found {len(all_source_runs_data)} total runs, grouped into {len(grouped_runs_data)} base groups.")

    time_threshold = datetime.now(timezone.utc) - timedelta(hours=PROCESS_LAST_X_HOURS)

    # --- 4.3. Prepare arguments for the pool ---
    all_args = []
    group_count = len(grouped_runs_data)
    for i, (base_group_name, env_runs_dict) in enumerate(grouped_runs_data.items()):
        args_tuple = (
            base_group_name,
            env_runs_dict,
            dest_runs_cache.get(base_group_name), # Pass the specific entry
            time_threshold,
            (i + 1, group_count) # For logging
        )
        all_args.append(args_tuple)

    # --- 4.4. Run processing pool ---
    print(f"\n--- Starting processing pool with {NUM_PROCESSES} workers ---")
    
    # Use imap_unordered for efficiency (process as they come)
    # This acts as a dynamic "pull" queue, exactly as requested.
    # We wrap it with tqdm for a master progress bar.
    with multiprocessing.Pool(NUM_PROCESSES) as pool:
        for _ in tqdm(pool.imap_unordered(process_group, all_args), total=len(all_args), desc="Processing Groups"):
            pass # The work is done in the worker, just need to iterate

    print("\nAll groups processed.")

# --- 5. RUN SCRIPT ---
if __name__ == "__main__":
    # This guard is ESSENTIAL for multiprocessing
    main()