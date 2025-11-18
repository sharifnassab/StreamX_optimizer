#!/usr/bin/env python3
from _slurm_generator import generate_slurm

RESOURCE_DEFAULTS = {
    "account":  "def-sutton",
    "max_time": "12:59:00",
    "cpus":     1,
    "mem":     '10G',
    "gpus":    '0',   #  v100:1,  0
    "constraint": "granite"    # this is a CPU type on Nibi
}

RESOURCE_OVERRIDES = {
    # ("BP", "ResNet18"): {"time": "08:00:00", "gpus": 2, "mem": "32G"},
}

PYTHON_ENTRYPOINT = "stream_ac_continuous_offline_observer.py"

COMMON_ENV = {
    #"env_name":         "Ant-v5",
    "total_steps":      5_000_000,
    #
    "log_backend":      "wandb_offline",
    "log_dir":          "/home/asharif/scratch/StreamX_optimizer/WandB_offline", 
    "log_dir_for_pickle": "/home/asharif/scratch/StreamX_optimizer/pickles_observer",
    "project":          "StreamX_Offline_Observer",
    "log_freq":         400_000,
    "bin_size":         50_000,
}



# ------------------------------------------------------------------
# ---------                  SWEEPS                 ----------------
# ------------------------------------------------------------------

run_description = 'test0'

HYPER_SWEEPS = []

environments = ['Ant', 'HalfCheetah', 'Hopper', 'Walker2d', 'Humanoid']
seeds = [i for i in range(5)]



for env_name in environments:
    dataset_path = f"/home/asharif/scratch/StreamX_optimizer/offline_datasets/ObGD_ObGD_lam_0.8/seed0/{env_name}"
    dataset_name = 'ObGD_ObGD_lam_0.8__seed0'

    if 0: 
        HYPER_SWEEPS.append({
            "env_name":             [env_name],
            "dataset_name":         [dataset_name],
            "dataset_path":         [dataset_path],
            "observer_optimizer":   ['monte_carlo'], #'ObGD_sq', 'ObGD_sq_plain'],# 'AdaptiveObGD', 'ObGD_sq', 'ObGD_sq_plain'],
            "seed":                 seeds,
        })

    if 0: 
        HYPER_SWEEPS.append({
            "env_name":               [env_name],
            "dataset_name":           [dataset_name],
            "dataset_path":           [dataset_path],
            #
            "observer_hidden_depth":  [2],
            "observer_hidden_width":  [128],
            "observer_initialization_sparsity": [0.9],
            "seed":                   seeds,
            #
            "observer_optimizer":     ['ObGD'],
            "observer_kappa":         [2.0],
            "observer_lamda":         [0.8], 
        })
    

    if 0: # no Sparse Init
        HYPER_SWEEPS.append({
            "env_name":               [env_name],
            "dataset_name":           [dataset_name],
            "dataset_path":           [dataset_path],
            #
            "observer_hidden_depth":  [2],
            "observer_hidden_width":  [128],
            "observer_initialization_sparsity": [0.0],
            "seed":                   seeds,
            #
            "observer_optimizer":     ['ObGD'],
            "observer_kappa":         [2.0],
            "observer_lamda":         [0.8], 
        })
    

    if 0: # ObGD larger net
        HYPER_SWEEPS.append({
            "env_name":               [env_name],
            "dataset_name":           [dataset_name],
            "dataset_path":           [dataset_path],
            #
            "observer_hidden_depth":  [5],
            "observer_hidden_width":  [512],
            "observer_initialization_sparsity": [0.9],
            "seed":                   seeds,
            #
            "observer_optimizer":     ['ObGD'],
            "observer_kappa":         [2.0],
            "observer_lamda":         [0.8], 
        })
    
    if 0: 
        HYPER_SWEEPS.append({
            "env_name":               [env_name],
            "dataset_name":           [dataset_name],
            "dataset_path":           [dataset_path],
            #
            "observer_hidden_depth":  [2],
            "observer_hidden_width":  [128],
            "observer_initialization_sparsity": [0.9],
            "seed":                   seeds,
            #
            "observer_optimizer":     ['ObtC'],
            "observer_lamda":         [0.8], 
            "observer_kappa":         [2.0], 
            "observer_entrywise_normalization": ['RMSProp', 'none'],
            "observer_beta2":         [0.999],
            "observer_sig_power":     [2],
            "observer_in_trace_sample_scaling":['False'],
        })

    
    if 0: # OboC
        HYPER_SWEEPS.append({
            "env_name":               [env_name],
            "dataset_name":           [dataset_name],
            "dataset_path":           [dataset_path],
            #
            "observer_hidden_depth":  [2],
            "observer_hidden_width":  [128],
            "observer_initialization_sparsity": [0.9],
            "seed":                   seeds,
            #
            "observer_optimizer":     ['OboC'],
            "observer_lamda":         [0.8], 
            "observer_kappa":         [1.5, 2.0 ,3.0], 
            "observer_momentum":      [0.9], 
            "observer_entrywise_normalization": ['RMSProp'],
            "observer_beta2":         [0.999],
            "observer_sig_power":     [2],
            "observer_in_trace_sample_scaling":['False'],
        })
    
    if 0: # Obo
        HYPER_SWEEPS.append({
            "env_name":               [env_name],
            "dataset_name":           [dataset_name],
            "dataset_path":           [dataset_path],
            #
            "observer_hidden_depth":  [2],
            "observer_hidden_width":  [128],
            "observer_initialization_sparsity": [0.9],
            "seed":                   seeds,
            #
            "observer_optimizer":     ['OboC'],
            "observer_lamda":         [0.8], 
            "observer_kappa":         [2.0], 
            "observer_momentum":      [0.9], 
            "observer_entrywise_normalization": ['RMSProp'],
            "observer_beta2":         [0.999],
            "observer_sig_power":     [2],
            "observer_in_trace_sample_scaling":['True'],
        })

    if 0: # no momentum
        HYPER_SWEEPS.append({
            "env_name":               [env_name],
            "dataset_name":           [dataset_name],
            "dataset_path":           [dataset_path],
            #
            "observer_hidden_depth":  [2],
            "observer_hidden_width":  [128],
            "observer_initialization_sparsity": [0.9],
            "seed":                   seeds,
            #
            "observer_optimizer":     ['OboC'],
            "observer_lamda":         [0.8], 
            "observer_kappa":         [2.0], 
            "observer_momentum":      [0.0], 
            "observer_entrywise_normalization": ['RMSProp', 'none'],
            "observer_beta2":         [0.999],
            "observer_sig_power":     [2],
            "observer_in_trace_sample_scaling":['False'],
        })

    if 0: 
        HYPER_SWEEPS.append({
            "env_name":               [env_name],
            "dataset_name":           [dataset_name],
            "dataset_path":           [dataset_path],
            #
            "observer_hidden_depth":  [2],
            "observer_hidden_width":  [128],
            "observer_initialization_sparsity": [0.9],
            "seed":                   seeds,
            #
            "observer_optimizer":     ['Obo'],
            "observer_lamda":         [0.8], 
            "observer_kappa":         [2.0], 
            "observer_momentum":      [0.0], 
            "observer_delta_clip":    ['10_avg_sq_max_10avg__dec_0.9998'],
            "observer_delta_norm":    ['none'],
            "observer_entrywise_normalization": ['RMSProp'],
            "observer_beta2":         [0.999],
            "observer_sig_power":     [2],
            "observer_in_trace_sample_scaling":['False'],
        })
        


    if 1: # OboC larger net
        HYPER_SWEEPS.append({
            "env_name":               [env_name],
            "dataset_name":           [dataset_name],
            "dataset_path":           [dataset_path],
            #
            "observer_hidden_depth":  [5],
            "observer_hidden_width":  [512],
            "observer_initialization_sparsity": [0.9],
            "seed":                   seeds,
            #
            "observer_optimizer":     ['OboC'],
            "observer_lamda":         [0.8], 
            "observer_kappa":         [2.0], 
            "observer_momentum":      [0.9], 
            "observer_entrywise_normalization": ['RMSProp', 'none'],
            "observer_beta2":         [0.99901],
            "observer_sig_power":     [2],
            "observer_in_trace_sample_scaling":['False'],
        })


    if 0: # no sparse init
        HYPER_SWEEPS.append({
            "env_name":               [env_name],
            "dataset_name":           [dataset_name],
            "dataset_path":           [dataset_path],
            #
            "observer_hidden_depth":  [2],
            "observer_hidden_width":  [128],
            "observer_initialization_sparsity": [0.0],
            "seed":                   seeds,
            #
            "observer_optimizer":     ['OboC'],
            "observer_lamda":         [0.8], 
            "observer_kappa":         [2.0], 
            "observer_momentum":      [0.9], 
            "observer_entrywise_normalization": ['RMSProp'],
            "observer_beta2":         [0.999],
            "observer_sig_power":     [2],
            "observer_in_trace_sample_scaling":['False'],
        })

    if 0: # larger beta2
        HYPER_SWEEPS.append({
            "env_name":               [env_name],
            "dataset_name":           [dataset_name],
            "dataset_path":           [dataset_path],
            #
            "observer_hidden_depth":  [2],
            "observer_hidden_width":  [128],
            "observer_initialization_sparsity": [0.9],
            "seed":                   seeds,
            #
            "observer_optimizer":     ['OboC'],
            "observer_lamda":         [0.8], 
            "observer_kappa":         [2.0], 
            "observer_momentum":      [0.9], 
            "observer_entrywise_normalization": ['RMSProp'],
            "observer_beta2":         [0.9999],
            "observer_sig_power":     [2],
            "observer_in_trace_sample_scaling":['False'],
        })

# ------------------------------------------------------------------
# --------- 2. Normally nothing below needs editing ----------------
# ------------------------------------------------------------------
REMOTE_RESULTS_ROOT = "/home/asharif/scratch/StreamX_optimizer/outputs"
VENV_ACTIVATE     = "~/venv_streamx/bin/activate"

"""
make_slurm.py – writes all Slurm artefacts into ./slurm/
• One export_⟨sweep⟩_⟨grp⟩.dat per (sweep, resource-group)
• One _submitter_⟨sweep⟩_⟨grp⟩.sh per export file
• time ≡ max_time precedence:
      1) sweep's max_time (unless "default")
      2) RESOURCE_OVERRIDES[…]["time"]
      3) RESOURCE_DEFAULTS["time"]
"""

if __name__ == "__main__":
    generate_slurm(
        common_env=COMMON_ENV,
        hyper_sweeps=HYPER_SWEEPS,
        resource_defaults=RESOURCE_DEFAULTS,
        resource_overrides=RESOURCE_OVERRIDES,
        remote_results_root=REMOTE_RESULTS_ROOT,
        python_entrypoint=PYTHON_ENTRYPOINT,
        venv_activate=VENV_ACTIVATE,
        uid_description=run_description,
    )
