#!/usr/bin/env python3
from _slurm_generator import generate_slurm

RESOURCE_DEFAULTS = {
    "account":  "def-sutton",
    "max_time": "04:00:00",
    "cpus":     1,
    "mem":     '2G',
    "gpus":    '0',   #  v100:1,  0
    "constraint": "granite"    # this is a CPU type on Nibi
}

RESOURCE_OVERRIDES = {
    # ("BP", "ResNet18"): {"time": "08:00:00", "gpus": 2, "mem": "32G"},
}

PYTHON_ENTRYPOINT = "stream_ac_continuous.py"

COMMON_ENV = {
    #"env_name":         "Ant-v5",
    "total_steps":      2_000_000,
    #
    "policy_optimizer": 'ObGD',
    "policy_kappa":     3.0,
    "policy_gamma":     0.99,
    "policy_lamda":     0.0,
    "policy_lr":        1.0,
    "policy_entropy_coeff": 0.01,
    #
    "critic_optimizer": 'ObGD_sq',
    "critic_kappa":     2.0,
    "critic_gamma":     0.99,
    "critic_lamda":     0.0,
    "critic_lr":        1.0,
    #
    "log_backend":      "wandb_offline",
    "log_dir":          "/home/asharif/scratch/StreamX_optimizer/WandB_offline", #"/home/asharif/StreamX_optimizer/WandB_offline",
    "logging_level":    "heavy",      # "light" , "heavy"
    "project":          "StreamX_OptDesign_Observe",
}



# ------------------------------------------------------------------
# ---------                  SWEEPS                 ----------------
# ------------------------------------------------------------------

run_description = 'test0'

HYPER_SWEEPS = []

environments = ['Ant-v5', 'HalfCheetah-v5', 'Hopper-v5', 'Walker2d-v5', 'Humanoid-v5']
seeds = [i for i in range(30)]



if 0: 
    HYPER_SWEEPS.append({
        "env_name":             environments,
        "observer_optimizer":   ['monte_carlo'], #'ObGD_sq', 'ObGD_sq_plain'],# 'AdaptiveObGD', 'ObGD_sq', 'ObGD_sq_plain'],
        "seed":                 [0],
    })

if 0: 
    HYPER_SWEEPS.append({
        "env_name":             environments,
        "observer_optimizer":     ['ObGD'],#, 'ObGD_sq', 'ObGD_sq_plain'],# 'AdaptiveObGD', 'ObGD_sq', 'ObGD_sq_plain'],
        "observer_kappa":         [2.0], #[1.0, 1.5, 2.0, 3.0],
        "observer_lamda":         [0.8], 
        "seed":                 seeds,
    })

if 0: 
    HYPER_SWEEPS.append({
        "env_name":             environments,
        "observer_optimizer":     ['Obn', 'ObnC'],
        "observer_kappa":         [2.0], #[1.0, 1.5, 2.0, 3.0],
        "observer_entrywise_normalization": ['RMSProp', 'none'],
        "observer_beta2":         [0.999],
        "observer_u_trace":       [0.01, .999],
        "seed":                 seeds,
    })


if 0: 
    HYPER_SWEEPS.append({
        "env_name":             environments,
        "observer_optimizer":     ['ObtC'],
        "observer_lamda":         [0.8], 
        "observer_kappa":         [2.0], #[1.0, 1.5, 2.0, 3.0],
        "observer_entrywise_normalization": ['RMSProp'],
        "observer_beta2":         [0.999],
        "observer_sig_power":     [2],
        "observer_in_trace_sample_scaling":['False', 'True'],
        "seed":                 seeds,
    })

if 0: 
    HYPER_SWEEPS.append({
        "env_name":             environments,
        "observer_optimizer":     ['ObtC'],
        "observer_lamda":         [0.8], 
        "observer_kappa":         [2.0], #[1.0, 1.5, 2.0, 3.0],
        "observer_entrywise_normalization": ['RMSProp'],
        "observer_beta2":         [0.999],
        "observer_sig_power":     [1],
        "observer_in_trace_sample_scaling":['False'],
        "seed":                 seeds,
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
