#!/usr/bin/env python3
from _slurm_generator import generate_slurm

RESOURCE_DEFAULTS = {
    "account":  "def-sutton",
    "max_time": "06:00:00",
    "cpus":     1,
    "mem":     '2G',
    "gpus":    '0',   #  v100:1,  0
    "constraint": "granite"    # this is a CPU type on Nibi
}

RESOURCE_OVERRIDES = {
    # ("BP", "ResNet18"): {"time": "08:00:00", "gpus": 2, "mem": "32G"},
}

#PYTHON_ENTRYPOINT = "stream_ac_continuous.py"
PYTHON_ENTRYPOINT = "stream_ac_continuous_meta.py"

COMMON_ENV = {
    #"env_name":         "Ant-v5",
    "total_steps":      5_000_000,
    #
    "policy_gamma":     0.99,
    "policy_lr":        1.0,
    #
    "critic_kappa":     2.0,
    "critic_entrywise_normalization": 'RMSProp',
    "critic_beta2":     0.999,
    "critic_gamma":     0.99,
    "critic_lr":        1.0,
    #
    "observer_optimizer": 'none',
    #
    "log_backend":          "wandb_offline",
    "log_dir":              "/home/asharif/scratch/StreamX_optimizer/WandB_offline", #"/home/asharif/StreamX_optimizer/WandB_offline",
    "log_dir_for_pickle":   "/home/asharif/scratch/StreamX_optimizer/Pickles",
    "logging_level":        "light",      # "light" , "heavy"
    "project":              "StreamX_OptDesign_policy_5m_bootstrap",
}



# ------------------------------------------------------------------
# ---------                  SWEEPS                 ----------------
# ------------------------------------------------------------------

run_description = 'test0'

HYPER_SWEEPS = []

environments = ['Ant-v5', 'HalfCheetah-v5', 'Hopper-v5', 'Walker2d-v5', 'Humanoid-v5', 'HumanoidStandup-v5']
seeds = [i for i in range(30)]


if False:  # ObGD - ObGD (standard)
    HYPER_SWEEPS.append({
        "env_name":             environments,
        "policy_optimizer":     ['ObGD'],
        "policy_kappa":         [3], # 3 is optimum consistently
        "policy_entropy_coeff": [0.01],
        "policy_lamda":         [0.0],
        "critic_lamda":         [0.0],
        "critic_optimizer":     ['ObGD'],   # ['ObGD', 'AdaptiveObGD', 'ObGD_sq', 'ObGD_sq_plain', 'Obn', 'ObnC'],
        "seed":                 seeds,
    })

if False:  # Obo - Obo (standard)
    HYPER_SWEEPS.append({
        "env_name":             environments,
        "policy_optimizer":     ['Obo'],
        "policy_lamda":         [0.8],
        "policy_kappa":         [20],
        "policy_momentum":      [0.9],
        "policy_u_trace":       [1],
        "policy_entrywise_normalization": ['RMSProp'],
        "policy_beta2":         [0.999],
        "policy_delta_clip":    ['20_avg_sq_max_20avg__dec_0.9998'],
        "policy_delta_norm":    ['.9998clipAbs'],
        "policy_sig_power":     [2],
        "policy_in_trace_sample_scaling":['False'],
        "policy_entropy_coeff": [0.01],
        "policy_weight_decay":  [0.0],
        "critic_optimizer":     ['Obo'],
        "critic_lamda":         [0.8],
        "critic_kappa":         [2.0], #[1.0, 1.5, 2.0, 3.0],
        "critic_momentum":      [0.9],
        "critic_u_trace":       [1.0],
        "critic_delta_clip":    ['20_avg_sq_max_20avg__dec_0.9998'],
        "critic_entrywise_normalization": ['RMSProp'],
        "critic_beta2":         [0.999],
        "critic_sig_power":     [2],
        "critic_in_trace_sample_scaling":['False'],
        "seed":                 seeds,
        #"run_name":             ["-Obo_k20_rmsp__del_10sq_Abs___Obo_k2_rmsp"],
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
