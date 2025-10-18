#!/usr/bin/env python3
from _slurm_generator import generate_slurm

RESOURCE_DEFAULTS = {
    "account":  "def-sutton",
    "max_time": "00:10:00",
    "cpus":     1,
    "mem":     '4G',
    "gpus":    '0',   #  v100:1,  0
    "constraint": "granite"    # this is a CPU type on Nibi
}

RESOURCE_OVERRIDES = {
    # ("BP", "ResNet18"): {"time": "08:00:00", "gpus": 2, "mem": "32G"},
}

PYTHON_ENTRYPOINT = "stream_ac_continuous_OptDesign_prediction.py"

COMMON_ENV = {
    "env_name":         "Ant-v5",
    "total_steps":      5_000,
    "seed":             0,
    "gamma":            0.99,
    "lamda":            0.0,
    "kappa_policy":     3.0,
    "kappa_value":      2.0,
    "observer_type":    "Obn",
    "u_trace_value":    0.99,
    "entryise_normalization_value": "RMSProp",
    "beta2_value":      0.999,
    "log_backend":      "wandb_offline",
    "log_dir":          "/home/asharif/projects/def-sutton/asharif/StreamX_overshoot_prevention/Git/WandB_offline",
    "project":          "test_stream_CC",
}



# ------------------------------------------------------------------
# ---------                  SWEEPS                 ----------------
# ------------------------------------------------------------------

run_description = 'test0'

HYPER_SWEEPS = []

if 1:
    HYPER_SWEEPS.append({"kappa_value":      [2.0],})

if 0:
    HYPER_SWEEPS.append({
        "beta_reg":             [1],
        "beta_averaging_type":  ['weighted'],
        "max_time":             ["00:15:00"],       # time or "default"
    })
if 0:
    HYPER_SWEEPS.append({
        "beta_reg":             [1,10],
        "beta_averaging_type":  ['uniform'],
        "stepsize_blocks":      ["[2,2]"],
        "max_time":             ["default"],       # time or "default"
    })











# ------------------------------------------------------------------
# --------- 2. Normally nothing below needs editing ----------------
# ------------------------------------------------------------------
REMOTE_RESULTS_ROOT = "/home/asharif/projects/def-sutton/asharif/MetaOptimize_Hierarchical/results/outputs"
VENV_ACTIVATE     = "~/meta_opt_vnv/bin/activate"

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
