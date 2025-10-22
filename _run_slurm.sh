#!/usr/bin/env bash
# run_remote.sh – copy export_*.dat & submit all submitters
# --------------------------------------------------------
# Run from repo root on the cluster *after* git pull.

REMOTE_RESULTS_ROOT=/home/asharif/scratch/StreamX_optimizer/outputs
SLURM_DIR=__slurm

set -euo pipefail

# Use different name to avoid clash with bash's readonly UID variable
RUN_UID=$(<"$SLURM_DIR/last_uid.txt" tr -d '\n')
REMOTE_DIR="$REMOTE_RESULTS_ROOT/$RUN_UID"

#echo "▶ Creating result folders under: $REMOTE_DIR"

# Create main output directory and all subfolders like 0_0/, 1_0/, ...
mkdir -p "$REMOTE_DIR"
for submitter in "$SLURM_DIR"/_submitter_*.sh; do
    base=$(basename "$submitter")
    group_id="${base#_submitter_}"
    group_id="${group_id%.sh}"
    mkdir -p "$REMOTE_DIR/$group_id"
done

# Legacy fallback (if anything still writes to outputs/)
#mkdir -p "$REMOTE_DIR/outputs"

#echo "▶ Copying export_*.dat to $REMOTE_DIR"
cp "$SLURM_DIR"/export_*.dat "$REMOTE_DIR/"

#echo "▶ Submitting Slurm arrays:"
for sh in "$SLURM_DIR"/_submitter_*.sh; do
    echo " "
    echo "  sbatch $sh"
    sbatch "$sh"
done

#echo "✅ All jobs submitted."
