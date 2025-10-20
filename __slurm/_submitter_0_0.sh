#!/bin/bash
#SBATCH --account=def-sutton
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16384M
#SBATCH --array=1-1
#SBATCH --output=/home/asharif/StreamX_optimizer/results/outputs/2025_10_20__13_35_58__test0/0_0/output_%j.txt
#SBATCH --constraint=granite

module purge
module load StdEnv/2023
module load python/3.10
module load mujoco/3.1.6

source ~/venv_streamx/bin/activate
export PYTHONNOUSERSITE=1


$(sed -n "${SLURM_ARRAY_TASK_ID}p" < /home/asharif/StreamX_optimizer/results/outputs/2025_10_20__13_35_58__test0/export_0_0.dat)
echo "Task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
python3 stream_ac_continuous_OptDesign_prediction.py --env_name=$env_name --total_steps=$total_steps --seed=$seed --gamma=$gamma --lamda=$lamda --kappa_policy=$kappa_policy --kappa_value=$kappa_value --observer_type=$observer_type --u_trace_value=$u_trace_value --entryise_normalization_value=$entryise_normalization_value --beta2_value=$beta2_value --log_backend=$log_backend --log_dir=$log_dir --project=$project --max_time=$max_time --uID=$uID
echo "Program test finished with exit code $? at: `date`"
