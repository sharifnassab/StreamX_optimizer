#!/bin/bash
#SBATCH --account=def-sutton
#SBATCH --time=02:59:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --array=1-10
#SBATCH --output=/home/asharif/scratch/StreamX_optimizer/outputs/2025_11_14__14_21_59__test0/8_0/output_%j.txt
#SBATCH --constraint=granite

module purge
module load StdEnv/2023
module load python/3.10
module load mujoco/3.1.6

source ~/venv_streamx/bin/activate
export PYTHONNOUSERSITE=1


$(sed -n "${SLURM_ARRAY_TASK_ID}p" < /home/asharif/scratch/StreamX_optimizer/outputs/2025_11_14__14_21_59__test0/export_8_0.dat)
echo "Task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
python3 stream_ac_continuous_offline_observer.py --total_steps=$total_steps --log_backend=$log_backend --log_dir=$log_dir --log_dir_for_pickle=$log_dir_for_pickle --project=$project --log_freq=$log_freq --bin_size=$bin_size --env_name=$env_name --dataset_name=$dataset_name --dataset_path=$dataset_path --observer_hidden_depth=$observer_hidden_depth --observer_hidden_width=$observer_hidden_width --observer_initialization_sparsity=$observer_initialization_sparsity --seed=$seed --observer_optimizer=$observer_optimizer --observer_lamda=$observer_lamda --observer_kappa=$observer_kappa --observer_momentum=$observer_momentum --observer_entrywise_normalization=$observer_entrywise_normalization --observer_beta2=$observer_beta2 --observer_sig_power=$observer_sig_power --observer_in_trace_sample_scaling=$observer_in_trace_sample_scaling --max_time=$max_time --uID=$uID
echo "Program test finished with exit code $? at: `date`"
