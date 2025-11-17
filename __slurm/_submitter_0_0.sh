#!/bin/bash
#SBATCH --account=def-sutton
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048M
#SBATCH --array=1-150
#SBATCH --output=/home/asharif/scratch/StreamX_optimizer/outputs/2025_11_17__13_50_01__test0/0_0/output_%j.txt
#SBATCH --constraint=granite

module purge
module load StdEnv/2023
module load python/3.10
module load mujoco/3.1.6

source ~/venv_streamx/bin/activate
export PYTHONNOUSERSITE=1


$(sed -n "${SLURM_ARRAY_TASK_ID}p" < /home/asharif/scratch/StreamX_optimizer/outputs/2025_11_17__13_50_01__test0/export_0_0.dat)
echo "Task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
python3 stream_ac_continuous.py --total_steps=$total_steps --policy_gamma=$policy_gamma --policy_lr=$policy_lr --critic_kappa=$critic_kappa --critic_entrywise_normalization=$critic_entrywise_normalization --critic_beta2=$critic_beta2 --critic_gamma=$critic_gamma --critic_lr=$critic_lr --observer_optimizer=$observer_optimizer --log_backend=$log_backend --log_dir=$log_dir --log_dir_for_pickle=$log_dir_for_pickle --logging_level=$logging_level --project=$project --env_name=$env_name --policy_optimizer=$policy_optimizer --policy_lamda=$policy_lamda --policy_kappa=$policy_kappa --policy_momentum=$policy_momentum --policy_u_trace=$policy_u_trace --policy_entrywise_normalization=$policy_entrywise_normalization --policy_beta2=$policy_beta2 --policy_delta_clip=$policy_delta_clip --policy_delta_norm=$policy_delta_norm --policy_sig_power=$policy_sig_power --policy_in_trace_sample_scaling=$policy_in_trace_sample_scaling --policy_entropy_coeff=$policy_entropy_coeff --policy_weight_decay=$policy_weight_decay --critic_optimizer=$critic_optimizer --critic_lamda=$critic_lamda --critic_momentum=$critic_momentum --critic_u_trace=$critic_u_trace --critic_sig_power=$critic_sig_power --critic_in_trace_sample_scaling=$critic_in_trace_sample_scaling --seed=$seed --run_name=$run_name --max_time=$max_time --uID=$uID
echo "Program test finished with exit code $? at: `date`"
