#!/bin/bash
#SBATCH --account=def-sutton
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048M
#SBATCH --array=1-900
#SBATCH --output=/home/asharif/scratch/StreamX_optimizer/outputs/2025_10_24__23_09_49__test0/1_0/output_%j.txt
#SBATCH --constraint=granite

module purge
module load StdEnv/2023
module load python/3.10
module load mujoco/3.1.6

source ~/venv_streamx/bin/activate
export PYTHONNOUSERSITE=1


$(sed -n "${SLURM_ARRAY_TASK_ID}p" < /home/asharif/scratch/StreamX_optimizer/outputs/2025_10_24__23_09_49__test0/export_1_0.dat)
echo "Task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
python3 stream_ac_continuous.py --total_steps=$total_steps --policy_gamma=$policy_gamma --policy_lamda=$policy_lamda --policy_lr=$policy_lr --critic_kappa=$critic_kappa --critic_entrywise_normalization=$critic_entrywise_normalization --critic_beta2=$critic_beta2 --critic_u_trace=$critic_u_trace --critic_gamma=$critic_gamma --critic_lamda=$critic_lamda --critic_lr=$critic_lr --observer_optimizer=$observer_optimizer --log_backend=$log_backend --log_dir=$log_dir --log_dir_for_pickle=$log_dir_for_pickle --logging_level=$logging_level --project=$project --env_name=$env_name --policy_optimizer=$policy_optimizer --policy_kappa=$policy_kappa --policy_entrywise_normalization=$policy_entrywise_normalization --policy_beta2=$policy_beta2 --policy_u_trace=$policy_u_trace --policy_delta_trace=$policy_delta_trace --policy_entropy_coeff=$policy_entropy_coeff --critic_optimizer=$critic_optimizer --seed=$seed --max_time=$max_time --uID=$uID
echo "Program test finished with exit code $? at: `date`"
