#!/bin/bash
#SBATCH --account=def-sutton
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048M
#SBATCH --array=1-180
#SBATCH --output=/home/asharif/scratch/StreamX_optimizer/outputs/2026_02_12__20_36_28___/5_0/output_%j.txt
#SBATCH --constraint=granite

module purge
module load StdEnv/2023
module load python/3.10
module load mujoco/3.1.6

source ~/venv_streamx/bin/activate
export PYTHONNOUSERSITE=1


$(sed -n "${SLURM_ARRAY_TASK_ID}p" < /home/asharif/scratch/StreamX_optimizer/outputs/2026_02_12__20_36_28___/export_5_0.dat)
echo "Task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
python3 stream_ac_continuous_meta.py --total_steps=$total_steps --policy_gamma=$policy_gamma --policy_entropy_coeff=$policy_entropy_coeff --policy_entrywise_normalization=$policy_entrywise_normalization --policy_beta2=$policy_beta2 --policy_delta_clip=$policy_delta_clip --policy_delta_norm=$policy_delta_norm --policy_momentum=$policy_momentum --policy_lr=$policy_lr --critic_gamma=$critic_gamma --critic_entrywise_normalization=$critic_entrywise_normalization --critic_beta2=$critic_beta2 --critic_delta_clip=$critic_delta_clip --critic_lr=$critic_lr --observer_optimizer=$observer_optimizer --log_backend=$log_backend --log_dir=$log_dir --log_dir_for_pickle=$log_dir_for_pickle --logging_level=$logging_level --project=$project --env_name=$env_name --policy_optimizer=$policy_optimizer --policy_lamda=$policy_lamda --policy_kappa=$policy_kappa --critic_optimizer=$critic_optimizer --critic_lamda=$critic_lamda --critic_kappa=$critic_kappa --critic_meta_stepsize=$critic_meta_stepsize --critic_epsilon_meta=$critic_epsilon_meta --critic_beta2_meta=$critic_beta2_meta --critic_stepsize_parameterization=$critic_stepsize_parameterization --critic_meta_loss_type=$critic_meta_loss_type --critic_meta_shadow_dist_reg=$critic_meta_shadow_dist_reg --seed=$seed --max_time=$max_time --uID=$uID
echo "Program test finished with exit code $? at: `date`"
