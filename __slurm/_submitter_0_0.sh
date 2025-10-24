#!/bin/bash
#SBATCH --account=def-sutton
#SBATCH --time=02:50:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048M
#SBATCH --array=1-300
#SBATCH --output=/home/asharif/scratch/StreamX_optimizer/outputs/2025_10_23__21_18_30__test0/0_0/output_%j.txt
#SBATCH --constraint=granite

module purge
module load StdEnv/2023
module load python/3.10
module load mujoco/3.1.6

source ~/venv_streamx/bin/activate
export PYTHONNOUSERSITE=1


$(sed -n "${SLURM_ARRAY_TASK_ID}p" < /home/asharif/scratch/StreamX_optimizer/outputs/2025_10_23__21_18_30__test0/export_0_0.dat)
echo "Task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
python3 stream_ac_continuous.py --total_steps=$total_steps --policy_optimizer=$policy_optimizer --policy_kappa=$policy_kappa --policy_gamma=$policy_gamma --policy_lamda=$policy_lamda --policy_lr=$policy_lr --policy_entropy_coeff=$policy_entropy_coeff --critic_optimizer=$critic_optimizer --critic_kappa=$critic_kappa --critic_gamma=$critic_gamma --critic_lamda=$critic_lamda --critic_lr=$critic_lr --log_backend=$log_backend --log_dir=$log_dir --logging_level=$logging_level --project=$project --env_name=$env_name --observer_optimizer=$observer_optimizer --observer_kappa=$observer_kappa --observer_entryise_normalization=$observer_entryise_normalization --observer_beta2=$observer_beta2 --observer_u_trace=$observer_u_trace --seed=$seed --max_time=$max_time --uID=$uID
echo "Program test finished with exit code $? at: `date`"
