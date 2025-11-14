#!/bin/bash
#SBATCH --account=def-sutton
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --array=1-5
#SBATCH --output=/home/asharif/scratch/StreamX_optimizer/outputs/2025_11_13__22_38_21__test0/0_0/output_%j.txt
#SBATCH --constraint=granite

module purge
module load StdEnv/2023
module load python/3.10
module load mujoco/3.1.6

source ~/venv_streamx/bin/activate
export PYTHONNOUSERSITE=1


$(sed -n "${SLURM_ARRAY_TASK_ID}p" < /home/asharif/scratch/StreamX_optimizer/outputs/2025_11_13__22_38_21__test0/export_0_0.dat)
echo "Task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
python3 stream_ac_continuous_offline_dataset_generator.py --total_steps=$total_steps --log_backend=$log_backend --seed=$seed --env_name=$env_name --save_dir=$save_dir --bin_length=$bin_length --policy_optimizer=$policy_optimizer --policy_lamda=$policy_lamda --policy_kappa=$policy_kappa --policy_gamma=$policy_gamma --policy_lr=$policy_lr --policy_entropy_coeff=$policy_entropy_coeff --critic_optimizer=$critic_optimizer --critic_lamda=$critic_lamda --critic_kappa=$critic_kappa --critic_gamma=$critic_gamma --critic_lr=$critic_lr --max_time=$max_time --uID=$uID
echo "Program test finished with exit code $? at: `date`"
