import numpy as np
import pickle, os
import matplotlib.pyplot as plt

def avg_return_curve(x, y, stride, total_steps):
    """
    Author: Rupam Mahmood (armahmood@ualberta.ca)
    :param x: A list of list of termination steps for each episode. len(x) == total number of runs
    :param y: A list of list of episodic return. len(y) == total number of runs
    :param stride: The timestep interval between two aggregate datapoints to be calculated
    :param total_steps: The total number of time steps to be considered
    :return: time steps for calculated data points, average returns for each data points, std-errs
    """
    assert len(x) == len(y)
    num_runs = len(x)
    avg_ret = np.zeros(total_steps // stride)
    stderr_ret = np.zeros(total_steps // stride)
    steps = np.arange(stride, total_steps + stride, stride)
    for i in range(0, total_steps // stride):
        rets = []
        avg_rets_per_run = []
        for run in range(num_runs):
            xa = np.array(x[run])
            ya = np.array(y[run])
            rets.append(ya[np.logical_and(i * stride < xa, xa <= (i + 1) * stride)].tolist())
            avg_rets_per_run.append(np.mean(rets[-1]))
        avg_ret[i] = np.mean(avg_rets_per_run)
        stderr_ret[i] = np.std(avg_rets_per_run) / np.sqrt(num_runs)
    return steps, avg_ret, stderr_ret

def load_pickle(data_dir):
    all_termination_time_steps, all_episodic_returns, env_name = [], [], ''
    for file in os.listdir(data_dir):
        if file.endswith(".pkl"):
            with open(os.path.join(data_dir, file), "rb") as f:
                episodic_returns, termination_time_steps, env_name = pickle.load(f)
                all_termination_time_steps.append(termination_time_steps)
                all_episodic_returns.append(episodic_returns)
    return all_episodic_returns, all_termination_time_steps



def main(data_dir, title,  int_space, total_steps, label_and_color_map):
    plt.figure(figsize=(8, 5))
    run_dirs = [x for x in  os.listdir(data_dir) if x!='.DS_Store']
    run_dirs = sorted(run_dirs)
    for run_dir in run_dirs:
        label, color = run_dir , 'grey'
        for key in label_and_color_map:
            if key in run_dir:
                label, color, active = label_and_color_map[key]['label'], label_and_color_map[key]['color'], label_and_color_map[key]['plot']
        if active in ['no']:
            continue
        all_episodic_returns, all_termination_time_steps = load_pickle(os.path.join(data_dir,run_dir))
        steps, avg_ret, stderr_ret = avg_return_curve(all_termination_time_steps, all_episodic_returns, int_space, total_steps)
        plt.fill_between(steps, avg_ret - stderr_ret, avg_ret + stderr_ret, color=f"tab:{color if color!='black' else 'gray'}", alpha=0.4)
        plt.plot(steps, avg_ret, linewidth=2.0, color=color, label=label)

    plt.legend()
    plt.xlabel("Time Step", fontsize=14)
    plt.ylabel(f"Average Episodic Return", fontsize=14)
    plt.title(title)
    #plt.show()
    plt.savefig(f"plots/{title}.pdf")

if __name__ == '__main__':
    import argparse

    label_and_color_map = {
        'Policy__ObGD__gam_0.99_lam_0.0_k_3.0_ent_0.01_lr_1.0_wd_0.0_____Critic__ObGD__gam_0.99_lam_0.0_k_2.0_lr_1.0_wd_0.0':
        {'plot':'no', 'label':r'ObGD - ObGD  ($\lambda=0$)', 'color':'grey'},
        'Policy__ObGD__gam_0.99_lam_0.0_k_3.0_ent_0.01_lr_1.0_wd_0.0_____Critic__ObnC__gam_0.99_lam_0.0_k_2.0_en_RMSProp_b2_0.999_u_0.01_lr_1.0_wd_0.0':
        {'plot':'no', 'label':r'ObGD - ObnC  ($\lambda=0$)', 'color':'orange'},
        'Policy__ObnN__gam_0.99_lam_0.0_k_20.0_en_RMSProp_b2_0.999_u_0.01_ent_0.01_lr_1.0_wd_0.0_delTr_0.01_____Critic__ObnC__gam_0.99_lam_0.0_k_2.0_en_RMSProp_b2_0.999_u_0.01_lr_1.0_wd_0.0':
        {'plot':'no', 'label':r'ObnN - ObnC  ($\lambda=0$)', 'color':'red'},
        'Policy__ObGD__gam_0.99_lam_0.8_k_3.0_ent_0.01_lr_1.0_wd_0.0_____Critic__ObGD__gam_0.99_lam_0.8_k_2.0_lr_1.0_wd_0.0':
        {'plot':'yes', 'label':r'ObGD - ObGD  ($\lambda=0.8$)', 'color':'blue'},
        'Policy__ObGD__gam_0.99_lam_0.8_k_3.0_ent_0.01_lr_1.0_wd_0.0____-Critic__ObtC__gam_0.99_lam_0.8_k_2.0_en_RMSProp_b2_0.999_wd_0.0_sigP_2.0_itss_False':
        {'plot':'yes', 'label':r'ObGD - ObtC  ($\lambda=0.8$)', 'color':'black'},
        'Policy__ObtN__gam_0.99_lam_0.8_k_20.0_en_RMSProp_b2_0.999_ent_0.01_wd_0.0_delTr_0.01_sigP_2.0_itss_False____-Critic__ObtC__gam_0.99_lam_0.8_k_2.0_en_RMSProp_b2_0.999_wd_0.0_sigP_2.0_itss_False':
        {'plot':'yes', 'label':r'ObtN - ObtC  ($\lambda=0.8$)', 'color':'green'},
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--mother_dir', type=str, default='/Users/arsalan/Desktop/Codes/StreamX/CC_outputs/Pickles_selected')
    parser.add_argument('--env_name', type=str, default='Ant')
    parser.add_argument('--int_space', type=int, default=100_000)
    parser.add_argument('--total_steps', type=int, default=5_000_000)
    args = parser.parse_args()

    env_name = args.env_name
    #main(data_dir = os.path.join(args.mother_dir,env_name), title=env_name, int_space=args.int_space, total_steps=args.total_steps, label_and_color_map=label_and_color_map)
    for env_name in ['Ant', 'HalfCheetah', 'Hopper', 'Walker', 'Humanoid']:
        print(env_name)
        main(data_dir = os.path.join(args.mother_dir,env_name), title=env_name, int_space=args.int_space, total_steps=args.total_steps, label_and_color_map=label_and_color_map)
    
