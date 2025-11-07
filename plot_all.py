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


def resolve_label_color_active(run_dir, label_and_color_map):
    label, color, active = run_dir, 'grey', True
    for key, cfg in label_and_color_map.items():
        if key in run_dir:
            label = cfg.get('label', label)
            color = cfg.get('color', color)
            active = str(cfg.get('plot', 'yes')).lower() not in ['no', 'false', '0']
    if color == 'black':  # avoid 'tab:black' -> use gray fill
        color = 'gray'
    return label, color, active

def load_and_compute_curve(data_dir, run_dir, int_space, total_steps):
    episodic_returns, termination_steps = load_pickle(os.path.join(data_dir, run_dir))
    return avg_return_curve(termination_steps, episodic_returns, int_space, total_steps)


def main_plot_in_separate_figures(data_dir, title, int_space, total_steps, label_and_color_map):
    # One figure per environment (many curves), saved as plots/{title}.pdf
    run_dirs = sorted([d for d in os.listdir(data_dir) if d != '.DS_Store'])
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    legend_handles, legend_labels = [], []
    for run_dir in run_dirs:
        label, color, active = resolve_label_color_active(run_dir, label_and_color_map)
        if not active: 
            continue
        steps, average_return, stderr_return = load_and_compute_curve(data_dir, run_dir, int_space, total_steps)
        handle = plt.fill_between(steps, average_return - stderr_return, average_return + stderr_return,
                                  color=f"tab:{color}", alpha=0.35)
        line_handle, = plt.plot(steps, average_return, linewidth=2, color=color, label=label)
        if label not in legend_labels:
            legend_handles.append(line_handle); legend_labels.append(label)

    plt.xlabel("Time Step"); plt.ylabel("Average Episodic Return")
    plt.title(title)
    if legend_handles: plt.legend(legend_handles, legend_labels)
    plt.savefig(f"plots/{title}.pdf", bbox_inches='tight', dpi=300)
    plt.close()




def main_plot_all_in_single_figure(data_dir, title, int_space, total_steps, label_and_color_map):
    #One big 2x3 figure:  5 subplots = 5 environments
    if not hasattr(main_plot_all_in_single_figure, "_accum"):
        os.makedirs("plots", exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, constrained_layout=True)
        axes = axes.ravel()
        legend_ax = axes[5]; legend_ax.axis('off')

        # Hide redundant y-tick labels: keep on axes[0] and axes[3] (first col)
        for i in [1, 2, 4]:
            axes[i].tick_params(labelleft=False)

        main_plot_all_in_single_figure._accum = {
            "fig": fig,
            "axes": axes,
            "legend_ax": legend_ax,
            "legend_handles": [],
            "legend_labels": [],
            "index": 0,
            "row_stats": {
                0: {"ymin": float("inf"), "ymax": float("-inf"), "axes": [axes[0], axes[1], axes[2]]},
                1: {"ymin": float("inf"), "ymax": float("-inf"), "axes": [axes[3], axes[4]]},
            }
        }

    A = main_plot_all_in_single_figure._accum
    if A["index"] >= 5:
        A["fig"].savefig("plots/All_Environments.pdf", bbox_inches='tight', dpi=300)
        plt.close(A["fig"])
        delattr(main_plot_all_in_single_figure, "_accum")
        return

    # Plot this environment into the next slot
    ax = A["axes"][A["index"]]
    env_name = title
    run_dirs = sorted([d for d in os.listdir(data_dir) if d != '.DS_Store'])
    row_id = 0 if A["index"] < 3 else 1

    for run_dir in run_dirs:
        label, color, active = resolve_label_color_active(run_dir, label_and_color_map)
        if not active:
            continue
        steps, avg, se = load_and_compute_curve(data_dir, run_dir, int_space, total_steps)
        ax.fill_between(steps, avg - se, avg + se, color=f"tab:{color}", alpha=0.35)
        line_handle, = ax.plot(steps, avg, linewidth=2, color=color, label=label)
        if label not in A["legend_labels"]:
            A["legend_handles"].append(line_handle); A["legend_labels"].append(label)

        # Update row-wise min/max
        y_low = float((avg - se).min()); y_high = float((avg + se).max())
        A["row_stats"][row_id]["ymin"] = min(A["row_stats"][row_id]["ymin"], y_low)
        A["row_stats"][row_id]["ymax"] = max(A["row_stats"][row_id]["ymax"], y_high)

    ax.set_title(env_name, fontsize=11)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    A["index"] += 1

    # Normalize y-limits when a row completes
    if A["index"] in (3, 5):
        rs = A["row_stats"][row_id]
        if rs["ymin"] < rs["ymax"]:
            pad = 0.02 * (rs["ymax"] - rs["ymin"])
            for ax_row in rs["axes"]:
                ax_row.set_ylim(rs["ymin"] - pad, rs["ymax"] + pad)

    # Finalize after the 5th environment
    if A["index"] == 5:
        A["fig"].supxlabel("Time Step")
        A["fig"].supylabel("Average Episodic Return")
        A["fig"].suptitle("All Environments", y=1.02)
        if A["legend_handles"]:
            A["legend_ax"].legend(A["legend_handles"], A["legend_labels"], loc='center', frameon=False)
        A["fig"].savefig("plots/All_Environments.pdf", bbox_inches='tight', dpi=300)
        plt.close(A["fig"])
        delattr(main_plot_all_in_single_figure, "_accum")






def main_old(data_dir, title,  int_space, total_steps, label_and_color_map):
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
    for env_name in ['Ant', 'HalfCheetah', 'Humanoid', 'Hopper', 'Walker']:
        print(env_name)
        #main_old(data_dir = os.path.join(args.mother_dir,env_name), title=env_name, int_space=args.int_space, total_steps=args.total_steps, label_and_color_map=label_and_color_map)
        #main_plot_in_separate_figures(data_dir = os.path.join(args.mother_dir,env_name), title=env_name, int_space=args.int_space, total_steps=args.total_steps, label_and_color_map=label_and_color_map)
        main_plot_all_in_single_figure(data_dir = os.path.join(args.mother_dir,env_name), title=env_name, int_space=args.int_space, total_steps=args.total_steps, label_and_color_map=label_and_color_map)