import os, pickle, argparse
os.environ["GYM_DISABLE_PLUGIN_AUTOLOAD"] = "1"
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal
from optim import ObGD as Optimizer
from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init

# NEW: our tiny logger (wandb or tensorboard)
from logger import get_logger

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
        super(Actor, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        mu = self.linear_mu(x)
        pre_std = self.linear_std(x)
        std = F.softplus(pre_std)
        return mu, std

class Critic(nn.Module):
    def __init__(self, n_obs=11, hidden_size=128):
        super(Critic, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer  = nn.Linear(hidden_size, hidden_size)
        self.linear_layer  = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)      
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        return self.linear_layer(x)

class StreamAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.value_net = Critic(n_obs=n_obs, hidden_size=hidden_size)
        self.optimizer_policy = Optimizer(self.policy_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_value = Optimizer(self.value_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def pi(self, x):
        return self.policy_net(x)

    def v(self, x):
        return self.value_net(x)

    def sample_action(self, s):
        x = torch.from_numpy(s).float()
        mu, std = self.pi(x)
        dist = Normal(mu, std)
        return dist.sample().numpy()

    # CHANGED: now returns an info dict with metrics (incl. TD error)
    def update_params(self, s, a, r, s_prime, done, entropy_coeff, overshooting_info=False):
        done_mask = 0 if done else 1
        s = torch.tensor(np.array(s), dtype=torch.float)
        a = torch.tensor(np.array(a), dtype=torch.float)
        r = torch.tensor(np.array(r), dtype=torch.float)
        s_prime = torch.tensor(np.array(s_prime), dtype=torch.float)
        done_mask_t = torch.tensor(np.array(done_mask), dtype=torch.float)

        v_s = self.v(s)
        v_prime = self.v(s_prime)
        td_target = r + self.gamma * v_prime * done_mask_t
        delta = td_target - v_s  # TD error

        mu, std = self.pi(s)
        dist = Normal(mu, std)

        # Losses
        log_prob_pi = -(dist.log_prob(a)).sum()
        value_output = -v_s
        entropy = dist.entropy().sum()
        entropy_pi = -entropy_coeff * entropy * torch.sign(delta).item()

        # Backprop
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()

        # Optimizer steps (capture optional info from ObGD)
        pol_info = self.optimizer_policy.step(delta.item(), reset=done)
        val_info = self.optimizer_value.step(delta.item(), reset=done)

        # Optional overshooting check
        if overshooting_info:
            v_s2 = self.v(s)
            v_prime2 = self.v(s_prime)
            td_target2 = r + self.gamma * v_prime2 * done_mask_t
            delta_bar = td_target2 - v_s2
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")

        # Collect metrics
        info = {
            "train/td_error": float(delta.item()),
            "train/value_v": float(v_s.item()),
            "train/policy_entropy": float(entropy.item()),
            "train/policy_log_prob": float((-log_prob_pi).item()),  # log π(a|s)
            "train/std_mean": float(std.mean().item()),
            "train/mu_norm": float(mu.norm().item()),
            "train/reward": float(r.item()),
        }
        if val_info is not None:
            for key in val_info:
                info[f'optimizer_val/{key}'] = val_info[key]
        if pol_info is not None:
            for key in pol_info:
                info[f'optimizer_pol/{key}'] = pol_info[key]

        return info

def main(env_name, seed, lr, gamma, lamda, total_steps, entropy_coeff, kappa_policy, kappa_value, debug, overshooting_info, render=False,
         log_backend="tensorboard", log_dir="runs", project="rl", run_name=None):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)
    agent = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], lr=lr, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))

    # NEW: init logger (TB or WandB)
    config = {
        "env_name": env_name, "seed": seed, "lr": lr, "gamma": gamma, "lamda": lamda,
        "total_steps": total_steps, "entropy_coeff": entropy_coeff,
        "kappa_policy": kappa_policy, "kappa_value": kappa_value,
    }
    logger = get_logger(
        backend=log_backend,
        log_dir=log_dir,
        project=project,
        run_name=run_name or f"{env.spec.id}_lr_{lr}_lambda_{lamda}_seed{seed}",
        config=config
    )
    # Optional: for wandb, watch models' grads
    logger.watch([agent.policy_net, agent.value_net], log="all")  # no-op on TB

    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)

    # Episode accumulators for ObGD stats
    ep_steps = 0
    ep_alpha_clipped_count = 0
    ep_min_inv_M_sum = 0.0

    for t in range(1, total_steps+1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)

        # Update and collect per-step metrics
        step_info = agent.update_params(s, a, r, s_prime,  terminated or truncated, entropy_coeff, overshooting_info)

        # Per-time-step logging (x-axis is the global time step)
        logger.log(step_info, step=t)

        # Accumulate episode-level ObGD stats if available
        ep_steps += 1
        if bool(step_info.get("policy_optimizer/alpha_clipped", False)):
            ep_alpha_clipped_count += 1
        if "policy_optimizer/min_inv_M" in step_info:
            ep_min_inv_M_sum += float(step_info["obgd/min_inv_M"])

        s = s_prime

        if terminated or truncated:
            ep_return = info["episode"]["r"]
            ep_len = info["episode"].get("l", ep_steps)

            # Compute requested episode-level metrics
            alpha_clip_pct = 100.0 * ep_alpha_clipped_count / max(ep_steps, 1)
            avg_min_inv_M = ep_min_inv_M_sum / max(ep_steps, 1) if ep_min_inv_M_sum > 0 else None

            # Log episode summaries at the terminating time step
            logger.log({
                "episode/return": float(ep_return),
                "episode/length": float(ep_len),
                "policy_optimizer/alpha_clipped_percent": float(alpha_clip_pct),
                "policy_optimizer/avg_min_inv_M": float(avg_min_inv_M) if avg_min_inv_M is not None else None,
            }, step=t)

            if debug:
                print(f"Episodic Return: {ep_return}, Time Step {t} | "
                      f"alpha_clipped%={alpha_clip_pct:.2f}, avg_min_inv_M={avg_min_inv_M}")

            returns.append(ep_return)
            term_time_steps.append(t)

            # Reset episode accumulators
            ep_steps = 0
            ep_alpha_clipped_count = 0
            ep_min_inv_M_sum = 0.0

            terminated, truncated = False, False
            s, _ = env.reset()

    env.close()
    logger.finish()  # NEW

    save_dir = "data_stream_ac_{}_lr{}_gamma{}_lamda{}_entropy_coeff{}".format(env.spec.id, lr, gamma, lamda, entropy_coeff)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')

    # NEW: logging choices/params
    parser.add_argument('--log_backend', type=str, default='wandb', choices=['tensorboard', 'wandb', 'none'])
    parser.add_argument('--log_dir', type=str, default='runs', help='TensorBoard log dir (if backend=tensorboard)')
    parser.add_argument('--project', type=str, default='StreamX', help='WandB project (if backend=wandb)')
    parser.add_argument('--run_name', type=str, default='', help='Run name for logger')

    args = parser.parse_args()
    
    main(args.env_name, args.seed, args.lr, args.gamma, args.lamda, args.total_steps, args.entropy_coeff,
         args.kappa_policy, args.kappa_value, args.debug, args.overshooting_info, args.render,
         log_backend=args.log_backend, log_dir=args.log_dir, project=f'{args.project}_{args.env_name}', run_name=f'{args.run_name}')
