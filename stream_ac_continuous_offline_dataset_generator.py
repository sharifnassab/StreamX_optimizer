import time
import os
import pickle  # no longer used for saving, but kept in case you still need it elsewhere
import argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal
from functools import partial

# --- Optimizers (Kept for agent training during data collection) ---
from optim import ObGD_sq as ObGDsq_Optimizer
from optim import ObGD_sq_plain as ObGDsqPlain_Optimizer
from optim import ObGD as ObGD_Optimizer
from optim import AdaptiveObGD as AdaptiveObGD_Optimizer
from optim import Obn as Obn_Optimizer
from optim import ObnC as ObnC_Optimizer
from optim import ObnN as ObnN_Optimizer
from optim import ObtC as ObtC_Optimizer
from optim import ObtCm as ObtCm_Optimizer
from optim import ObtN as ObtN_Optimizer
from optim import Obt as Obt_Optimizer
from optim import Obtnnz as Obtnnz_Optimizer
from optim import ObGDN as ObGDN_Optimizer
from optim import ObGDm as ObGDm_Optimizer

# --- Environment Wrappers ---
from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init

print('All imports done.')


def initialize_weights(m, sparsity=0.9):
    """Initializes network weights with sparsity."""
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=sparsity)
        m.bias.data.fill_(0.0)


class Actor(nn.Module):
    """Policy Network."""
    def __init__(self, n_obs=11, n_actions=3, hidden_depth=2, hidden_width=128, initialization_sparsity=0.9):
        super(Actor, self).__init__()
        self.fc_layer = nn.Linear(n_obs, hidden_width)
        # Register hidden layers properly so they are part of the module
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_width, hidden_width) for _ in range(hidden_depth - 1)]
        )
        self.linear_mu = nn.Linear(hidden_width, n_actions)
        self.linear_std = nn.Linear(hidden_width, n_actions)
        self.apply(partial(initialize_weights, sparsity=initialization_sparsity))

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = F.layer_norm(x, x.size())
            x = F.leaky_relu(x)
        mu = self.linear_mu(x)
        pre_std = self.linear_std(x)
        std = F.softplus(pre_std)  # Ensure std is positive
        return mu, std


class Critic(nn.Module):
    """Value Network."""
    def __init__(self, n_obs=11, hidden_depth=2, hidden_width=128, initialization_sparsity=0.9):
        super(Critic, self).__init__()
        self.fc_layer = nn.Linear(n_obs, hidden_width)
        # Register hidden layers properly
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_width, hidden_width) for _ in range(hidden_depth - 1)]
        )
        self.linear_layer = nn.Linear(hidden_width, 1)
        self.apply(partial(initialize_weights, sparsity=initialization_sparsity))

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = F.layer_norm(x, x.size())
            x = F.leaky_relu(x)
        return self.linear_layer(x)


def spec_to_name(spec: dict) -> str:
    """Converts a spec dictionary to a short string name."""
    short_hand_mapping = {
        'optimizer': '',
        'hidden_depth': '_net',
        'hidden_width': 'x',
        'initialization_sparsity': 'sp',
        'lamda': '_lam',
        'kappa': 'k',
        'gamma': 'gam',
        'entrywise_normalization': 'en',
        'beta2': 'b2',
        'u_trace': 'u',
        'entropy_coeff': 'ent',
        'momentum': 'm',
        'lr': 'lr',
        'weight_decay': 'wd',
        'delta_trace': 'delTr',
        'sig_power': 'sigP',
        'in_trace_sample_scaling': 'itss',
        'delta_clip': 'delClip',
        'delta_norm': 'delNorm',
    }
    list_params = []
    for key in short_hand_mapping:
        if key in spec:
            list_params.append(f"{short_hand_mapping[key]}_{spec[key]}")
    return '_'.join(list_params) if list_params else ''


def save_dataset_bin_npz(
    save_dir: str,
    env_name: str,
    seed: int,
    run_name: str,
    bin_index: int,
    states,
    actions,
    rewards_scaled,
    rewards_actual,
    next_states,
    dones,
):
    """
    Save one bin of data as a compressed NumPy .npz file in float32/bool.
    """
    env_short = env_name.split('-')[0]
    save_dir = os.path.join(save_dir, f'seed{seed}', f'{env_short}')
    if save_dir == 'none':
        print("save_dir is 'none', skipping save.")
        return

    if len(states) == 0:
        return

    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filename = f"bin{bin_index:05d}.npz"
        save_path = os.path.join(save_dir, filename)

        # Convert to compact dtypes
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards_scaled = np.asarray(rewards_scaled, dtype=np.float32)
        rewards_actual = np.asarray(rewards_actual, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.bool_)

        np.savez_compressed(
            save_path,
            states=states,
            actions=actions,
            rewards_scaled=rewards_scaled,
            rewards_actual=rewards_actual,
            next_states=next_states,
            dones=dones,
            env_name=np.array(env_name),
            seed=np.array(seed, dtype=np.int32),
            run_name=np.array(run_name),
        )
        print(
            f"Saved bin {bin_index} with {states.shape[0]} transitions to: {save_path}"
        )
    except Exception as e:
        print(f"Error saving bin {bin_index}: {e}")


class StreamAC(nn.Module):
    """Streaming Actor-Critic Agent."""
    def __init__(
        self,
        n_obs: int,
        n_actions: int,
        policy_spec: dict,
        critic_spec: dict,
    ):
        super().__init__()

        # Keep the raw specs
        self.policy_spec = dict(policy_spec or {})
        self.critic_spec = dict(critic_spec or {})

        # ---- Hyperparameters ----
        self.gamma_policy = float(self.policy_spec.get('gamma'))
        self.gamma_critic = float(self.critic_spec.get('gamma'))
        self.entropy_coeff = float(self.policy_spec.get('entropy_coeff'))

        # ---- Nets ----
        self.policy_net = Actor(
            n_obs=n_obs,
            n_actions=n_actions,
            hidden_depth=self.policy_spec['hidden_depth'],
            hidden_width=self.policy_spec['hidden_width'],
            initialization_sparsity=self.policy_spec['initialization_sparsity'],
        )
        self.critic_net = Critic(
            n_obs=n_obs,
            hidden_depth=self.critic_spec['hidden_depth'],
            hidden_width=self.critic_spec['hidden_width'],
            initialization_sparsity=self.critic_spec['initialization_sparsity'],
        )

        # ---- Optimizers (per-spec) ----
        self.optimizer_policy = self._build_optimizer(
            self.policy_net.parameters(), self.policy_spec, role="policy"
        )
        self.optimizer_critic = self._build_optimizer(
            self.critic_net.parameters(), self.critic_spec, role="critic"
        )

    def _build_optimizer(self, params, spec: dict, role: str):
        spec = spec or {}
        opt_name = spec.get('optimizer', 'none').strip().lower()
        if opt_name in ['none', 'monte_carlo']:
            return None

        lr = float(spec.get('lr', 3e-4))
        gamma = float(spec.get('gamma'))
        lamda = float(spec.get('lamda'))
        kappa = float(spec.get('kappa'))
        weight_decay = float(spec.get('weight_decay', 0.0))
        momentum = float(spec.get('momentum', 0.0))
        u_trace = float(spec.get('u_trace', 0.01))
        entrywise_normalization = spec.get('entrywise_normalization', 'none')
        beta2 = float(spec.get('beta2', 0.999))
        delta_trace = float(spec.get('delta_trace', 0.01))
        sig_power = float(spec.get('sig_power', 2))
        in_trace_sample_scaling = spec.get('delta_trace', False)
        delta_clip = spec.get('delta_clip', 'none')
        delta_norm = spec.get('delta_norm', 'none')

        if opt_name == 'obgd':
            return ObGD_Optimizer(params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        if opt_name in ('adaptiveobgd', 'adaptive_obgd'):
            return AdaptiveObGD_Optimizer(params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        if opt_name == 'obgd_sq':
            return ObGDsq_Optimizer(params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        if opt_name == 'obgd_sq_plain':
            return ObGDsqPlain_Optimizer(params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        if opt_name == 'obn':
            return Obn_Optimizer(
                params,
                lr=lr,
                gamma=gamma,
                lamda=lamda,
                kappa=kappa,
                u_trace=u_trace,
                entrywise_normalization=entrywise_normalization,
                beta2=beta2,
            )
        if opt_name == 'obnc':
            return ObnC_Optimizer(
                params,
                lr=lr,
                gamma=gamma,
                lamda=lamda,
                kappa=kappa,
                u_trace=u_trace,
                entrywise_normalization=entrywise_normalization,
                beta2=beta2,
            )
        if opt_name == 'obnn':
            return ObnN_Optimizer(
                params,
                lr=lr,
                gamma=gamma,
                lamda=lamda,
                kappa=kappa,
                delta_trace=delta_trace,
                u_trace=u_trace,
                entrywise_normalization=entrywise_normalization,
                beta2=beta2,
            )
        if opt_name == 'obtc':
            return ObtC_Optimizer(
                params,
                gamma=gamma,
                lamda=lamda,
                kappa=kappa,
                weight_decay=weight_decay,
                sig_power=sig_power,
                entrywise_normalization=entrywise_normalization,
                beta2=beta2,
                in_trace_sample_scaling=in_trace_sample_scaling,
            )
        if opt_name == 'obtn':
            return ObtN_Optimizer(
                params,
                gamma=gamma,
                lamda=lamda,
                kappa=kappa,
                weight_decay=weight_decay,
                sig_power=sig_power,
                delta_trace=delta_trace,
                entrywise_normalization=entrywise_normalization,
                beta2=beta2,
                in_trace_sample_scaling=in_trace_sample_scaling,
            )
        if opt_name == 'obt':
            return Obt_Optimizer(
                params,
                gamma=gamma,
                lamda=lamda,
                kappa=kappa,
                weight_decay=weight_decay,
                sig_power=sig_power,
                delta_clip=delta_clip,
                delta_norm=delta_norm,
                entrywise_normalization=entrywise_normalization,
                beta2=beta2,
                in_trace_sample_scaling=in_trace_sample_scaling,
            )
        if opt_name == 'obtnnz':
            return Obtnnz_Optimizer(
                params,
                gamma=gamma,
                lamda=lamda,
                kappa=kappa,
                weight_decay=weight_decay,
                delta_clip=delta_clip,
                delta_norm=delta_norm,
                u_trace=u_trace,
                entrywise_normalization=entrywise_normalization,
                beta2=beta2,
            )
        if opt_name == 'obgdn':
            return ObGDN_Optimizer(
                params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm
            )
        if opt_name == 'obgdm':
            return ObGDm_Optimizer(
                params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, momentum=momentum
            )
        if opt_name == 'obtcm':
            return ObtCm_Optimizer(
                params,
                gamma=gamma,
                lamda=lamda,
                kappa=kappa,
                weight_decay=weight_decay,
                sig_power=sig_power,
                momentum=momentum,
                entrywise_normalization=entrywise_normalization,
                beta2=beta2,
                in_trace_sample_scaling=in_trace_sample_scaling,
            )

        raise ValueError(f"Unknown optimizer '{spec.get('optimizer')}' for role '{role}'.")

    def pi(self, x):
        return self.policy_net(x)

    def v(self, x):
        return self.critic_net(x)

    def sample_action(self, s):
        """Samples an action from the policy given a state."""
        # Ensure state is a numpy array before converting
        if not isinstance(s, np.ndarray):
            s = np.array(s)
        x = torch.from_numpy(s).float()
        with torch.no_grad():  # No need to track gradients for action sampling
            mu, std = self.pi(x)
        dist = Normal(mu, std)
        return dist.sample().detach().numpy()

    def update_params(self, s, a, r, s_prime, done):
        """Performs a single training update."""
        done_mask = 0.0 if done else 1.0

        # Convert inputs to tensors
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(np.array(a), dtype=torch.float32)
        r = torch.tensor(np.array(r), dtype=torch.float32)
        s_prime = torch.tensor(np.array(s_prime), dtype=torch.float32)
        done_mask_t = torch.tensor(done_mask, dtype=torch.float32)

        v_s = self.v(s)
        with torch.no_grad():  # v_prime is part of the target, no gradient
            v_prime = self.v(s_prime)

        # Calculate TD errors
        td_target_critic = r + self.gamma_critic * v_prime * done_mask_t
        td_target_policy = r + self.gamma_policy * v_prime * done_mask_t
        delta_critic = (td_target_critic - v_s).item()  # Use .item() for scalar deltas
        delta_policy = (td_target_policy - v_s).item()

        mu, std = self.pi(s)
        dist = Normal(mu, std)

        log_prob_pi = dist.log_prob(a).sum()  # Sign flip: make it positive for gradient ascent
        entropy_pi = dist.entropy().sum()

        # Critic update
        self.optimizer_critic.zero_grad()
        (-v_s).backward()  # Gradient for v_s is -1 * dV/dw
        info_critic = self.optimizer_critic.step(delta_critic, reset=done)

        # Policy update
        self.optimizer_policy.zero_grad()
        loss_policy = -log_prob_pi - self.entropy_coeff * entropy_pi
        loss_policy.backward()
        info_policy = self.optimizer_policy.step(delta_policy, reset=done)

        return {'policy': info_policy, 'critic': info_critic}


def main(env_name, seed, total_steps, max_time, policy_spec, critic_spec, save_dir, run_name_suffix, bin_length):
    """Main data generation loop with float32 + binned .npz saving."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Environment Setup ---
    env = gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)  # For `info['reward_immediate']`
    env = gym.wrappers.ClipAction(env)

    gamma_env = critic_spec.get('gamma', policy_spec.get('gamma'))
    env = ScaleReward(env, gamma=gamma_env)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)

    # --- Agent Setup ---
    agent = StreamAC(
        n_obs=env.observation_space.shape[0],
        n_actions=env.action_space.shape[0],
        policy_spec=policy_spec,
        critic_spec=critic_spec,
    )

    # ---- Generate Run Name ----
    run_name = (
        f'{env_name}____-Policy_{spec_to_name(policy_spec)}'
        f'____-Critic_{spec_to_name(critic_spec)}'
        + (f"____{run_name_suffix}" if (run_name_suffix != '') else '')
    )
    print(f"Run Name: {run_name}")

    print(f"--- Starting Data Generation ---")
    print(f"Environment: {env_name}")
    print(f"Seed: {seed}")
    print(f"Total Steps (budget): {total_steps}")
    print(f"Saving to Dir: {save_dir}")
    print(f"Bin length: {bin_length}")
    print(f"----------------------------------")

    # --- Time bookkeeping ---
    start_time = time.time()
    max_time_in_seconds = sum(
        int(x) * 60 ** i for i, x in enumerate(reversed(max_time.split(":")))
    )

    # --- Per-field buffers for current bin ---
    states = []
    actions = []
    rewards_scaled = []
    rewards_actual = []
    next_states = []
    dones = []

    bin_index = 0
    total_transitions = 0

    s, _ = env.reset(seed=seed)

    max_epoch_time = 0.0
    epoch_start_time = time.time()

    for t in range(1, int(total_steps) + 1):
        a = agent.sample_action(s)
        s_prime, r_scaled, terminated, truncated, info = env.step(a)

        r_actual = info.get('reward_immediate', r_scaled)
        done = terminated or truncated

        # Store transition in current bin buffer
        states.append(s)
        actions.append(a)
        rewards_scaled.append(r_scaled)
        rewards_actual.append(r_actual)
        next_states.append(s_prime)
        dones.append(done)

        # Update agent with scaled reward
        agent.update_params(s, a, r_scaled, s_prime, done)

        s = s_prime

        # Flush bin if full
        if len(states) >= bin_length:
            save_dataset_bin_npz(
                save_dir=save_dir,
                env_name=env_name,
                seed=seed,
                run_name=run_name,
                bin_index=bin_index,
                states=states,
                actions=actions,
                rewards_scaled=rewards_scaled,
                rewards_actual=rewards_actual,
                next_states=next_states,
                dones=dones,
            )
            total_transitions += len(states)
            bin_index += 1
            states, actions, rewards_scaled, rewards_actual, next_states, dones = [], [], [], [], [], []

        if done:
            ep_return = info["episode"]["r"]
            print(f"Step: {t}, Episode Return: {ep_return:.2f}")

            # --- Time guard ---
            current_run_time = time.time() - start_time
            max_epoch_time = max(time.time() - epoch_start_time, max_epoch_time)
            epoch_start_time = time.time()
            if current_run_time > max_time_in_seconds - 30 - 1.2 * max_epoch_time:
                print("Terminating early due to time constraint...")
                break

            s, _ = env.reset()

    # Flush any remaining transitions in the last (possibly partial) bin
    if len(states) > 0:
        save_dataset_bin_npz(
            save_dir=save_dir,
            env_name=env_name,
            seed=seed,
            run_name=run_name,
            bin_index=bin_index,
            states=states,
            actions=actions,
            rewards_scaled=rewards_scaled,
            rewards_actual=rewards_actual,
            next_states=next_states,
            dones=dones,
        )
        total_transitions += len(states)
        bin_index += 1

    total_runtime = int(time.time() - start_time)
    print(f"\nTotal time: {time.gmtime(total_runtime)}")
    print(f"Total transitions saved: {total_transitions} across {bin_index} bins.")
    env.close()


if __name__ == '__main__':
    optimizer_choices = [
        'ObGD', 'ObGD_sq', 'ObGD_sq_plain', 'Obn', 'ObnC', 'ObnN', 'AdaptiveObGD',
        'ObtC', 'ObtN', 'Obt', 'Obtnnz', 'ObGDN', 'ObGDm', 'ObtCm'
    ]

    parser = argparse.ArgumentParser(description='Offline Data Generator using StreamAC')

    # --- Core Arguments ---
    parser.add_argument('--env_name', type=str, default='Ant-v5')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total_steps', type=int, default=100_000)
    parser.add_argument('--max_time', type=str, default='1000:00:00')
    parser.add_argument('--save_dir', type=str, default='none', help='/Users/arsalan/Desktop/Codes/StreamX/offline_dataset/ObGD_ObGD_lam_0.8/')
    parser.add_argument('--run_name', type=str, default='', help='Suffix for run name and save file name.')
    parser.add_argument('--bin_length', type=int, default=30_000,help='Number of transitions per saved dataset bin.')

    # --- Policy Arguments ---
    parser.add_argument('--policy_hidden_depth', type=int, default=2)
    parser.add_argument('--policy_hidden_width', type=int, default=128)
    parser.add_argument('--policy_initialization_sparsity', type=float, default=0.9)
    parser.add_argument('--policy_optimizer', type=str, default='ObGD', choices=optimizer_choices)
    parser.add_argument('--policy_kappa', type=float, default=3.0)
    parser.add_argument('--policy_gamma', type=float, default=0.99)
    parser.add_argument('--policy_lamda', type=float, default=0.0)
    parser.add_argument('--policy_lr', type=float, default=1.0)
    parser.add_argument('--policy_momentum', type=float, default=0.0)
    parser.add_argument('--policy_entropy_coeff', type=float, default=0.01)
    parser.add_argument('--policy_u_trace', type=float, default=0.01)
    parser.add_argument('--policy_entrywise_normalization', type=str, default='RMSProp')
    parser.add_argument('--policy_beta2', type=float, default=0.999)
    parser.add_argument('--policy_delta_trace', type=float, default=0.01)
    parser.add_argument('--policy_weight_decay', type=float, default=0.0)
    parser.add_argument('--policy_in_trace_sample_scaling', type=str, default='True',
                        choices=['True', 'False'])
    parser.add_argument('--policy_sig_power', type=float, default=2)
    parser.add_argument('--policy_delta_clip', type=str, default='none')
    parser.add_argument('--policy_delta_norm', type=str, default='none')

    # --- Critic Arguments ---
    parser.add_argument('--critic_hidden_depth', type=int, default=2)
    parser.add_argument('--critic_hidden_width', type=int, default=128)
    parser.add_argument('--critic_initialization_sparsity', type=float, default=0.9)
    parser.add_argument('--critic_optimizer', type=str, default='ObnC', choices=optimizer_choices)
    parser.add_argument('--critic_kappa', type=float, default=2.0)
    parser.add_argument('--critic_gamma', type=float, default=0.99)
    parser.add_argument('--critic_lamda', type=float, default=0.0)
    parser.add_argument('--critic_lr', type=float, default=1.0)
    parser.add_argument('--critic_momentum', type=float, default=0.0)
    parser.add_argument('--critic_u_trace', type=float, default=0.01)
    parser.add_argument('--critic_entrywise_normalization', type=str, default='RMSProp')
    parser.add_argument('--critic_beta2', type=float, default=0.999)
    parser.add_argument('--critic_delta_trace', type=float, default=0.01)
    parser.add_argument('--critic_weight_decay', type=float, default=0.0)
    parser.add_argument('--critic_in_trace_sample_scaling', type=str, default='True',
                        choices=['True', 'False'])
    parser.add_argument('--critic_sig_power', type=float, default=2)
    parser.add_argument('--critic_delta_clip', type=str, default='none')
    parser.add_argument('--critic_delta_norm', type=str, default='none')

    args = parser.parse_args()

    # ---- Define Optimizer Params ----
    shared_params = [
        'optimizer', 'kappa', 'gamma', 'lamda', 'weight_decay',
        'hidden_depth', 'hidden_width', 'initialization_sparsity'
    ]
    required_optimizer_params = {
        'none': [],
        'monte_carlo': ['gamma'],
        'ObGD': shared_params + ['lr'],
        'ObGD_sq': shared_params + ['lr'],
        'ObGD_sq_plain': shared_params + ['lr'],
        'AdaptiveObGD': shared_params + ['lr'],
        'Obn': shared_params + ['lr', 'entrywise_normalization', 'beta2', 'u_trace'],
        'ObnC': shared_params + ['lr', 'entrywise_normalization', 'beta2', 'u_trace'],
        'ObnN': shared_params + ['lr', 'entrywise_normalization', 'beta2', 'u_trace', 'delta_trace'],
        'ObtC': shared_params + ['entrywise_normalization', 'beta2', 'sig_power', 'in_trace_sample_scaling'],
        'ObtCm': shared_params + ['entrywise_normalization', 'beta2', 'sig_power',
                                  'in_trace_sample_scaling', 'momentum'],
        'ObtN': shared_params + ['entrywise_normalization', 'beta2', 'sig_power',
                                 'in_trace_sample_scaling', 'delta_trace'],
        'Obt': shared_params + ['entrywise_normalization', 'beta2', 'sig_power',
                                'in_trace_sample_scaling', 'delta_clip', 'delta_norm'],
        'Obtnnz': shared_params + ['entrywise_normalization', 'beta2', 'u_trace',
                                   'delta_clip', 'delta_norm'],
        'ObGDN': shared_params + ['lr', 'delta_clip', 'delta_norm'],
        'ObGDm': shared_params + ['lr', 'momentum'],
    }

    def build_spec(kind, args, required_optimizer_params) -> dict:
        opt = getattr(args, f'{kind}_optimizer')
        required_params = required_optimizer_params.get(opt, [])
        spec = {'optimizer': opt}
        for key in required_params:
            if hasattr(args, f'{kind}_{key}'):
                spec[key] = getattr(args, f'{kind}_{key}')
        return spec

    # ---- Build Spec Dicts ----
    policy_spec = build_spec(kind='policy', args=args, required_optimizer_params=required_optimizer_params)
    policy_spec.update({'entropy_coeff': args.policy_entropy_coeff})
    critic_spec = build_spec(kind='critic', args=args, required_optimizer_params=required_optimizer_params)

    # ---- Run Main ---
    main(
        args.env_name,
        args.seed,
        args.total_steps,
        args.max_time,
        policy_spec=policy_spec,
        critic_spec=critic_spec,
        save_dir=args.save_dir,
        run_name_suffix=args.run_name,
        bin_length=args.bin_length,
    )

