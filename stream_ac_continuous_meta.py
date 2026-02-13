import time
start_time = time.time()
import os, pickle, argparse
os.environ["GYM_DISABLE_PLUGIN_AUTOLOAD"] = "1"
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal
from functools import partial
from copy import deepcopy

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
from optim import Obtm as Obtm_Optimizer
from optim import Obonz as Obonz_Optimizer
from optim import Obo as Obo_Optimizer
from optim import OboC as OboC_Optimizer
from optim import OboBase as OboBase_Optimizer
from optim import OboMetaOpt as OboMetaOpt_Optimizer
from optim import OboMetaZero as OboMetaZero_Optimizer
from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init

# NEW: our tiny logger (wandb or tensorboard)
from logger import get_logger
print('All imports done. 2')  

def initialize_weights(m, sparsity=0.9):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=sparsity)
        m.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_depth=2, hidden_width=128, initialization_sparsity=0.9):
        super(Actor, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_width)
        self.hidden_layers = [nn.Linear(hidden_width, hidden_width) for _ in range(hidden_depth-1)]
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
        std = F.softplus(pre_std)
        return mu, std

class Critic(nn.Module):
    def __init__(self, n_obs=11, hidden_depth=2, hidden_width=128, initialization_sparsity=0.9):
        super(Critic, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_width)
        self.hidden_layers = [nn.Linear(hidden_width, hidden_width) for _ in range(hidden_depth-1)]
        self.linear_layer  = nn.Linear(hidden_width, 1)
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
    

def compute_weight_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.requires_grad:  # only count learnable params
            total += torch.sum(p ** 2)
    return torch.sqrt(total)

def spec_to_name(spec: dict) -> str:
    short_hand_mapping={
                        'optimizer':    '',
                        'hidden_depth': '_net',
                        'hidden_width':  'x',
                        'initialization_sparsity': 'sp',
                        'lamda':        '_lam',
                        'kappa':        'k',
                        'gamma':        'gam',
                        'entrywise_normalization': 'en',
                        'beta2':        'b2',
                        'u_trace':      'u',
                        'entropy_coeff':'ent',
                        'momentum':     'm',
                        'lr':           'lr',
                        'weight_decay': 'wd',
                        'delta_trace':  'delTr',
                        'sig_power':    'sigP',
                        'in_trace_sample_scaling': 'itss',
                        'delta_clip':   'delClip',
                        'delta_norm':   'delNorm',
                        'meta_stepsize': '__-metaStep',
                        'beta2_meta':   'b2', 
                        'stepsize_parameterization':'', 
                        'h_decay_meta': 'hd', 
                        'clip_zeta_meta': 'clipz',
                        'epsilon_meta': 'eps',
                        'meta_loss_type': 'Loss',
                        'meta_shadow_dist_reg': 'ShadowDistReg',
                        }
    list_params = []
    for key in short_hand_mapping:
        if key in spec:
            list_params.append(f"{short_hand_mapping[key]}_{spec[key]}")
    return '_'.join(list_params) if list_params else ''




class StreamAC(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_actions: int,
        policy_spec: dict,
        critic_spec: dict,
        observer_spec: dict = {'optimizer':'none'},
    ):
        super().__init__()

        # Keep the raw specs (useful for logging/inspect)
        self.policy_spec   = dict(policy_spec or {})
        self.critic_spec   = dict(critic_spec or {})
        self.observer_spec = dict(observer_spec or {})

        # ---- Hyperparameters ----
        self.gamma_policy   = float(self.policy_spec.get('gamma'))
        self.gamma_critic   = float(self.critic_spec.get('gamma'))
        self.entropy_coeff  = float(self.policy_spec.get('entropy_coeff'))
        self.observer_exists = str(self.observer_spec.get('optimizer', 'none')).lower() not in ['none','monte_carlo']
        if self.observer_exists:
            self.gamma_observer = float(self.observer_spec.get('gamma'))    
        
        # ---- Nets ----
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_depth=self.policy_spec['hidden_depth'],    hidden_width=self.policy_spec['hidden_width'],   initialization_sparsity=self.policy_spec['initialization_sparsity'])
        self.critic_net = Critic(n_obs=n_obs,                     hidden_depth=self.critic_spec['hidden_depth'],    hidden_width=self.critic_spec['hidden_width'],   initialization_sparsity=self.critic_spec['initialization_sparsity'])
        if self.observer_exists:
            self.observer_net = Critic(n_obs=n_obs,               hidden_depth=self.observer_spec['hidden_depth'],  hidden_width=self.observer_spec['hidden_width'], initialization_sparsity=self.observer_spec['initialization_sparsity'])

        
        # ---- Optimizers (per-spec) ----
        self.optimizer_policy   = self._build_optimizer(self.policy_net,   self.policy_spec,   role="policy")
        self.optimizer_critic   = self._build_optimizer(self.critic_net,   self.critic_spec,   role="critic")
        self.optimizer_observer = (self._build_optimizer(self.observer_net, self.observer_spec, role="observer") if self.observer_exists else None)


    def _build_optimizer(self, network, spec: dict, role: str):
        params = network.parameters()
        spec = spec or {}
        opt_name = spec.get('optimizer', 'none').strip().lower()
        if opt_name in ['none','monte_carlo']:
            return None
        
        lr     = float(spec.get('lr', 3e-4))
        gamma  = float(spec.get('gamma'))
        lamda  = float(spec.get('lamda'))
        kappa  = float(spec.get('kappa'))
        weight_decay = float(spec.get('weight_decay', 0.0))
        momentum = float(spec.get('momentum', 0.0))
        entropy_coeff = float(spec.get('entropy_coeff', 0.01))

        # Only for Obn / Obt
        u_trace = float(spec.get('u_trace', 0.01))
        entrywise_normalization = spec.get('entrywise_normalization', 'none')
        beta2  = float(spec.get('beta2', 0.999))
        delta_trace = float(spec.get('delta_trace', 0.01))
        sig_power = float(spec.get('sig_power', 2))
        in_trace_sample_scaling = spec.get('delta_trace', False)
        delta_clip = spec.get('delta_clip', 'none')
        delta_norm = spec.get('delta_norm', 'none')

        # for meta-optimizers
        meta_stepsize = float(spec.get('meta_stepsize', 1e-3))
        beta2_meta    = float(spec.get('beta2_meta', 0.999))
        stepsize_parameterization= spec.get('stepsize_parameterization', 'sigmoid')
        h_decay_meta  = float(spec.get('h_decay_meta', 0.9999))
        clip_zeta_meta= spec.get('clip_zeta_meta', 'none')
        epsilon_meta = float(spec.get('epsilon_meta', 1e-3))
        meta_loss_type = spec.get('meta_loss_type')
        meta_shadow_dist_reg = float(spec.get('meta_shadow_dist_reg', 0.0))

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
                params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa,
                u_trace=u_trace, entrywise_normalization=entrywise_normalization, beta2=beta2
            )
        if opt_name == 'obnc':
            return ObnC_Optimizer(
                params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa,
                u_trace=u_trace, entrywise_normalization=entrywise_normalization, beta2=beta2
            )
        if opt_name == 'obnn':
            return ObnN_Optimizer(
                params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, delta_trace=delta_trace,
                u_trace=u_trace, entrywise_normalization=entrywise_normalization, beta2=beta2
            )
        if opt_name == 'obtc':
            return ObtC_Optimizer(
                params, gamma=gamma, lamda=lamda, kappa=kappa,  weight_decay=weight_decay, sig_power=sig_power,
                entrywise_normalization=entrywise_normalization, beta2=beta2, in_trace_sample_scaling=in_trace_sample_scaling
            )
        if opt_name == 'obtn':
            return ObtN_Optimizer(
                params, gamma=gamma, lamda=lamda, kappa=kappa,  weight_decay=weight_decay, sig_power=sig_power, delta_trace=delta_trace,
                entrywise_normalization=entrywise_normalization, beta2=beta2, in_trace_sample_scaling=in_trace_sample_scaling
            )
        
        if opt_name == 'obt':
            return Obt_Optimizer(
                params, gamma=gamma, lamda=lamda, kappa=kappa,  weight_decay=weight_decay, sig_power=sig_power, delta_clip=delta_clip,  delta_norm=delta_norm,
                entrywise_normalization=entrywise_normalization, beta2=beta2, in_trace_sample_scaling=in_trace_sample_scaling
            )

        if opt_name == 'obtnnz':
            return Obtnnz_Optimizer(
                params, gamma=gamma, lamda=lamda, kappa=kappa,  weight_decay=weight_decay, delta_clip=delta_clip,  delta_norm=delta_norm,
                u_trace=u_trace, entrywise_normalization=entrywise_normalization, beta2=beta2
            )
        
        if opt_name == 'obonz':
            return Obonz_Optimizer(
                params, gamma=gamma, lamda=lamda, kappa=kappa,  weight_decay=weight_decay, delta_clip=delta_clip,  delta_norm=delta_norm, momentum=momentum,
                u_trace=u_trace, entrywise_normalization=entrywise_normalization, beta2=beta2
            )

        if opt_name == 'obgdn':
            return ObGDN_Optimizer(
                params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip,  delta_norm=delta_norm)
        
        if opt_name == 'obgdm':
            return ObGDm_Optimizer(
                params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, momentum=momentum)
        
        if opt_name == 'obtcm':
            return ObtCm_Optimizer(
                params, gamma=gamma, lamda=lamda, kappa=kappa, weight_decay=weight_decay, sig_power=sig_power, momentum=momentum, 
                entrywise_normalization=entrywise_normalization, beta2=beta2, in_trace_sample_scaling=in_trace_sample_scaling
            )
        
        if opt_name == 'obtm':
            return Obtm_Optimizer(
                params, gamma=gamma, lamda=lamda, kappa=kappa,  weight_decay=weight_decay, sig_power=sig_power, delta_clip=delta_clip,  delta_norm=delta_norm, momentum=momentum,
                entrywise_normalization=entrywise_normalization, beta2=beta2, in_trace_sample_scaling=in_trace_sample_scaling
            )
        
        if opt_name == 'obo':
            return Obo_Optimizer(
                params, gamma=gamma, lamda=lamda, kappa=kappa,  weight_decay=weight_decay, sig_power=sig_power, delta_clip=delta_clip,  delta_norm=delta_norm, momentum=momentum,
                entrywise_normalization=entrywise_normalization, beta2=beta2, in_trace_sample_scaling=in_trace_sample_scaling
            )
        
        if opt_name == 'oboc':
            return OboC_Optimizer(
                params, gamma=gamma, lamda=lamda, kappa=kappa, weight_decay=weight_decay, sig_power=sig_power, momentum=momentum, 
                entrywise_normalization=entrywise_normalization, beta2=beta2, in_trace_sample_scaling=in_trace_sample_scaling
            )
        if opt_name == 'obobase':
            return OboBase_Optimizer(
                params, gamma=gamma, lamda=lamda, kappa=kappa,  weight_decay=weight_decay, delta_clip=delta_clip,  delta_norm=delta_norm, momentum=momentum,
                entrywise_normalization=entrywise_normalization, beta2=beta2
            )
        
        if opt_name == 'obometaopt':
            return OboMetaOpt_Optimizer(
                params, gamma=gamma, lamda=lamda, kappa=kappa,  weight_decay=weight_decay, delta_clip=delta_clip,  delta_norm=delta_norm, momentum=momentum,
                entrywise_normalization=entrywise_normalization, beta2=beta2, meta_stepsize=meta_stepsize, beta2_meta=beta2_meta, stepsize_parameterization=stepsize_parameterization, h_decay_meta=h_decay_meta
            )
        
        if opt_name == 'obometazero':
            return OboMetaZero_Optimizer(
                network, role, gamma=gamma, lamda=lamda, kappa=kappa, entropy_coeff=entropy_coeff, weight_decay=weight_decay, delta_clip=delta_clip,  delta_norm=delta_norm, momentum=momentum, entrywise_normalization=entrywise_normalization, beta2=beta2, 
                meta_stepsize=meta_stepsize, beta2_meta=beta2_meta, stepsize_parameterization=stepsize_parameterization, epsilon_meta=epsilon_meta, meta_loss_type=meta_loss_type, meta_shadow_dist_reg=meta_shadow_dist_reg
            )
        

        
        raise ValueError(f"Unknown optimizer '{spec.get('optimizer')}' for role '{role}'.")


    def pi(self, x):
        return self.policy_net(x)
    
    def pi_shadow(self, x):
        return self.policy_net_shadow(x)
    
    def v(self, x):
        return self.critic_net(x)
    
    def v_shadow(self, x):
        return self.critic_net_shadow(x)
    
    def v_observer(self, x):
        return self.observer_net(x)

    def sample_action(self, s):
        x = torch.from_numpy(s).float()
        mu, std = self.pi(x)
        dist = Normal(mu, std)
        return dist.sample().detach().numpy()

    def compute_grad_v(self,optimizer,v):
        value_params_grouped = [list(g["params"]) for g in optimizer.param_groups]
        value_params = [p for grp in value_params_grouped for p in grp]
        v_grads_flat = torch.autograd.grad(v.sum(), value_params)
        it = iter(v_grads_flat)
        return [[next(it) for _ in grp] for grp in value_params_grouped]

    def update_params(self, s, a, r, s_prime, terminated, truncated):
        terminated_mask = 0 if terminated else 1 
        done = 1 if terminated or truncated else 0
        s = torch.tensor(np.array(s), dtype=torch.float)
        a = torch.tensor(np.array(a), dtype=torch.float)
        r = torch.tensor(np.array(r), dtype=torch.float)
        s_prime = torch.tensor(np.array(s_prime), dtype=torch.float)
        terminated_mask_t = torch.tensor(np.array(terminated_mask), dtype=torch.float)

        critic_has_shadow = (hasattr(self.optimizer_critic, 'opt_type') and self.optimizer_critic.opt_type in ['OboMetaZero'])
        policy_has_shadow = (hasattr(self.optimizer_policy, 'opt_type') and self.optimizer_policy.opt_type in ['OboMetaZero'])
        
        
        with torch.no_grad():
            v_prime = self.v(s_prime)
        v_s = self.v(s)
        td_target_critic = r + self.gamma_critic * v_prime * terminated_mask_t
        td_target_policy = r + self.gamma_policy * v_prime * terminated_mask_t
        delta_critic = td_target_critic - v_s
        delta_policy = td_target_policy - v_s

        if critic_has_shadow:
            info_critic = self.optimizer_critic.step(s, a, r, s_prime, terminated_mask_t, v_s, v_prime, delta_critic.item(), reset=done)
        else:
            self.critic_net.zero_grad()
            v_s.backward()
            info_critic = self.optimizer_critic.step(delta_critic.item(), reset=done)
        

        if policy_has_shadow:
            info_policy = self.optimizer_policy.step(s, a, r, s_prime, terminated_mask_t, None, None, delta_policy.item(), reset=done)
        else:
            mu, std = self.pi(s)
            dist = Normal(mu, std)

            log_prob_pi = (dist.log_prob(a)).sum()
            entropy_pi = self.entropy_coeff * dist.entropy().sum() * torch.sign(delta_policy).item()
            self.policy_net.zero_grad()
            (log_prob_pi + entropy_pi).backward()
            info_policy = self.optimizer_policy.step(delta_policy.item(), reset=done)



        

        # observer update:
        if self.observer_exists: 
            self.optimizer_observer.zero_grad()
        info_observer = {}
        if self.observer_exists:
            v_s_obs, v_prime_obs = self.v_observer(s), self.v_observer(s_prime)
            (-v_s_obs).backward()
            delta_obs = r + self.gamma_observer * v_prime_obs * terminated_mask_t - v_s_obs
            info_observer = self.optimizer_observer.step(delta_obs.item(), reset=done)
            # For logging:
            info_observer = {**(info_observer or {}),
                "td_error": float(delta_obs.item()),
                "v(s)": float(v_s_obs.item()),
            }


        # # Logging:
        # info_policy = {**(info_policy or {}),
        #     "policy_log_prob": float((-log_prob_pi).item()),  # log π(a|s)
        #     "std_mean": float(std.mean().item()),
        #     "mu_norm": float(mu.norm().item()),
        #     "reward": float(r.item()),
        # }
        # info_critic = {**(info_critic or {}),
        #     "td_error": float(delta_critic.item()),
        #     "v(s)": float(v_s.item()),
        # }

        return  {'policy':info_policy, 'critic':info_critic, 'observer':info_observer}

def compute_monte_carlo_v_at_the_end_of_episode(list_ep_R, gamma):
    G = 0.0
    list_v = []
    for r in reversed(list_ep_R):
        G = r + gamma * G
        list_v.append(G+0.0)
    return [v for v in reversed(list_v)]


def compute_prediction_MSE(list_ep_R, list_ep_v, gamma):
    G = 0.0; ep_sse = 0.0; ep_abss = 0.0
    for v, r in zip(reversed(list_ep_v), reversed(list_ep_R)):
        G = r + gamma * G
        err = G - v
        ep_sse  += err * err
        ep_abss += abs(err)
    return {'ep_MSE_error':ep_sse,  'ep_abs_error':ep_abss}

def compute_prediction_MSE_end_of_episode_W(list_ep_R, list_ep_S, observer_net, gamma):
    list_ep_v = []
    with torch.no_grad():
        for s in list_ep_S:
            list_ep_v.append(observer_net(torch.tensor(np.array(s), dtype=torch.float)).item())
    return compute_prediction_MSE(list_ep_R, list_ep_v, gamma)


def main(env_name, seed, total_steps, max_time, policy_spec, critic_spec, observer_spec={'optimizer':'none'}, logging_spec={'backend':'none'}, debug=False, render=False):
    torch.manual_seed(seed); np.random.seed(seed)

    render = bool(logging_spec.get('render', False))
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)

    gamma_env = critic_spec.get('gamma', policy_spec.get('gamma'))
    env = ScaleReward(env, gamma=gamma_env)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)

    agent = StreamAC(
        n_obs=env.observation_space.shape[0],
        n_actions=env.action_space.shape[0],
        policy_spec=policy_spec,
        critic_spec=critic_spec,
        observer_spec=observer_spec,
    )

    # ---- Logging ----
    run_name = (f'{env.spec.id}__bs____-Policy_{spec_to_name(policy_spec)}____-Critic_{spec_to_name(critic_spec)}' +
                (f'____-Observer_{spec_to_name(observer_spec)}' if agent.observer_exists else '') + 
                (f"____{logging_spec.get('run_name', '')}" if (logging_spec.get('run_name', '')!='') else ''))
    config = {
        "env_name": env_name, 
        "seed": seed,
        "policy": policy_spec,
        "critic": critic_spec,
        "observer": observer_spec,
        "run_name": run_name,
    }
    print(f"Run Name: {run_name}")
    print(f"seed:{seed}")

    logger = get_logger(
        backend=logging_spec.get('backend', 'wandb'),
        log_dir=logging_spec.get('dir', 'runs'),
        project=logging_spec.get('project', 'StreamX_OptDesign'),
        run_name=run_name+f'__seed{seed}',
        config=config,
    )
    logging_level = logging_spec.get('level')
    if logging_level in ['heavy']:
        logger.watch([agent.policy_net, agent.critic_net], log="all")  # no-op for TB

    if debug:
        print(f"seed: {seed} env: {env.spec.id}")

    # ---- Train ----
    start_time = time.time()
    max_time_in_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(max_time.split(":"))))

    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    list_ep_R, list_ep_v, list_ep_S, list_policy_delta_used = [], {'critic':[], 'observer':[]}, [], []
    ep_steps = 0
    ep_min_inv_M_sum = 0.0
    ep_policy_std = 0.0
    max_epoch_time = 0.0
    episode_number = 0
    epoch_start_time = time.time()
    

    for t in range(1, int(total_steps) + 1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)

        list_ep_R.append(r)
        list_ep_S.append(s)

        step_info = agent.update_params(s, a, r, s_prime, terminated, truncated,)
        s = s_prime

        for net in ['critic', 'observer']:
            list_ep_v[net].append(step_info[net].get('v(s)',0.0))
        if 'delta_used' in step_info['policy']:
            list_policy_delta_used.append(step_info['policy']['delta_used'])

        if (t % 500_000) <= 2025 and  logging_level in ['heavy']:
            expanded_step_info = {}
            for net_type in step_info:
                section = step_info.get(net_type) or {}
                for metric, val in section.items():
                    expanded_step_info[f'{net_type}/{metric}'] = val
            # expanded_step_info.update({
            #                         'rewards/original':info['reward_immediate'], 
            #                         'rewards/scaled':r,
            #                         'rewards/ratio': r/info['reward_immediate'],
            #                       })
            logger.log(expanded_step_info, step=t)


        ep_steps += 1
        ep_min_inv_M_sum += float(step_info.get('policy', {}).get('min_inv_M', 0.0))
        ep_policy_std += float(step_info.get('policy', {}).get('std_mean', 0.0))

        if terminated or truncated:
            #print(f"{max([abs(x) for x in list_policy_delta_used]):.2f}", '\t', f"{sum([abs(x) for x in list_policy_delta_used])/len(list_policy_delta_used):.2f}", '\t', f"{np.sqrt(sum([x**2 for x in list_policy_delta_used])/len(list_policy_delta_used)):.2f}")
            if agent.observer_exists:
                ep_pred_error = compute_prediction_MSE(list_ep_R, list_ep_v['observer'], agent.gamma_observer)
                ep_pred_error_end_of_episode_W = compute_prediction_MSE_end_of_episode_W(list_ep_R, list_ep_S, agent.observer_net, agent.gamma_observer)
            ep_pred_error_critic = compute_prediction_MSE(list_ep_R, list_ep_v['critic'], agent.gamma_critic)
            
            if observer_spec['optimizer'] == 'monte_carlo':
                monte_carlo_v = compute_monte_carlo_v_at_the_end_of_episode(list_ep_R, observer_spec['gamma'])
                if (t % 500_000) <= 3001: 
                    for ii in range(ep_steps):
                        logger.log({"observer/v(s)": monte_carlo_v[ii], "observer/delta":list_ep_R[ii], "observer/td_error":list_ep_R[ii]}, step=t-ep_steps+ii+1)

            ep_return = info["episode"]["r"]
            ep_len = info["episode"].get("l", ep_steps)
            # avg_min_inv_M = ep_min_inv_M_sum / max(ep_steps, 1) if ep_min_inv_M_sum > 0 else None
            avg_policy_std = ep_policy_std / max(ep_steps, 1) if ep_policy_std > 0 else None
            
            eta_policy = agent.optimizer_policy.eta.item() if hasattr(agent.optimizer_policy, 'eta') else None
            zeta_policy = agent.optimizer_policy.zeta.item() if hasattr(agent.optimizer_policy, 'zeta') else None
            eta_critic = agent.optimizer_critic.eta.item() if hasattr(agent.optimizer_critic, 'eta') else None
            zeta_critic = agent.optimizer_critic.zeta.item() if hasattr(agent.optimizer_critic, 'zeta') else None
            
            log_payload = {
                "_episode/return": float(ep_return),
                "_episode/length": float(ep_len),
                #"policy/avg_min_inv_M": float(avg_min_inv_M) if avg_min_inv_M is not None else None,
                "policy/avg_policy_std": float(avg_policy_std) if avg_policy_std is not None else None,
                #"critic_prediction/episode_MSE":  float(ep_pred_error_critic['ep_MSE_error']),
                #"critic_prediction/episode_abs":  float(ep_pred_error_critic['ep_abs_error']),
                "critic_prediction/episode_RMSE": float(np.sqrt(ep_pred_error_critic['ep_MSE_error'])),
            }

            for step_size_type, val in [('eta_policy', eta_policy), ('eta_critic', eta_critic), ('zeta_policy', zeta_policy),  ('zeta_critic', zeta_critic)]:
                if val is not None:
                    log_payload[f'step_size/{step_size_type}'] = val

            # if 'delta_used' in step_info['policy']:
            #     avg_abs_delta_used = sum([abs(x) for x in list_policy_delta_used])/len(list_policy_delta_used)
            #     avg_rms_delta_used = np.sqrt(sum([x**2 for x in list_policy_delta_used])/len(list_policy_delta_used))
            #     log_payload.update({
            #         'policy/delta_used_avg_abs':avg_abs_delta_used,
            #         'policy/delta_used_rms':avg_rms_delta_used,
            #     })
            if episode_number%40==0:
                log_payload.update({
                    "network/policy_w_norm": compute_weight_norm(agent.policy_net),
                    "network/critic_w_norm": compute_weight_norm(agent.critic_net),
                })
                if agent.observer_exists:
                    log_payload.update({"network/observer_w_norm": compute_weight_norm(agent.observer_net)}) 
                    
            if agent.observer_exists:
                log_payload.update({
                    #"observer_prediction/episode_MSE":  float(ep_pred_error['ep_MSE_error']),
                    "observer_prediction/episode_abs":  float(ep_pred_error['ep_abs_error']),
                    "observer_prediction/episode_RMSE": float(np.sqrt(ep_pred_error['ep_MSE_error'])),
                    #"observer_prediction/episode_MSE_end_of_episode_W":  float(ep_pred_error_end_of_episode_W['ep_MSE_error']),
                    "observer_prediction/episode_abs_end_of_episode_W":  float(ep_pred_error_end_of_episode_W['ep_abs_error']),
                    "observer_prediction/episode_RMSE_end_of_episode_W": float(np.sqrt(ep_pred_error_end_of_episode_W['ep_MSE_error'])),
                })
            logging_frequency = 10 if logging_level in ['light'] else 1 if logging_level in ['heavy'] else 1
            if episode_number%logging_frequency==0:
                logger.log(log_payload, step=t)


            if debug:
                print(f"Ep Return: {np.round(ep_return,1)},  \t eta_actor: {np.round(eta_policy,5)},   \t eta_critic: {np.round(eta_critic,5)},  \t Time Step {t}")
                

            # time guard
            max_epoch_time = max(time.time() - epoch_start_time, max_epoch_time)
            epoch_start_time = time.time()
            if time.time() - start_time > max_time_in_seconds - 30 - 1.2 * max_epoch_time:
                print("Terminating early due to time constraint...")
                break
            
            returns.append(ep_return)
            term_time_steps.append(t)

            # reset episode accumulators
            ep_steps = 0
            ep_min_inv_M_sum = 0.0
            ep_policy_std = 0.0
            s, _ = env.reset()
            list_ep_R, list_ep_v, list_ep_S, list_policy_delta_used = [], {'critic':[], 'observer':[]}, [], []
            episode_number+=1

    print(f"total time = {time.gmtime(int(time.time() - start_time))}")

    logger.log({'time/time': np.round((time.time() - start_time)/3600,2)}, step=t)

    env.close()
    logger.finish()

    if logging_spec['dir_pickle'] != 'none':
        name_of_file =(run_name if len(run_name)<255 else
                       logging_spec.get('run_name') if (logging_spec.get('run_name', '')!='') else 
                       run_name[:255])
        save_dir = os.path.join(logging_spec['dir_pickle'],  name_of_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
            pickle.dump((returns, term_time_steps, env_name), f)




if __name__ == '__main__':
    optimizer_choices = ['ObGD', 'Obo', 'OboBase', 'OboMetaOpt', 'OboMetaZero', 'ObGD_sq', 'ObGD_sq_plain', 'Obn', 'ObnC', 'ObnN', 'AdaptiveObGD', 'ObtC', 'ObtN', 'Obt', 'Obtnnz', 'Obonz', 'ObGDN', 'ObGDm', 'ObtCm', 'Obtm', 'OboC']
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='Ant-v5')  # HalfCheetah-v4
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    parser.add_argument('--max_time', type=str, default='1000:00:00')  # in HH:MM:SS

    parser.add_argument('--policy_hidden_depth', type=int, default=2)
    parser.add_argument('--policy_hidden_width', type=int, default=128)
    parser.add_argument('--policy_initialization_sparsity', type=float, default=0.9)  # larger is more sparse
    parser.add_argument('--policy_optimizer', type=str, default='ObGD', choices=optimizer_choices)
    parser.add_argument('--policy_kappa', type=float, default=3.0)
    parser.add_argument('--policy_gamma', type=float, default=0.99)
    parser.add_argument('--policy_lamda', type=float, default=0.0)
    parser.add_argument('--policy_lr', type=float, default=1.0)
    parser.add_argument('--policy_momentum', type=float, default=0.0)
    parser.add_argument('--policy_entropy_coeff', type=float, default=0.01)  # was entropy_coeff
    parser.add_argument('--policy_u_trace', type=float, default=0.01)  # for Obn
    parser.add_argument('--policy_entrywise_normalization', type=str, default='RMSProp')  # 'none' or 'RMSProp'
    parser.add_argument('--policy_beta2', type=float, default=0.999)  # for Obn
    parser.add_argument('--policy_delta_trace', type=float, default=0.01)  # for ObnN
    parser.add_argument('--policy_weight_decay', type=float, default=0.0) 
    parser.add_argument('--policy_in_trace_sample_scaling', type=str, default='True', choices=['True', 'False'])  # for Obt
    parser.add_argument('--policy_sig_power', type=float, default=2) # for Obt
    parser.add_argument('--policy_delta_clip', type=str, default='none') # for Obt    # 'none', '1', '10_avg_sq__dec_0.99', '10_avg_abs__dec_0.95', '10_avg_abs_max_1__dec_0.9', '10_avg_sq_max_20avg__dec_0.99'
    parser.add_argument('--policy_delta_norm', type=str, default='none') # for Obt    # 'none', '99sq', '99abs', '995clipSq', '995clipAbs'
    parser.add_argument('--policy_meta_stepsize', type=float, default=1e-3)
    parser.add_argument('--policy_beta2_meta', type=float, default=0.999)
    parser.add_argument('--policy_stepsize_parameterization', type=str, default='sigmoid') # 'sigmoid', 'identity', 'exp'
    parser.add_argument('--policy_h_decay_meta', type=float, default=0.9999)
    parser.add_argument('--policy_epsilon_meta', type=float, default=1e-3)
    parser.add_argument('--policy_meta_loss_type', type=str, default='IS')
    parser.add_argument('--policy_meta_shadow_dist_reg', type=float, default=0.0)
    parser.add_argument('--policy_clip_zeta_meta', type=str, default='none')  # 'none', 'minmax_10'

    parser.add_argument('--critic_hidden_depth', type=int, default=2)
    parser.add_argument('--critic_hidden_width', type=int, default=128)
    parser.add_argument('--critic_initialization_sparsity', type=float, default=0.9)  # larger is more sparse
    parser.add_argument('--critic_optimizer', type=str, default='ObnC', choices=optimizer_choices)
    parser.add_argument('--critic_kappa', type=float, default=2.0)
    parser.add_argument('--critic_gamma', type=float, default=0.99)
    parser.add_argument('--critic_lamda', type=float, default=0.0)
    parser.add_argument('--critic_lr', type=float, default=1.0)
    parser.add_argument('--critic_momentum', type=float, default=0.0)
    parser.add_argument('--critic_u_trace', type=float, default=0.01)  # for Obn
    parser.add_argument('--critic_entrywise_normalization', type=str, default='RMSProp')  # 'none' or 'RMSProp'
    parser.add_argument('--critic_beta2', type=float, default=0.999)  # for Obn
    parser.add_argument('--critic_delta_trace', type=float, default=0.01)  # for ObnN
    parser.add_argument('--critic_weight_decay', type=float, default=0.0) 
    parser.add_argument('--critic_in_trace_sample_scaling', type=str, default='True', choices=['True', 'False']) # for Obt
    parser.add_argument('--critic_sig_power', type=float, default=2) # for Obt
    parser.add_argument('--critic_delta_clip', type=str, default='none') # for Obt    # 'none', '1', '10_avg_sq__dec_0.99', '10_avg_abs__dec_0.95', '10_avg_abs_max_1__dec_0.9', '10_avg_sq_max_20avg__dec_0.99'
    parser.add_argument('--critic_delta_norm', type=str, default='none') # for Obt    # 'none', '99sq', '99abs', '995clipSq', '995clipAbs'
    parser.add_argument('--critic_meta_stepsize', type=float, default=1e-3)
    parser.add_argument('--critic_beta2_meta', type=float, default=0.999)
    parser.add_argument('--critic_stepsize_parameterization', type=str, default='sigmoid') # 'sigmoid', 'identity', 'exp'
    parser.add_argument('--critic_h_decay_meta', type=float, default=0.9999)
    parser.add_argument('--critic_epsilon_meta', type=float, default=1e-3)
    parser.add_argument('--critic_meta_loss_type', type=str, default='TD') # TD, RG, MC__mu_0.999__epEndOnly_False__epContagious_Flase
    parser.add_argument('--critic_meta_shadow_dist_reg', type=float, default=0.0)
    parser.add_argument('--critic_clip_zeta_meta', type=str, default='none')  # 'none', 'minmax_10'

    parser.add_argument('--observer_hidden_depth', type=int, default=2)
    parser.add_argument('--observer_hidden_width', type=int, default=128)
    parser.add_argument('--observer_initialization_sparsity', type=float, default=0.9)  # larger is more sparse
    parser.add_argument('--observer_optimizer', type=str, default='none', choices=optimizer_choices+['none','monte_carlo'])
    parser.add_argument('--observer_kappa', type=float, default=2.0)
    parser.add_argument('--observer_gamma', type=float, default=0.99)
    parser.add_argument('--observer_lamda', type=float, default=0.0)
    parser.add_argument('--observer_lr', type=float, default=1.0)
    parser.add_argument('--observer_momentum', type=float, default=0.0)
    parser.add_argument('--observer_u_trace', type=float, default=0.01)  # for Obn
    parser.add_argument('--observer_entrywise_normalization', type=str, default='RMSProp')  # 'none' or 'RMSProp'
    parser.add_argument('--observer_beta2', type=float, default=0.999)  # for Obn
    parser.add_argument('--observer_delta_trace', type=float, default=0.01)  # for ObnN
    parser.add_argument('--observer_weight_decay', type=float, default=0.0) 
    parser.add_argument('--observer_in_trace_sample_scaling', type=str, default='True', choices=['True', 'False']) # for Obt
    parser.add_argument('--observer_sig_power', type=float, default=2) # for Obt
    parser.add_argument('--observer_delta_clip', type=str, default='none') # for Obt    # 'none', '1', '10_avg_sq__dec_0.99', '10_avg_abs__dec_0.95', '10_avg_abs_max_1__dec_0.9', '10_avg_sq_max_20avg__dec_0.99'
    parser.add_argument('--observer_delta_norm', type=str, default='none') # for Obt    # 'none', '99sq', '99abs', '995clipSq', '995clipAbs'
    parser.add_argument('--observer_meta_stepsize', type=float, default=1e-4)
    parser.add_argument('--observer_beta2_meta', type=float, default=0.999)
    parser.add_argument('--observer_stepsize_parameterization', type=str, default='sigmoid') # 'sigmoid', 'identity', 'exp'
    parser.add_argument('--observer_h_decay_meta', type=float, default=0.9999)
    parser.add_argument('--observer_epsilon_meta', type=float, default=1e-3)
    parser.add_argument('--observer_meta_loss_type', type=str, default='TD') # TD, RG, MC__mu_0.999__epEndOnly_False__epContagious_Flase
    parser.add_argument('--observer_meta_shadow_dist_reg', type=float, default=0.0)
    parser.add_argument('--observer_clip_zeta_meta', type=str, default='none')  # 'none', 'minmax_10'

    parser.add_argument('--log_backend', type=str, default='none', choices=['none', 'tensorboard', 'wandb', 'wandb_offline'])
    parser.add_argument('--log_dir', type=str, default='/home/asharif/StreamX_optimizer/WandB_offline', help='WandB offline log dir (if backend=wandb_offline)')
    parser.add_argument('--log_dir_for_pickle', type=str, default='none', help='/home/asharif/StreamX_optimizer/pickle')
    parser.add_argument('--logging_level', type=str, default='light', help='how much detail to show on wandb', choices=['light', 'heavy'])
    parser.add_argument('--project', type=str, default='test_stream_CC', help='WandB project (if backend=wandb)')
    parser.add_argument('--run_name', type=str, default='', help='Run name for logger')  # __sqrt_coeff
    parser.add_argument('--uID', type=str, default='', help='')  # not used
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()

    # ---- Spec dicts ----
    shared_params = ['optimizer', 'kappa', 'gamma', 'lamda', 'weight_decay', 'hidden_depth', 'hidden_width', 'initialization_sparsity']
    required_optimizer_params = {
        'none':         [],         # for observer
        'monte_carlo':  ['gamma'],  # for observer
        'ObGD':         shared_params + ['lr'],
        'ObGD_sq':      shared_params + ['lr'],
        'ObGD_sq_plain':shared_params + ['lr'],
        'AdaptiveObGD': shared_params + ['lr'],
        'Obn':          shared_params + ['lr', 'entrywise_normalization', 'beta2', 'u_trace'],
        'ObnC':         shared_params + ['lr', 'entrywise_normalization', 'beta2', 'u_trace'],
        'ObnN':         shared_params + ['lr', 'entrywise_normalization', 'beta2', 'u_trace', 'delta_trace'],
        'ObtC':         shared_params + ['entrywise_normalization', 'beta2', 'sig_power', 'in_trace_sample_scaling'],
        'ObtCm':        shared_params + ['entrywise_normalization', 'beta2', 'sig_power', 'in_trace_sample_scaling', 'momentum'],
        'ObtN':         shared_params + ['entrywise_normalization', 'beta2', 'sig_power', 'in_trace_sample_scaling', 'delta_trace'],
        'Obt':          shared_params + ['entrywise_normalization', 'beta2', 'sig_power', 'in_trace_sample_scaling', 'delta_clip', 'delta_norm'],
        'Obtm':         shared_params + ['entrywise_normalization', 'beta2', 'sig_power', 'in_trace_sample_scaling', 'delta_clip', 'delta_norm', 'momentum'],
        'Obtnnz':       shared_params + ['entrywise_normalization', 'beta2', 'u_trace', 'delta_clip', 'delta_norm'],
        'Obonz':        shared_params + ['entrywise_normalization', 'beta2', 'u_trace', 'delta_clip', 'delta_norm', 'momentum'],
        'ObGDN':        shared_params + ['lr', 'delta_clip', 'delta_norm'],
        'ObGDm':        shared_params + ['lr', 'momentum'],
        'Obo':          shared_params + ['entrywise_normalization', 'beta2', 'sig_power', 'in_trace_sample_scaling', 'delta_clip', 'delta_norm', 'momentum', 'u_trace'],
        'OboC':         shared_params + ['entrywise_normalization', 'beta2', 'sig_power', 'in_trace_sample_scaling', 'momentum', 'u_trace'],
        'OboBase':      shared_params + ['entrywise_normalization', 'beta2', 'delta_clip', 'delta_norm', 'momentum'],
        'OboMetaOpt':   shared_params + ['entrywise_normalization', 'beta2', 'delta_clip', 'delta_norm', 'momentum'] + ['meta_stepsize', 'beta2_meta', 'stepsize_parameterization', 'h_decay_meta'],
        'OboMetaZero':   shared_params + ['entrywise_normalization', 'beta2', 'delta_clip', 'delta_norm', 'momentum'] + ['meta_stepsize', 'beta2_meta', 'stepsize_parameterization', 'epsilon_meta', 'meta_loss_type', 'meta_shadow_dist_reg'],
        }
    
    def build_spec(kind, args, required_optimizer_params) -> dict:
        opt = getattr(args, f'{kind}_optimizer')
        required_params = required_optimizer_params[opt]
        spec = {'optimizer': opt}
        for key in required_params:
            spec[key] = getattr(args, f'{kind}_{key}')
        return spec

    policy_spec   = build_spec(kind='policy',   args=args, required_optimizer_params=required_optimizer_params)
    policy_spec.update({'entropy_coeff': args.policy_entropy_coeff})
    critic_spec   = build_spec(kind='critic',   args=args, required_optimizer_params=required_optimizer_params)
    observer_spec = build_spec(kind='observer', args=args, required_optimizer_params=required_optimizer_params)

    # ---- Logging dict ----
    logging_spec = {
        'backend': args.log_backend,
        'dir': f'{args.log_dir}_{args.env_name}',
        'dir_pickle': args.log_dir_for_pickle,
        'level': args.logging_level,
        'project': f'{args.project}_{args.env_name}',
        'run_name': args.run_name,
        'uID': args.uID,
    }

    main(args.env_name, args.seed, args.total_steps, args.max_time, policy_spec=policy_spec, critic_spec=critic_spec, observer_spec=observer_spec, logging_spec=logging_spec, debug=args.debug, render=args.render)  

