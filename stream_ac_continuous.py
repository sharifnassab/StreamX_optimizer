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

from optim import ObGD_sq as ObGDsq_Optimizer
from optim import ObGD_sq_plain as ObGDsqPlain_Optimizer
from optim import ObGD as ObGD_Optimizer
from optim import AdaptiveObGD as AdaptiveObGD_Optimizer
from optim import Obn as Obn_Optimizer
from optim import ObnC as ObnC_Optimizer
from optim import ObnN as ObnN_Optimizer
from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init

# NEW: our tiny logger (wandb or tensorboard)
from logger import get_logger
print('All imports done. 2')  

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
    

def compute_weight_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.requires_grad:  # only count learnable params
            total += torch.sum(p ** 2)
    return torch.sqrt(total)

def spec_to_name(spec: dict) -> str:
    short_hand_mapping={'optimizer':    '',
                        'gamma':        '_gam',
                        'lamda':        'lam',
                        'kappa':        'k',
                        'entrywise_normalization': 'en',
                        'beta2':        'b2',
                        'u_trace':      'u',
                        'entropy_coeff':'ent',
                        'lr':           'lr',
                        'weight_decay': 'wd',
                        'delta_trace':  'delTr'
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
        hidden_size: int = 128,
    ):
        super().__init__()

        # Keep the raw specs (useful for logging/inspect)
        self.policy_spec   = dict(policy_spec or {})
        self.critic_spec   = dict(critic_spec or {})
        self.observer_spec = dict(observer_spec or {})

        # ---- Nets ----
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.critic_net = Critic(n_obs=n_obs, hidden_size=hidden_size)
        self.observer_net = Critic(n_obs=n_obs, hidden_size=hidden_size)

        # ---- Hyperparameters ----
        self.gamma_policy   = float(self.policy_spec.get('gamma'))
        self.gamma_critic   = float(self.critic_spec.get('gamma'))
        self.entropy_coeff  = float(self.policy_spec.get('entropy_coeff'))
        self.observer_exists = str(self.observer_spec.get('optimizer', 'none')).lower() != 'none'
        if self.observer_exists:
            self.gamma_observer = float(self.observer_spec.get('gamma'))    
        
        # ---- Optimizers (per-spec) ----
        self.optimizer_policy   = self._build_optimizer(self.policy_net.parameters(),   self.policy_spec,   role="policy")
        self.optimizer_critic   = self._build_optimizer(self.critic_net.parameters(),   self.critic_spec,   role="critic")
        self.optimizer_observer = (self._build_optimizer(self.observer_net.parameters(), self.observer_spec, role="observer") if self.observer_exists else None)


    def _build_optimizer(self, params, spec: dict, role: str):
        spec = spec or {}
        opt_name = spec.get('optimizer', 'none').strip().lower()
        if opt_name == 'none':
            return None
        
        lr     = float(spec.get('lr', 3e-4))
        gamma  = float(spec.get('gamma'))
        lamda  = float(spec.get('lamda'))
        kappa  = float(spec.get('kappa'))

        # Only for Obn
        u_trace = float(spec.get('u_trace', 0.01))
        entrywise_normalization = spec.get('entrywise_normalization', 'none')
        beta2  = float(spec.get('beta2', 0.999))
        delta_trace = float(spec.get('delta_trace', 0.01))


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

        raise ValueError(f"Unknown optimizer '{spec.get('optimizer')}' for role '{role}'.")


    def pi(self, x):
        return self.policy_net(x)

    def v(self, x):
        return self.critic_net(x)
    
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

    def update_params(self, s, a, r, s_prime, done):
        done_mask = 0 if done else 1
        s = torch.tensor(np.array(s), dtype=torch.float)
        a = torch.tensor(np.array(a), dtype=torch.float)
        r = torch.tensor(np.array(r), dtype=torch.float)
        s_prime = torch.tensor(np.array(s_prime), dtype=torch.float)
        done_mask_t = torch.tensor(np.array(done_mask), dtype=torch.float)

        v_s, v_prime = self.v(s), self.v(s_prime)

        td_target_critic = r + self.gamma_critic * v_prime * done_mask_t
        td_target_policy = r + self.gamma_policy * v_prime * done_mask_t
        delta_critic = td_target_critic - v_s
        delta_policy = td_target_policy - v_s

        mu, std = self.pi(s)
        dist = Normal(mu, std)

        log_prob_pi = -(dist.log_prob(a)).sum()
        entropy_pi = -self.entropy_coeff * dist.entropy().sum() * torch.sign(delta_policy).item()
        self.optimizer_policy.zero_grad()
        self.optimizer_critic.zero_grad()
        if self.observer_exists: 
            self.optimizer_observer.zero_grad()
        

        (-v_s).backward()
        (log_prob_pi + entropy_pi).backward()
        info_policy = self.optimizer_policy.step(delta_policy.item(), reset=done)
        info_critic = self.optimizer_critic.step(delta_critic.item(), reset=done)


        # observer update:
        info_observer = {}
        if self.observer_exists:
            v_s_obs, v_prime_obs = self.v_observer(s), self.v_observer(s_prime)
            (-v_s_obs).backward()
            delta_obs = r + self.gamma_observer * v_prime_obs * done_mask_t - v_s_obs
            info_observer = self.optimizer_observer.step(delta_obs.item(), reset=done)
            # For logging:
            info_observer = {**(info_observer or {}),
                "td_error": float(delta_obs.item()),
                "v(s)": float(v_s_obs.item()),
            }


        # Logging:
        info_policy = {**(info_policy or {}),
            "policy_log_prob": float((-log_prob_pi).item()),  # log π(a|s)
            "std_mean": float(std.mean().item()),
            "mu_norm": float(mu.norm().item()),
            "reward": float(r.item()),
        }
        info_critic = {**(info_critic or {}),
            "td_error": float(delta_critic.item()),
            "v(s)": float(v_s.item()),
        }
        
        # Optional overshooting check
        if False: #if self.overshooting_info:
            v_s2 = self.v(s)
            v_prime2 = self.v(s_prime)
            td_target2 = r + self.gamma * v_prime2 * done_mask_t
            delta_bar = td_target2 - v_s2
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")

        return  {'policy':info_policy, 'critic':info_critic, 'observer':info_observer}

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
    run_name = (f'{env.spec.id}_____Policy_{spec_to_name(policy_spec)}_____Critic_{spec_to_name(critic_spec)}' +
                (f'_____Observer_{spec_to_name(observer_spec)}' if agent.observer_exists else '') + 
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
    list_ep_R, list_ep_v, list_ep_S = [], {'critic':[], 'observer':[]}, []
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

        step_info = agent.update_params(s, a, r, s_prime, (terminated or truncated),)
        s = s_prime

        for net in ['critic', 'observer']:
            list_ep_v[net].append(step_info[net].get('v(s)',0.0))

        if (t % 500_000) <= 2025 and  logging_level in ['heavy']:
            expanded_step_info = {}
            for net_type in step_info:
                section = step_info.get(net_type) or {}
                for metric, val in section.items():
                    expanded_step_info[f'{net_type}/{metric}'] = val
            expanded_step_info.update({
                                    'rewards/original':info['reward_immediate'], 
                                    'rewards/scaled':r,
                                    'rewards/ratio': r/info['reward_immediate'],
                                  })
            logger.log(expanded_step_info, step=t)


        ep_steps += 1
        ep_min_inv_M_sum += float(step_info.get('policy', {}).get('min_inv_M', 0.0))
        ep_policy_std += float(step_info.get('policy', {}).get('std_mean', 0.0))

        if terminated or truncated:
            if agent.observer_exists:
                ep_pred_error = compute_prediction_MSE(list_ep_R, list_ep_v['observer'], agent.gamma_observer)
                ep_pred_error_end_of_episode_W = compute_prediction_MSE_end_of_episode_W(list_ep_R, list_ep_S, agent.observer_net, agent.gamma_observer)
            ep_pred_error_critic = compute_prediction_MSE(list_ep_R, list_ep_v['critic'], agent.gamma_critic)
            

            ep_return = info["episode"]["r"]
            ep_len = info["episode"].get("l", ep_steps)
            avg_min_inv_M = ep_min_inv_M_sum / max(ep_steps, 1) if ep_min_inv_M_sum > 0 else None
            avg_policy_std = ep_policy_std / max(ep_steps, 1) if ep_policy_std > 0 else None

            log_payload = {
                "_episode/return": float(ep_return),
                "_episode/length": float(ep_len),
                #"policy/avg_min_inv_M": float(avg_min_inv_M) if avg_min_inv_M is not None else None,
                "policy/avg_policy_std": float(avg_policy_std) if avg_policy_std is not None else None,
                #"critic_prediction/episode_MSE":  float(ep_pred_error_critic['ep_MSE_error']),
                #"critic_prediction/episode_abs":  float(ep_pred_error_critic['ep_abs_error']),
                "critic_prediction/episode_RMSE": float(np.sqrt(ep_pred_error_critic['ep_MSE_error'])),
            }
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
            logger.log(log_payload, step=t)


            if debug:
                print(f"Episodic Return: {ep_return}, Time Step {t}")

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
            list_ep_R, list_ep_v, list_ep_S = [], {'critic':[], 'observer':[]}, []
            episode_number+=1

    print(f"total time = {time.gmtime(int(time.time() - start_time))}")

    env.close()
    logger.finish()

    if logging_spec['dir_pickle'] != 'none':
        save_dir = os.path.join(logging_spec['dir_pickle'],  run_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
            pickle.dump((returns, term_time_steps, env_name), f)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='Ant-v5')  # HalfCheetah-v4
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    parser.add_argument('--max_time', type=str, default='1000:00:00')  # in HH:MM:SS

    parser.add_argument('--policy_optimizer', type=str, default='ObGD', choices=['ObGD', 'ObGD_sq', 'ObGD_sq_plain', 'Obn', 'ObnC', 'ObnN', 'AdaptiveObGD'])
    parser.add_argument('--policy_kappa', type=float, default=3.0)
    parser.add_argument('--policy_gamma', type=float, default=0.99)
    parser.add_argument('--policy_lamda', type=float, default=0.0)
    parser.add_argument('--policy_lr', type=float, default=1.0)
    parser.add_argument('--policy_entropy_coeff', type=float, default=0.01)  # was entropy_coeff
    parser.add_argument('--policy_u_trace', type=float, default=0.01)  # for Obn
    parser.add_argument('--policy_entrywise_normalization', type=str, default='RMSProp')  # 'none' or 'RMSProp'
    parser.add_argument('--policy_beta2', type=float, default=0.999)  # for Obn
    parser.add_argument('--policy_delta_trace', type=float, default=0.01)  # for ObnN
    parser.add_argument('--policy_weight_decay', type=float, default=0.0) 
    
    parser.add_argument('--critic_optimizer', type=str, default='ObnC', choices=['ObGD', 'ObGD_sq', 'ObGD_sq_plain', 'Obn', 'ObnC', 'ObnN', 'AdaptiveObGD'])
    parser.add_argument('--critic_kappa', type=float, default=2.0)
    parser.add_argument('--critic_gamma', type=float, default=0.99)
    parser.add_argument('--critic_lamda', type=float, default=0.0)
    parser.add_argument('--critic_lr', type=float, default=1.0)
    parser.add_argument('--critic_u_trace', type=float, default=0.01)  # for Obn
    parser.add_argument('--critic_entrywise_normalization', type=str, default='RMSProp')  # 'none' or 'RMSProp'
    parser.add_argument('--critic_beta2', type=float, default=0.999)  # for Obn
    parser.add_argument('--critic_delta_trace', type=float, default=0.01)  # for ObnN
    parser.add_argument('--critic_weight_decay', type=float, default=0.0) 
    

    parser.add_argument('--observer_optimizer', type=str, default='none', choices=['none', 'ObGD', 'ObGD_sq', 'ObGD_sq_plain', 'Obn', 'ObnC', 'ObnN', 'AdaptiveObGD'])
    parser.add_argument('--observer_kappa', type=float, default=2.0)
    parser.add_argument('--observer_gamma', type=float, default=0.99)
    parser.add_argument('--observer_lamda', type=float, default=0.0)
    parser.add_argument('--observer_lr', type=float, default=1.0)
    parser.add_argument('--observer_u_trace', type=float, default=0.01)  # for Obn
    parser.add_argument('--observer_entrywise_normalization', type=str, default='RMSProp')  # 'none' or 'RMSProp'
    parser.add_argument('--observer_beta2', type=float, default=0.999)  # for Obn
    parser.add_argument('--observer_delta_trace', type=float, default=0.01)  # for ObnN
    parser.add_argument('--observer_weight_decay', type=float, default=0.0) 
    
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
    policy_spec = {
        'optimizer': args.policy_optimizer,
        'kappa': args.policy_kappa,
        'gamma': args.policy_gamma,
        'lamda': args.policy_lamda,
        'weight_decay': args.policy_weight_decay,
        'lr': args.policy_lr,
        'entropy_coeff': args.policy_entropy_coeff}
    if policy_spec['optimizer'] in ['Obn','ObnC','ObnN']:
        policy_spec.update({'u_trace': args.policy_u_trace,
                            'entrywise_normalization': args.policy_entrywise_normalization,
                            'beta2': args.policy_beta2})
        if policy_spec['optimizer'] in ['ObnN']:
            policy_spec.update({'delta_trace': args.policy_delta_trace})

    critic_spec = {
        'optimizer': args.critic_optimizer,
        'kappa': args.critic_kappa,
        'gamma': args.critic_gamma,
        'lamda': args.critic_lamda,
        'weight_decay': args.critic_weight_decay,
        'lr': args.critic_lr}
    if critic_spec['optimizer'] in ['Obn','ObnC','ObnN']:
        critic_spec.update({'u_trace': args.critic_u_trace,
                            'entrywise_normalization': args.critic_entrywise_normalization,
                            'beta2': args.critic_beta2})
        if critic_spec['optimizer'] in ['ObnN']:
            critic_spec.update({'delta_trace': args.critic_delta_trace})

    observer_spec = {'optimizer': args.observer_optimizer}
    if observer_spec['optimizer'] != 'none':
        observer_spec.update({
            'kappa': args.observer_kappa,
            'gamma': args.observer_gamma,
            'lamda': args.observer_lamda,
            'weight_decay': args.observer_weight_decay,
            'lr': args.observer_lr})
    if observer_spec['optimizer'] in ['Obn','ObnC','ObnN']:
        observer_spec.update({'u_trace': args.observer_u_trace,
                              'entrywise_normalization': args.observer_entrywise_normalization,
                              'beta2': args.observer_beta2})
        if observer_spec['optimizer'] in ['ObnN']:
            observer_spec.update({'delta_trace': args.observer_delta_trace})

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

