import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from optim import OboPolicy, OboValue
from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, ScaleReward, SampleMeanStd
from sparse_init import sparse_init

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class TruncatedNormal:
    """Truncated Normal distribution with bounds [-1, 1] for each dimension."""
    def __init__(self, mean, std, low=-1.0, high=1.0, eps=1e-6):
        # Clamp mean to be within reasonable bounds relative to [-1, 1]
        self.mean = torch.clamp(mean, -10.0, 10.0)
        # Ensure std is positive and not too small
        self.std = torch.clamp(std, 1e-3, 10.0)
        self.low = low
        self.high = high
        self.eps = eps
        
        # Standardized bounds: (bound - mean) / std
        self.alpha = (low - self.mean) / self.std
        self.beta = (high - self.mean) / self.std
        
        # Clamp alpha and beta to avoid extreme values
        self.alpha = torch.clamp(self.alpha, -10.0, 10.0)
        self.beta = torch.clamp(self.beta, -10.0, 10.0)
        
        # CDF values at bounds
        self.normal = torch.distributions.Normal(0, 1)
        self.cdf_alpha = self.normal.cdf(self.alpha)
        self.cdf_beta = self.normal.cdf(self.beta)
        # Normalization constant with minimum value to avoid division by zero
        self.Z = torch.clamp(self.cdf_beta - self.cdf_alpha, min=eps)
        
    def rsample(self):
        """Sample using reparameterization trick with inverse CDF method."""
        # Sample uniform in [0, 1]
        u = torch.rand_like(self.mean)
        # Scale to [Φ(α), Φ(β)]
        u_scaled = self.cdf_alpha + u * self.Z
        # Clamp to avoid numerical issues at boundaries
        u_scaled = torch.clamp(u_scaled, self.eps, 1.0 - self.eps)
        # Inverse CDF (quantile function)
        z = self.normal.icdf(u_scaled)
        # Clamp z to avoid extreme values
        z = torch.clamp(z, -10.0, 10.0)
        # Transform back to original scale
        sample = self.mean + self.std * z
        # Clamp to ensure bounds (for numerical stability)
        return torch.clamp(sample, self.low, self.high)
    
    def log_prob(self, value):
        """Log probability of value under truncated normal."""
        # Clamp value to bounds
        value = torch.clamp(value, self.low, self.high)
        # Standardize
        z = (value - self.mean) / self.std
        # Log prob = log(pdf(z)) - log(std) - log(Z)
        log_prob = self.normal.log_prob(z) - torch.log(self.std) - torch.log(self.Z)
        return log_prob
    
    def entropy(self):
        """Exact entropy of truncated normal distribution.
        
        H = log(σ * Z * sqrt(2πe)) + (α*φ(α) - β*φ(β)) / (2*Z)
        where φ is the standard normal PDF, Φ is CDF, Z = Φ(β) - Φ(α)
        """
        # PDF values at bounds
        pdf_alpha = torch.exp(self.normal.log_prob(self.alpha))
        pdf_beta = torch.exp(self.normal.log_prob(self.beta))
        
        # Entropy formula for truncated normal
        # Add small epsilon to avoid log(0)
        log_term = torch.log(self.std * self.Z * np.sqrt(2 * np.pi * np.e) + self.eps)
        correction_term = (self.alpha * pdf_alpha - self.beta * pdf_beta) / (2 * self.Z)
        entropy = log_term + correction_term
        
        # Clamp entropy to reasonable values
        entropy = torch.clamp(entropy, -10.0, 10.0)
        
        return entropy

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

class QFunction(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
        super(QFunction, self).__init__()
        self.fc_layer = nn.Linear(n_obs + n_actions, hidden_size)
        self.hidden_layer  = nn.Linear(hidden_size, hidden_size)
        self.linear_layer  = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)      
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        return self.linear_layer(x)

class StreamAVG(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr_actor=1.0, lr_critic=1.0, gamma=0.99, lamda=0.8, 
                 kappa_policy=3.0, kappa_value=2.0, alpha=0.01):
        super(StreamAVG, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.q_net = QFunction(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.optimizer_policy = OboPolicy(self.policy_net.parameters(), lr=lr_actor, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_q = OboValue(self.q_net.parameters(), lr=lr_critic, gamma=gamma, lamda=lamda, kappa=kappa_value)
        self.action_stats = SampleMeanStd(shape=(n_actions,))

    def pi(self, x):
        return self.policy_net(x)

    def q(self, x, a):
        return self.q_net(x, a)

    def sample_action(self, s):
        x = torch.from_numpy(s).float()
        mu, std = self.pi(x)
        dist = TruncatedNormal(mu, std, low=-1.0, high=1.0)
        action = dist.rsample()
        return action

    def normalize_action(self, a):
        """Normalize actions using running statistics."""
        mean = torch.tensor(self.action_stats.mean, dtype=torch.float32)
        std = torch.tensor(np.sqrt(self.action_stats.var), dtype=torch.float32)
        return (a - mean) / (std + 1e-8)

    def update_params(self, s, a, r, s_prime, done):
        done_mask = 0 if done else 1
        s = torch.tensor(np.array(s), dtype=torch.float)
        r = torch.tensor(np.array(r), dtype=torch.float)
        s_prime = torch.tensor(np.array(s_prime), dtype=torch.float)
        done_mask = torch.tensor(np.array(done_mask), dtype=torch.float)

        # Normalize actions before passing to Q-function
        a_norm = self.normalize_action(a.detach())
        q_sa = self.q(s, a_norm)
        
        with torch.no_grad():
            mu_prime, std_prime = self.pi(s_prime)
            dist_prime = TruncatedNormal(mu_prime, std_prime, low=-1.0, high=1.0)
            a_prime = dist_prime.rsample()
            entropy_prime = dist_prime.entropy().sum()
            
            a_prime_norm = self.normalize_action(a_prime)
            q_prime = self.q(s_prime, a_prime_norm)
            target_q = r + done_mask * self.gamma * (q_prime + self.alpha * entropy_prime)

        delta = target_q - q_sa
        q_loss = -q_sa

        mu, std = self.pi(s)
        dist = TruncatedNormal(mu, std, low=-1.0, high=1.0)
        a_reparam = dist.rsample()
        entropy = dist.entropy().sum()
        a_reparam_norm = self.normalize_action(a_reparam)
        q_reparam = self.q(s, a_reparam_norm)

        #print(a.detach().numpy())
        #print(a_reparam.detach().numpy())    
        #print()
        policy_loss = -self.alpha * entropy - q_reparam

        self.optimizer_q.zero_grad()
        q_loss.backward()
        self.optimizer_q.step(delta.item(), reset=done)

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step(1.0, reset=1.0)

def main(env_name, seed, lr_actor, lr_critic, gamma, lamda, total_steps, alpha, kappa_policy, kappa_value, debug, render=False):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)
    agent = StreamAVG(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value, alpha=alpha)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    for t in range(1, total_steps+1):
        a_tensor = agent.sample_action(s)
        a_numpy = a_tensor.detach().numpy()
        agent.action_stats.update(a_numpy)
        s_prime, r, terminated, truncated, info = env.step(a_numpy)
        agent.update_params(s, a_tensor, r, s_prime, terminated or truncated)
        s = s_prime
        if terminated or truncated:
            if debug:
                print("Episodic Return: {}, Time Step {}".format(info['episode']['r'].item(), t))
            returns.append(info['episode']['r'].item())
            term_time_steps.append(t)
            terminated, truncated = False, False
            s, _ = env.reset()
    env.close()
    save_dir = "data_stream_avg_{}_lra{}_lrc{}_gamma{}_lamda{}_alpha{}_kappap{}_kappav{}".format(env.spec.id, lr_actor, lr_critic, gamma, lamda, alpha, kappa_policy, kappa_value)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AVG with Truncated Normal')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr_actor', type=float, default=0.0001, help='Learning rate for actor/policy network')
    parser.add_argument('--lr_critic', type=float, default=1.0, help='Learning rate for critic/Q network')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.0)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    parser.add_argument('--alpha', type=float, default=0.001, help='Entropy coefficient')
    parser.add_argument('--kappa_policy', type=float, default=100000.0)
    parser.add_argument('--kappa_value', type=float, default=5.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args.env_name, args.seed, args.lr_actor, args.lr_critic, args.gamma, args.lamda, args.total_steps, args.alpha, args.kappa_policy, args.kappa_value, args.debug, args.render)