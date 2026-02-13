from dataclasses import dataclass
import random



class error_critic():
    def __init__(self, meta_loss_type, gamma):
        '''
        meta_loss_type in ['TD', 'RG', 'MC__mu_0.999__epEndOnly_False__epContagious_Flase']
        '''
        self.loss_type = meta_loss_type.split('__')[0].lower()
        self.gamma = gamma
        if self.loss_type == 'mc':
            self.mc_mu = float(meta_loss_type.split('__mu_')[-1].split('__')[0])
            self.ep_contagious = meta_loss_type.split('__epContagious_')[-1] in ['True', 'true', '1']
            self.error_past_episodes = 0.0
            self.in_episode_MC_loss = AggregateMonteCarloError(gamma=gamma, mu=self.mc_mu)
        #self.ep_end_only = meta_loss_type.lower().startswith('mc') and meta_loss_type.split('__epEndOnly_')[1].split('__')[0] in ['True', 'true', '1']


    def step(self, v_s, v_prime, r, v_prime_main, reset):
        # v_main is the value (v_prime) of the main network, for both main and shadow critics
        
        if self.loss_type == 'td':
            return (r + self.gamma * v_prime_main - v_s)**2
        if self.loss_type == 'rg':
            return (r + self.gamma * v_prime - v_s)**2
        if self.loss_type == 'mc':
            error_in_episode = self.in_episode_MC_loss.step(r, v_s, v_prime)
            self.error_past_episodes *= self.mc_mu
            f_t = error_in_episode + (self.error_past_episodes if self.ep_contagious else 0.0) 
            if reset:
                self.error_past_episodes += error_in_episode
                self.in_episode_MC_loss.reset()
        return f_t



@dataclass
class AggregateMonteCarloError:
    """
    Online computation of
      f_t(w_t) = (1-mu) * sum_{tau=ep_start}^t mu^{t-tau}
                  (R_{tau:t} + gamma^{t-tau+1} V_{w_t}(s_{t+1}) - V_{w_tau}(s_tau))^2
    where
      R_{tau:t} = sum_{i=tau}^t gamma^{i-tau} r_i.

    Call step(r_t, v_t, v_next) with:
      v_t    = V_{w_t}(s_t)       (current weights)
      v_next = V_{w_t}(s_{t+1})   (same current weights)
    """
    gamma: float
    mu: float

    def __post_init__(self):
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1].")
        if not (0.0 <= self.mu < 1.0):
            raise ValueError("mu must be in [0, 1).")
        self.q = self.mu * self.gamma
        self.q2 = self.mu * (self.gamma * self.gamma)
        self.reset()

    def reset(self):
        # Main sums
        self.S = 0.0   # (kept; unused by corrected f_t)
        self.C = 0.0   # (kept; unused by corrected f_t)
        self.B = 0.0   # sum mu^{t-tau} v_tau^2
        self.D = 0.0   # (kept; unused by corrected f_t)
        self.A = 0.0   # sum mu^{t-tau} R_{tau:t}^2
        self.E = 0.0   # sum mu^{t-tau} R_{tau:t} v_tau

        # Auxiliary traces induced by gamma^{t-tau}
        self.T  = 0.0  # (kept; unused by corrected f_t)
        self.T2 = 0.0  # sum (mu*gamma^2)^{t-tau}
        self.U  = 0.0  # sum (mu*gamma)^{t-tau} R_{tau:t}
        self.Cq = 0.0  # sum (mu*gamma)^{t-tau} v_tau

    def step(self, r_t: float, v_t: float, v_next: float) -> float:
        r_t = float(r_t)
        v_t = float(v_t)
        v_next = float(v_next)

        # cache previous values needed
        S_prev, C_prev, B_prev = self.S, self.C, self.B
        D_prev, A_prev, E_prev = self.D, self.A, self.E
        T_prev, T2_prev, U_prev, Cq_prev = self.T, self.T2, self.U, self.Cq

        # easy traces
        S_t = 1.0 + self.mu * S_prev
        C_t = v_t + self.mu * C_prev
        B_t = v_t * v_t + self.mu * B_prev

        # auxiliary geometric traces
        T_t  = 1.0 + self.q  * T_prev
        T2_t = 1.0 + self.q2 * T2_prev
        Cq_t = v_t + self.q  * Cq_prev

        # return-related traces
        D_t = self.mu * D_prev + r_t * T_t
        E_t = self.mu * E_prev + r_t * Cq_t

        U_t = self.q * U_prev + r_t * T2_t
        A_t = self.mu * A_prev + 2.0 * r_t * self.q * U_prev + (r_t * r_t) * T2_t

        # store updated
        self.S, self.C, self.B = S_t, C_t, B_t
        self.T, self.T2, self.Cq = T_t, T2_t, Cq_t
        self.D, self.E, self.U, self.A = D_t, E_t, U_t, A_t

        # corrected f_t: gamma^{t-tau+1} v_next = gamma^{t-tau} * (gamma v_next)
        c_t = self.gamma * v_next
        f_t = (1.0 - self.mu) * (
            A_t + B_t + (c_t * c_t) * T2_t
            - 2.0 * c_t * Cq_t + 2.0 * c_t * U_t
            - 2.0 * E_t
        )
        return f_t


if __name__ == "__main__":
    # online computation
    #online = AggregateMonteCarloError(gamma=0.99, mu=0.95)
    
    def brute_force_ft(gamma, mu, rs, vs, vnexts):
        outs = []
        for t in range(len(rs)):
            acc = 0.0
            for tau in range(t + 1):
                # R_{tau:t}
                R = 0.0
                for i in range(tau, t + 1):
                    R += (gamma ** (i - tau)) * rs[i]
                term = (R + (gamma ** (t - tau + 1)) * vnexts[t] - vs[tau]) ** 2
                acc += (mu ** (t - tau)) * term
            outs.append((1.0 - mu) * acc)
        return outs

    # run one random check
    gamma = 0.9
    mu = 0.7
    T = 100
    rs = [random.uniform(-1, 1) for _ in range(T)]
    vs = [random.uniform(-1, 1) for _ in range(T)]
    vnexts = [random.uniform(-1, 1) for _ in range(T)]

    agg = AggregateMonteCarloError(gamma=gamma, mu=mu)
    online = [agg.step(rs[t], vs[t], vnexts[t]) for t in range(T)]
    brute = brute_force_ft(gamma, mu, rs, vs, vnexts)

    print(sum(online)/T, (max(abs(a - b) for a, b in zip(online, brute))))
