import torch
import math
from copy import deepcopy
from torch.distributions import Normal
from delta_clipper import delta_clipper
from meta_opt_helper import activation_function, activation_function_inverse, clip_zeta_meta_function
from MetaZero_loss_calculation import error_critic



class OboBaseStats(torch.optim.Optimizer): 
    def __init__(self, params, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', beta2=0.999, rmspower=2.0, entrywise_normalization='none', weight_decay=0.0, momentum=0.0):
        defaults = dict(gamma=gamma, lamda=lamda, beta2=beta2, rmspower=rmspower, beta_momentum=momentum)
        self.gamma = gamma
        self.lamda = lamda
        self.eta = torch.tensor(1.0/kappa)
        self.weight_decay = weight_decay
        self.u_bar = 0.0
        self.sigma = 0.0
        self.t_val = 0
        self.delta_norm=delta_norm
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.delta_clipper = delta_clipper(clip_type=delta_clip, normalization_type=delta_norm)
        super(OboBaseStats, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        safe_delta = self.delta_clipper.clip_and_norm(delta)
        self.delta_normalized = safe_delta
        self.t_val += 1

        # base update
        norm_grad = 0
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                
                if self.entrywise_normalization.lower() in ['rmsprop']:
                    v.mul_(group["beta2"]).add_(torch.pow(torch.abs(p.grad), group["rmspower"]), alpha=1.0 - group["beta2"])
                    v_hat = torch.pow(v / (1.0 - group["beta2"] ** self.t_val), 1.0/group["rmspower"]) + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad)
                z_sum += ((e.square() / v_hat).sum()).abs().item()
                norm_grad += ((p.grad.square() / v_hat).sum()).abs().item()
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad-self.sigma)
        norm_normalizer = math.sqrt((self.sigma/(1-(self.gamma*self.lamda)**self.t_val)) + 1e-12) 
        z_normalizer = math.sqrt(z_sum/(1-(self.gamma*self.lamda)**self.t_val))
        normalizer_ = norm_normalizer * z_normalizer
        step_size = 1 / normalizer_

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                if group["beta_momentum"]>0:
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p.data)
                    m = state["momentum"]
                    if self.entrywise_normalization.lower() == 'rmsprop':
                        m.mul_(group["beta_momentum"]).addcdiv_(e, state["rmsprop_v_hat"], value= safe_delta * step_size * (1-group["beta_momentum"]))
                    else:
                        m.mul_(group["beta_momentum"]).add_(e, alpha= safe_delta * step_size * (1-group["beta_momentum"]))
                    delta_w = m
                else:
                    if self.entrywise_normalization.lower() == 'rmsprop':
                        delta_w = safe_delta * step_size * e / state["rmsprop_v_hat"]
                    else:
                        delta_w = safe_delta * step_size * e

                if self.weight_decay > 0.0:
                    p.data.mul_(1.0-self.eta*self.weight_decay)

                p.data.add_(delta_w, alpha=self.eta)

                if reset:
                    e.zero_()

        info = {'step_size':float(step_size), 'norm_grad':float(norm_grad), 'z_sum':float(z_sum), 'delta':float(delta), 'safe_delta':float(safe_delta), 'abs_delta':float(abs(delta))}
        return info



class OboMetaZero(): 
    def __init__(self, network, role, gamma=0.99, lamda=0.0, kappa=2.0, entropy_coeff=0.01, delta_clip='none', delta_norm='none', beta2=0.999,  rmspower=2.0, entrywise_normalization='none', 
            weight_decay=0.0, momentum=0.0, meta_stepsize=1e-3, beta2_meta=0.999, stepsize_parameterization='exp', epsilon_meta=1e-3, meta_loss_type='none', meta_shadow_dist_reg=0.0, clip_zeta_meta='none'):
        self.opt_type = 'OboMetaZero'
        self.role = role

        self.net = network
        self.net_shadow = deepcopy(network)

        self.optimizer =        OboBase(self.net.parameters(), gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm, beta2=beta2, rmspower=rmspower, entrywise_normalization=entrywise_normalization, weight_decay=weight_decay, momentum=momentum)
        self.optimizer_shadow = OboBase(self.net_shadow.parameters(), gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm, beta2=beta2, rmspower=rmspower, entrywise_normalization=entrywise_normalization, weight_decay=weight_decay, momentum=momentum)  

        self.gamma = gamma
        self.entropy_coeff = entropy_coeff if role=='policy' else None
        self.meta_stepsize = meta_stepsize
        self.beta2_meta = beta2_meta
        self.epsilon_meta = epsilon_meta
        self.stepsize_parameterization = activation_function(stepsize_parameterization)
        self.shadow_distance_regulizer_coeff = meta_shadow_dist_reg

        self.update_meta_only_at_the_end_of_episodes = False
        if role=='critic':
            self.error_critic_main = error_critic(meta_loss_type=meta_loss_type, gamma=gamma)
            self.error_critic_shadow = error_critic(meta_loss_type=meta_loss_type, gamma=gamma)
            if meta_loss_type.split('__epEndOnly_')[-1].split('__')[0] in ['True', 'true', '1']:
                self.update_meta_only_at_the_end_of_episodes = True

        self.eta = torch.tensor(1.0/kappa)
        self.zeta = activation_function_inverse(stepsize_parameterization, 1/kappa)
        self.v_meta = 1.0
        self.t_meta = 1
        self.aggregate_beta2_meta = 1.0
        self.end_time_of_last_episode = 0
        self.clip_zeta_meta = clip_zeta_meta_function(clip_zeta_meta)

    def policy_calculations(self, local_net, s, a, delta, importance_sampling=False, target_policy_prob=None):
        mu, std = local_net(s)
        dist = Normal(mu, std)

        log_prob_pi = (dist.log_prob(a)).sum()
        prob_pi = torch.exp(log_prob_pi)
        if importance_sampling and target_policy_prob is not None:
            IS_ = prob_pi.item()/target_policy_prob
        else:            
            IS_ = 1.0
        pure_entropy = dist.entropy().sum()
        entropy_pi = self.entropy_coeff * dist.entropy().sum() * torch.sign(torch.tensor(delta)).item()
        local_net.zero_grad()
        (IS_*log_prob_pi + entropy_pi).backward()

        return prob_pi.item(), pure_entropy.item()

    def step(self, s, a, r, s_prime, terminated_mask_t, v_s, v_prime, delta, reset):
        if self.role == 'critic':
            self.net.zero_grad()
            v_s.backward()

            with torch.no_grad():
                v_prime_shadow = self.net_shadow(s_prime)
            v_s_shadow = self.net_shadow(s)
            with torch.no_grad():
                td_target_critic_shadow = r + self.gamma * v_prime_shadow * terminated_mask_t
                delta_shadow = (td_target_critic_shadow - v_s_shadow).item()
            self.net_shadow.zero_grad()
            v_s_shadow.backward()

        elif self.role == 'policy':
            delta_shadow = delta
            prob_pi, entropy_pi = self.policy_calculations(self.net, s, a, delta, importance_sampling=False)
            prob_pi_shadow, entropy_pi_shadow = self.policy_calculations(self.net_shadow, s, a, delta_shadow, importance_sampling=True, target_policy_prob=prob_pi)
            
            importance_sampling = prob_pi_shadow/prob_pi
            
        
        info = self.optimizer.step(delta, reset=reset)
        info_shadow = self.optimizer_shadow.step(delta_shadow, reset=reset)

        # meta update
        if self.role == 'critic':
            error_main = self.error_critic_main.step(v_s.item(), v_prime.item(), r, v_prime.item(), reset)
            error_shadow = self.error_critic_shadow.step(v_s_shadow.item(), v_prime_shadow.item(), r, v_prime.item(), reset)
        elif self.role == 'policy':
            delta_normalized = self.optimizer.delta_normalized
            error_main = -(delta_normalized + self.entropy_coeff*entropy_pi)
            error_shadow = -(importance_sampling*delta_normalized +  self.entropy_coeff*entropy_pi_shadow)

        if reset or not self.update_meta_only_at_the_end_of_episodes:
            self.t_meta += 1
            meta_grad = (error_shadow-error_main)/self.epsilon_meta
            self.v_meta =  self.v_meta + ((1-self.beta2_meta)/(1-self.beta2_meta**(self.t_meta-1))) * (meta_grad**2 - self.v_meta) 
            self.zeta = self.zeta - self.meta_stepsize * meta_grad / (math.sqrt(self.v_meta) + 1e-8)
            self.zeta = self.clip_zeta_meta(self.zeta)

        # # time stretching implementation of episodic critic update
        # self.t_meta += 1
        # if reset and self.update_meta_only_at_the_end_of_episodes:
        #     stretch_exponent = 2.0/3.0    # episode_length^stretch_exponent.  The logic is as follows: in case of brownian motion, exponent should be 0.5, and in case of linear growth, it should be 1.0. We use an exponent in between because we do not know the pattern in advance.
        #     length_of_current_episode = self.t_meta - self.end_time_of_last_episode
        #     stretched_ep_len = length_of_current_episode ** stretch_exponent
        #     self.end_time_of_last_episode = self.t_meta + 0

        #     stretched_beta2_meta = self.beta2_meta ** stretched_ep_len
        #     stretched_meta_stepsize = self.meta_stepsize * stretched_ep_len
        #     self.aggregate_beta2_meta *= stretched_beta2_meta

        #     meta_grad = (error_shadow-error_main)/self.epsilon_meta
        #     self.v_meta =  self.v_meta + ((1-stretched_beta2_meta)/(1-self.aggregate_beta2_meta)) * (meta_grad**2 - self.v_meta) 
        #     self.zeta = self.zeta - stretched_meta_stepsize * meta_grad / (math.sqrt(self.v_meta) + 1e-8)
        #     self.zeta = self.clip_zeta_meta(self.zeta)

        # elif not self.update_meta_only_at_the_end_of_episodes:
        #     meta_grad = (error_shadow-error_main)/self.epsilon_meta
        #     self.v_meta =  self.v_meta + ((1-self.beta2_meta)/(1-self.beta2_meta**(self.t_meta-1))) * (meta_grad**2 - self.v_meta) 
        #     self.zeta = self.zeta - self.meta_stepsize * meta_grad / (math.sqrt(self.v_meta) + 1e-8)
        #     self.zeta = self.clip_zeta_meta(self.zeta)


        self.eta = self.stepsize_parameterization(self.zeta)
        self.optimizer.eta = self.eta
        self.optimizer_shadow.eta = self.stepsize_parameterization(self.zeta + self.epsilon_meta)
        
        if self.shadow_distance_regulizer_coeff>0: # regularization to pull shadow netwrok toward main network
            with torch.no_grad():
                for param, param_shadow in zip(self.net.parameters(), self.net_shadow.parameters()):
                    param_shadow.add_(param.data - param_shadow.data, alpha=self.shadow_distance_regulizer_coeff)
            
        
        return {**info, **{f'{key}_shadow':info_shadow[key] for key in info_shadow}}


    



class OboMetaOpt(torch.optim.Optimizer): 
    def __init__(self, params, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', beta2=0.999,  rmspower=2.0, entrywise_normalization='none', 
                 weight_decay=0.0, momentum=0.0, meta_stepsize=1e-4, beta2_meta=0.999, stepsize_parameterization='exp', h_decay_meta=0.9999, clip_zeta_meta='none'):
        defaults = dict(gamma=gamma, lamda=lamda, beta2=beta2, rmspower=rmspower, beta_momentum=momentum)
        self.gamma = gamma
        self.lamda = lamda
        self.weight_decay = weight_decay
        self.u_bar = 0.0
        self.sigma = 0.0
        self.t_val = 0
        self.delta_norm=delta_norm
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.delta_clipper = delta_clipper(clip_type=delta_clip, normalization_type=delta_norm)
        # Meta parameters:
        self.meta_stepsize = meta_stepsize
        self.beta2_meta = beta2_meta
        self.h_decay_meta = h_decay_meta
        self.stepsize_parameterization = activation_function(stepsize_parameterization)
        self.zeta = activation_function_inverse(stepsize_parameterization, 1/kappa)
        self.v_meta = 1.0
        self.clip_zeta_meta = clip_zeta_meta_function(clip_zeta_meta)
        super(OboMetaOpt, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        safe_delta = self.delta_clipper.clip_and_norm(delta)
        self.t_val += 1

        # meta update
        if self.t_val>=2:
            meta_grad = 0.0
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    h = state["h_meta"]
                    meta_grad += safe_delta * ((p.grad * h).sum()).abs().item()
            self.v_meta =  self.v_meta + ((1-self.beta2_meta)/(1-self.beta2_meta**(self.t_val-1))) * (meta_grad**2 - self.v_meta) 
            self.zeta = self.zeta - self.meta_stepsize * meta_grad / (math.sqrt(self.v_meta) + 1e-8)
            self.zeta = self.clip_zeta_meta(self.zeta)
        
        eta = self.stepsize_parameterization(self.zeta)
        self.eta = eta

        # base update
        norm_grad = 0
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                
                if self.entrywise_normalization.lower() in ['rmsprop']:
                    v.mul_(group["beta2"]).add_(torch.pow(torch.abs(p.grad), group["rmspower"]), alpha=1.0 - group["beta2"])
                    v_hat = torch.pow(v / (1.0 - group["beta2"] ** self.t_val), 1.0/group["rmspower"]) + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad)
                z_sum += ((e.square() / v_hat).sum()).abs().item()
                norm_grad += ((p.grad.square() / v_hat).sum()).abs().item()
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad-self.sigma)
        norm_normalizer = math.sqrt((self.sigma/(1-(self.gamma*self.lamda)**self.t_val)) + 1e-12) 
        z_normalizer = math.sqrt(z_sum/(1-(self.gamma*self.lamda)**self.t_val))
        normalizer_ = norm_normalizer * z_normalizer
        step_size = 1 / normalizer_

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                if group["beta_momentum"]>0:
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p.data)
                    m = state["momentum"]
                    if self.entrywise_normalization.lower() == 'rmsprop':
                        m.mul_(group["beta_momentum"]).addcdiv_(e, state["rmsprop_v_hat"], value= safe_delta * step_size * (1-group["beta_momentum"]))
                    else:
                        m.mul_(group["beta_momentum"]).add_(e, alpha= safe_delta * step_size * (1-group["beta_momentum"]))
                    delta_w = m
                else:
                    if self.entrywise_normalization.lower() == 'rmsprop':
                        delta_w = safe_delta * step_size * e / state["rmsprop_v_hat"]
                    else:
                        delta_w = safe_delta * step_size * e

                if "h_meta" not in state:
                    state["h_meta"] = torch.zeros_like(p.data)
                h = state["h_meta"]
                h.mul_(self.h_decay_meta)

                if self.weight_decay > 0.0:
                    p.data.mul_(1.0-eta*self.weight_decay)
                    h.mul_(1.0-eta*self.weight_decay)

                p.data.add_(delta_w, alpha=eta)
                h.data.add_(delta_w)

                if reset:
                    e.zero_()

        info = {'clipped_step_size':float(step_size), 'delta':float(delta), 'delta_used':float(safe_delta), 'abs_delta':float(abs(delta)), 'norm2_eligibility_trace':float(z_sum)}
        return info


class OboBase(torch.optim.Optimizer): 
    def __init__(self, params, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', beta2=0.999, rmspower=2.0, entrywise_normalization='none', 
                 weight_decay=0.0, momentum=0.0):
        defaults = dict(gamma=gamma, lamda=lamda, beta2=beta2, beta_momentum=momentum,  rmspower=rmspower)
        self.gamma = gamma
        self.lamda = lamda
        self.eta = torch.tensor(1.0/kappa)
        self.weight_decay = weight_decay
        self.u_bar = 0.0
        self.sigma = 0.0
        self.t_val = 0
        self.delta_norm=delta_norm
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.delta_clipper = delta_clipper(clip_type=delta_clip, normalization_type=delta_norm)
        # Meta parameters:
        self.v_meta = 1.0
        super(OboBase, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        safe_delta = self.delta_clipper.clip_and_norm(delta)
        self.delta_normalized = safe_delta
        self.t_val += 1

        # base update
        norm_grad = 0
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                
                if self.entrywise_normalization.lower() in ['rmsprop']:
                    v.mul_(group["beta2"]).add_(torch.pow(torch.abs(p.grad), group["rmspower"]), alpha=1.0 - group["beta2"])
                    v_hat = torch.pow(v / (1.0 - group["beta2"] ** self.t_val), 1.0/group["rmspower"]) + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad)
                z_sum += ((e.square() / v_hat).sum()).abs().item()
                norm_grad += ((p.grad.square() / v_hat).sum()).abs().item()
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad-self.sigma)
        norm_normalizer = math.sqrt((self.sigma/(1-(self.gamma*self.lamda)**self.t_val)) + 1e-12) 
        z_normalizer = math.sqrt(z_sum/(1-(self.gamma*self.lamda)**self.t_val))
        normalizer_ = norm_normalizer * z_normalizer
        step_size = 1 / normalizer_

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                if group["beta_momentum"]>0:
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p.data)
                    m = state["momentum"]
                    if self.entrywise_normalization.lower() == 'rmsprop':
                        m.mul_(group["beta_momentum"]).addcdiv_(e, state["rmsprop_v_hat"], value= safe_delta * step_size * (1-group["beta_momentum"]))
                    else:
                        m.mul_(group["beta_momentum"]).add_(e, alpha= safe_delta * step_size * (1-group["beta_momentum"]))
                    delta_w = m
                else:
                    if self.entrywise_normalization.lower() == 'rmsprop':
                        delta_w = safe_delta * step_size * e / state["rmsprop_v_hat"]
                    else:
                        delta_w = safe_delta * step_size * e

                if self.weight_decay > 0.0:
                    p.data.mul_(1.0-self.eta*self.weight_decay)

                p.data.add_(delta_w, alpha=self.eta)

                if reset:
                    e.zero_()

        info = {'clipped_step_size':float(step_size), 'delta':float(delta), 'delta_used':float(safe_delta), 'abs_delta':float(abs(delta)), 'norm2_eligibility_trace':float(z_sum)}
        return info





class OboMeta_old(torch.optim.Optimizer): 
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', beta2=0.999, entrywise_normalization='none', 
                 sig_power=2, in_trace_sample_scaling=True, weight_decay=0.0, momentum=0.0, u_trace=1.0,
                 meta_stepsize=1e-4, beta2_meta=0.999, stepsize_parameterization='sigmoid', h_decay_meta=0.9999, clip_zeta_meta='none'):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, beta2=beta2, beta_momentum=momentum)
        self.gamma = gamma
        self.lamda = lamda
        self.weight_decay = weight_decay
        self.sig_power = sig_power
        self.u_trace = u_trace
        self.u_bar = 0.0
        self.sigma = 0.0
        self.t_val = 0
        self.delta_norm=delta_norm
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.in_trace_sample_scaling = in_trace_sample_scaling  # if True, z accumulates grads scaled down by their norms. This can be applied with or without entrywise_normalization.
        self.delta_clipper = delta_clipper(clip_type=delta_clip, normalization_type=delta_norm)
        # Meta parameters:
        self.meta_stepsize = meta_stepsize
        self.beta2_meta = beta2_meta
        self.h_decay_meta = h_decay_meta
        self.stepsize_parameterization = activation_function(stepsize_parameterization)
        self.zeta = activation_function_inverse(stepsize_parameterization, 1/kappa)
        self.v_meta = 1.0
        self.clip_zeta_meta = clip_zeta_meta_function(clip_zeta_meta)
        super(OboMeta_old, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        safe_delta = self.delta_clipper.clip_and_norm(delta)

        self.t_val += 1
        if self.t_val>=2:
            meta_grad = 0.0
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    h = state["h_meta"]
                    meta_grad += ((p.grad * h).sum()).abs().item()*safe_delta
            self.v_meta =  self.v_meta + ((1-self.beta2_meta)/(1-self.beta2_meta**(self.t_val-1))) * (meta_grad**2 - self.v_meta) 
            self.zeta = self.zeta + self.meta_stepsize * meta_grad / (math.sqrt(self.v_meta) + 1e-8)
            #if not self.delta_norm=='none': print(meta_grad / (math.sqrt(self.v_meta) + 1e-8))
            self.zeta = self.clip_zeta_meta(self.zeta)
        
        eta = self.stepsize_parameterization(self.zeta)
        self.eta = eta

        norm_grad = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() in ['rmsprop']:
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    v_hat = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                norm_grad += ((p.grad.square() / v_hat).sum()).abs().item()
        

        if self.in_trace_sample_scaling in [True, 'True', 'true', 1, '1']:
            scale_of_sample_in_trace = 1.0/norm_grad
        else:
            scale_of_sample_in_trace = 1.0

        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() == 'rmsprop':
                    v_hat_for_trace_norm = state["rmsprop_v_hat"] 
                elif self.entrywise_normalization.lower() == 'none':
                    v_hat_for_trace_norm = 1.0
                else:
                    raise(ValueError(f'     {self.entrywise_normalization}     not supported'))
                
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=scale_of_sample_in_trace)
                z_sum += ((e.square() / v_hat_for_trace_norm).sum()).abs().item()
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad**(self.sig_power/2.0)-self.sigma)
        norm_normalizer = ((self.sigma/(1-(self.gamma*self.lamda)**self.t_val)) + 1e-12) ** (1.0/self.sig_power)
        z_normalizer = math.sqrt(z_sum/(1-(self.gamma*self.lamda)**self.t_val))
        u = norm_normalizer * z_normalizer
        self.u_bar +=  self.u_trace * (u-self.u_bar)
        normalizer_ = max(u, math.sqrt(u*self.u_bar/(1-(1-self.u_trace)**self.t_val)))
        dot_product =  normalizer_
        step_size = 1 / dot_product
        #print(abs(safe_delta), safe_delta/delta)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p.data)
                if "h_meta" not in state:
                    state["h_meta"] = torch.zeros_like(p.data)
                m = state["momentum"]
                h = state["h_meta"]

                m.mul_(group["beta_momentum"]).add_(e, alpha=-safe_delta*eta*step_size)
                h.mul_(self.h_decay_meta)
                if self.weight_decay > 0.0:
                    p.data.mul_(1.0-eta*self.weight_decay)
                    h.mul_(1.0-eta*self.weight_decay)
                if self.entrywise_normalization.lower() == 'rmsprop':
                    p.data.addcdiv_(m, state["rmsprop_v_hat"], value=1-group["beta_momentum"])
                    h.data.addcdiv_(m, state["rmsprop_v_hat"], value=1-group["beta_momentum"])
                elif self.entrywise_normalization.lower() == 'none':
                    p.data.add_(m, alpha=1-group["beta_momentum"])
                    h.data.add_(m, alpha=1-group["beta_momentum"])
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'delta_used':safe_delta, 'abs_delta':abs(delta), 'norm1_eligibility_trace':z_sum}
        return info




class Obo(torch.optim.Optimizer): # same as Obn but also has delta_clipping
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', beta2=0.999, entrywise_normalization='none', sig_power=2, in_trace_sample_scaling=True, weight_decay=0.0, momentum=0.0, u_trace=1.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, beta2=beta2, beta_momentum=momentum)
        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.weight_decay = weight_decay
        self.sig_power = sig_power
        self.u_trace = u_trace
        self.u_bar = 0.0
        self.sigma = 0.0
        self.t_val = 0
        self.eta=torch.tensor(1/kappa)
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.in_trace_sample_scaling = in_trace_sample_scaling  # if True, z accumulates grads scaled down by their norms. This can be applied with or without entrywise_normalization.
        self.delta_clipper = delta_clipper(clip_type=delta_clip, normalization_type=delta_norm)
        super(Obo, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        self.t_val += 1
        norm_grad = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() in ['rmsprop']:
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    v_hat = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                norm_grad += ((p.grad.square() / v_hat).sum()).abs().item()
        

        if self.in_trace_sample_scaling in [True, 'True', 'true', 1, '1']:
            scale_of_sample_in_trace = 1.0/norm_grad
        else:
            scale_of_sample_in_trace = 1.0

        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() == 'rmsprop':
                    v_hat_for_trace_norm = state["rmsprop_v_hat"] 
                elif self.entrywise_normalization.lower() == 'none':
                    v_hat_for_trace_norm = 1.0
                else:
                    raise(ValueError(f'     {self.entrywise_normalization}     not supported'))
                
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=scale_of_sample_in_trace)
                z_sum += ((e.square() / v_hat_for_trace_norm).sum()).abs().item()
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad**(self.sig_power/2.0)-self.sigma)
        norm_normalizer = ((self.sigma/(1-(self.gamma*self.lamda)**self.t_val)) + 1e-12) ** (1.0/self.sig_power)
        z_normalizer = math.sqrt(z_sum/(1-(self.gamma*self.lamda)**self.t_val))
        u = norm_normalizer * z_normalizer
        self.u_bar +=  self.u_trace * (u-self.u_bar)
        normalizer_ = max(u, math.sqrt(u*self.u_bar/(1-(1-self.u_trace)**self.t_val)))
        dot_product =  self.kappa * normalizer_
        step_size = 1 / dot_product
        safe_delta = self.delta_clipper.clip_and_norm(delta)
        #print(abs(safe_delta), safe_delta/delta)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p.data)
                m = state["momentum"]
                m.mul_(group["beta_momentum"]).add_(e, alpha=-safe_delta*step_size)
                if self.weight_decay > 0.0:
                    p.data.mul_(1.0-self.weight_decay/self.kappa)
                if self.entrywise_normalization.lower() == 'rmsprop':
                    p.data.addcdiv_(m, state["rmsprop_v_hat"], value=1-group["beta_momentum"])
                elif self.entrywise_normalization.lower() == 'none':
                    p.data.add_(m, alpha=1-group["beta_momentum"])
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'delta_used':safe_delta, 'abs_delta':abs(delta), 'norm1_eligibility_trace':z_sum}
        return info

class Obonz(torch.optim.Optimizer): # Ob with trace and no norm z
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', beta2=0.999, entrywise_normalization='none', u_trace=0.01, weight_decay=0.0, momentum=0.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, beta2=beta2, beta_momentum=momentum)
        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.weight_decay = weight_decay
        self.u_trace = u_trace
        self.u_bar = 0.0
        self.t_val = 0
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.delta_clipper = delta_clipper(clip_type=delta_clip, normalization_type=delta_norm)
        super(Obonz, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        self.t_val += 1
        norm_grad = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() in ['rmsprop']:
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    v_hat = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                norm_grad += ((p.grad.square()/ v_hat).sum()).abs().item()
        

        self.u_bar +=  self.u_trace * (norm_grad-self.u_bar)
        norm_normalizer = max(norm_grad, math.sqrt(norm_grad*self.u_bar/(1-(1-self.u_trace)**self.t_val)))
        
        scale_of_sample_in_trace = 1.0 / norm_normalizer
        safe_delta = self.delta_clipper.clip_and_norm(delta)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p.data)
                m = state["momentum"]
                if self.entrywise_normalization.lower() == 'rmsprop':
                    v_hat = state["rmsprop_v_hat"] 
                
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=scale_of_sample_in_trace)
                
                p.data.mul_(1.0-self.weight_decay/self.kappa)
                m.mul_(group["beta_momentum"]).add_(e, alpha=-safe_delta/self.kappa)
                if self.entrywise_normalization.lower() == 'rmsprop':
                    p.data.addcdiv_(m, v_hat, value=1-group["beta_momentum"])
                elif self.entrywise_normalization.lower() in ['rmspropintrace', 'none']:
                    p.data.add_(m, alpha=1-group["beta_momentum"])
                if reset:
                    e.zero_()

        info = {'delta':delta, 'delta_used':safe_delta, 'abs_delta':abs(delta)}
        return info
    


class OboC(torch.optim.Optimizer): # same as Obn but also has delta_clipping
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0,  beta2=0.999, entrywise_normalization='none', sig_power=2, in_trace_sample_scaling=True, weight_decay=0.0, momentum=0.0, u_trace=1.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, beta2=beta2, beta_momentum=momentum)
        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.sigma = 0.0
        self.u_trace = u_trace
        self.u_bar = 0.0
        self.weight_decay = weight_decay
        self.sig_power = sig_power
        self.t_val = 0
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.in_trace_sample_scaling = in_trace_sample_scaling  # if True, z accumulates grads scaled down by their norms. This can be applied with or without entrywise_normalization.
        super(OboC, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        self.t_val += 1
        norm_grad = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() in ['rmsprop']:
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    v_hat = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                norm_grad += ((p.grad.square()/ v_hat).sum()).abs().item()
        

        if self.in_trace_sample_scaling in [True, 'True', 'true', 1, '1']:
            scale_of_sample_in_trace = 1.0/norm_grad
        else:
            scale_of_sample_in_trace = 1.0

        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() == 'rmsprop':
                    v_hat_for_trace_norm = state["rmsprop_v_hat"] 
                elif self.entrywise_normalization.lower() == 'none':
                    v_hat_for_trace_norm = 1.0
                else:
                    raise(ValueError(f'     {self.entrywise_normalization}     not supported'))
                
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=scale_of_sample_in_trace)
                z_sum += ((e.square() / v_hat_for_trace_norm).sum()).abs().item()
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad**(self.sig_power/2.0)-self.sigma)
        norm_normalizer = ((self.sigma/(1-(self.gamma*self.lamda)**self.t_val)) + 1e-16) ** (1.0/self.sig_power)
        z_normalizer = math.sqrt(z_sum/(1-(self.gamma*self.lamda)**self.t_val))
        u = norm_normalizer * z_normalizer
        self.u_bar +=  self.u_trace * (u-self.u_bar)
        normalizer_ = max(u, math.sqrt(u*self.u_bar/(1-(1-self.u_trace)**self.t_val)))
        delta_bar = max(abs(delta), 1.0)
        #delta_bar = 1.0
        dot_product = delta_bar * self.kappa * normalizer_
        step_size = 1 / dot_product

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p.data)
                m = state["momentum"]
                m.mul_(group["beta_momentum"]).add_(e, alpha=-delta*step_size)
                if self.weight_decay > 0.0:
                    p.data.mul_(1.0-self.weight_decay/self.kappa)
                # if self.entrywise_normalization.lower() == 'rmsprop':
                #     p.data.addcdiv_(e, state["rmsprop_v_hat"], value=-delta*step_size)
                # elif self.entrywise_normalization.lower() in ['rmspropintrace', 'none']:
                #     p.data.add_(e, alpha=-delta*step_size)
                if self.entrywise_normalization.lower() == 'rmsprop':
                    p.data.addcdiv_(m, state["rmsprop_v_hat"], value=1-group["beta_momentum"])
                elif self.entrywise_normalization.lower() == 'none':
                    p.data.add_(m, alpha=1-group["beta_momentum"])
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'abs_delta':abs(delta), 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info


class Obtm(torch.optim.Optimizer): # same as Obn but also has delta_clipping
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', beta2=0.999, entrywise_normalization='none', sig_power=2, in_trace_sample_scaling=True, weight_decay=0.0, momentum=0.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, beta2=beta2, beta_momentum=momentum)
        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.weight_decay = weight_decay
        self.sig_power = sig_power
        self.sigma = 0.0
        self.t_val = 0
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.in_trace_sample_scaling = in_trace_sample_scaling  # if True, z accumulates grads scaled down by their norms. This can be applied with or without entrywise_normalization.
        self.delta_clipper = delta_clipper(clip_type=delta_clip, normalization_type=delta_norm)
        super(Obtm, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        self.t_val += 1
        norm_grad = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() in ['rmsprop', 'rmspropintrace']:
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    v_hat = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                norm_grad += ((p.grad.square()* v_hat).sum()).abs().item()
        

        if self.in_trace_sample_scaling in [True, 'True', 'true', 1, '1']:
            scale_of_sample_in_trace = 1.0/norm_grad
        else:
            scale_of_sample_in_trace = 1.0

        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower()=='rmspropintrace':
                    v_hat_in_trace = state["rmsprop_v_hat"] 
                    v_hat_for_trace_norm = 1.0
                elif self.entrywise_normalization.lower() == 'rmsprop':
                    v_hat_in_trace = torch.ones_like(p.data)
                    v_hat_for_trace_norm = state["rmsprop_v_hat"] 
                elif self.entrywise_normalization.lower() == 'none':
                    v_hat_in_trace = torch.ones_like(p.data)
                    v_hat_for_trace_norm = 1.0
                else:
                    raise(ValueError(f'     {self.entrywise_normalization}     not supported'))
                
                e.mul_(group["gamma"] * group["lamda"]).addcdiv_(p.grad, v_hat_in_trace, value=scale_of_sample_in_trace)
                z_sum += ((e.square()* v_hat_for_trace_norm).sum()).abs().item()
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad**(self.sig_power/2.0)-self.sigma)
        norm_normalizer = ((self.sigma/(1-(self.gamma*self.lamda)**self.t_val)) + 1e-16) ** (1.0/self.sig_power)
        z_normalizer = math.sqrt(z_sum/(1-(self.gamma*self.lamda)**self.t_val))
        dot_product =  self.kappa * norm_normalizer * z_normalizer
        step_size = 1 / dot_product
        safe_delta = self.delta_clipper.clip_and_norm(delta)
        #print(delta, '\t', safe_delta)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p.data)
                m = state["momentum"]
                m.mul_(group["beta_momentum"]).add_(e, alpha=-safe_delta*step_size)
                p.data.mul_(1.0-self.weight_decay/self.kappa)
                if self.entrywise_normalization.lower() == 'rmsprop':
                #   p.data.addcdiv_(e, state["rmsprop_v_hat"], value=-safe_delta*step_size)
                    p.data.addcdiv_(m, state["rmsprop_v_hat"], value=1-group["beta_momentum"])
                elif self.entrywise_normalization.lower() in ['rmspropintrace', 'none']:
                #   p.data.add_(e, alpha=-safe_delta*step_size)
                    p.data.add_(m, alpha=1-group["beta_momentum"])
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'delta_used':safe_delta, 'abs_delta':abs(delta), 'norm1_eligibility_trace':z_sum}
        return info

class Obtnnzm(torch.optim.Optimizer): # Ob with trace and no norm z
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', beta2=0.999, entrywise_normalization='none', u_trace=0.01, weight_decay=0.0, momentum=0.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, beta2=beta2, beta_momentum=momentum)
        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.weight_decay = weight_decay
        self.u_trace = u_trace
        self.u_bar = 0.0
        self.t_val = 0
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.delta_clipper = delta_clipper(clip_type=delta_clip, normalization_type=delta_norm)
        super(Obtnnzm, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        self.t_val += 1
        norm_grad = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() in ['rmsprop', 'rmspropintrace']:
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    v_hat = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                norm_grad += ((p.grad.square()* v_hat).sum()).abs().item()
        

        self.u_bar +=  self.u_trace * (norm_grad-self.u_bar)
        norm_normalizer = max(norm_grad, math.sqrt(norm_grad*self.u_bar/(1-(1-self.u_trace)**self.t_val)))
        
        scale_of_sample_in_trace = 1.0 / norm_normalizer
        safe_delta = self.delta_clipper.clip_and_norm(delta)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p.data)
                m = state["momentum"]
                if self.entrywise_normalization.lower() == 'rmsprop':
                    v_hat = state["rmsprop_v_hat"] 
                
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=scale_of_sample_in_trace)
                
                p.data.mul_(1.0-self.weight_decay/self.kappa)
                m.mul_(group["beta_momentum"]).add_(e, alpha=-safe_delta/self.kappa)
                if self.entrywise_normalization.lower() == 'rmsprop':
                    #p.data.addcdiv_(e, v_hat, value=-safe_delta/self.kappa)
                    p.data.addcdiv_(m, v_hat, value=1-group["beta_momentum"])
                elif self.entrywise_normalization.lower() in ['rmspropintrace', 'none']:
                    #p.data.add_(e, alpha=-safe_delta/self.kappa)
                    p.data.add_(m, alpha=1-group["beta_momentum"])
                if reset:
                    e.zero_()

        info = {'delta':delta, 'delta_used':safe_delta, 'abs_delta':abs(delta)}
        return info
    


class Obtnnz(torch.optim.Optimizer): # Ob with trace and no norm z
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', beta2=0.999, entrywise_normalization='none', u_trace=0.01, weight_decay=0.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, beta2=beta2)
        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.weight_decay = weight_decay
        self.u_trace = u_trace
        self.u_bar = 0.0
        self.t_val = 0
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.delta_clipper = delta_clipper(clip_type=delta_clip, normalization_type=delta_norm)
        super(Obtnnz, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        self.t_val += 1
        norm_grad = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() in ['rmsprop', 'rmspropintrace']:
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    v_hat = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                norm_grad += ((p.grad.square()* v_hat).sum()).abs().item()
        

        self.u_bar +=  self.u_trace * (norm_grad-self.u_bar)
        norm_normalizer = max(norm_grad, math.sqrt(norm_grad*self.u_bar/(1-(1-self.u_trace)**self.t_val)))
        
        scale_of_sample_in_trace = 1.0 / norm_normalizer
        safe_delta = self.delta_clipper.clip_and_norm(delta)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() == 'rmsprop':
                    v_hat = state["rmsprop_v_hat"] 
                
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=scale_of_sample_in_trace)
                
                p.data.mul_(1.0-self.weight_decay/self.kappa)
                if self.entrywise_normalization.lower() == 'rmsprop':
                    p.data.addcdiv_(e, v_hat, value=-safe_delta/self.kappa)
                elif self.entrywise_normalization.lower() in ['rmspropintrace', 'none']:
                    p.data.add_(e, alpha=-safe_delta/self.kappa)
                if reset:
                    e.zero_()
        
                

        info = {'delta':delta, 'delta_used':safe_delta, 'abs_delta':abs(delta)}
        return info
    


class Obt(torch.optim.Optimizer): # same as Obn but also has delta_clipping
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', beta2=0.999, entrywise_normalization='none', sig_power=2, in_trace_sample_scaling=True, weight_decay=0.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, beta2=beta2)
        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.weight_decay = weight_decay
        self.sig_power = sig_power
        self.sigma = 0.0
        self.t_val = 0
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.in_trace_sample_scaling = in_trace_sample_scaling  # if True, z accumulates grads scaled down by their norms. This can be applied with or without entrywise_normalization.
        self.delta_clipper = delta_clipper(clip_type=delta_clip, normalization_type=delta_norm)
        super(Obt, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        self.t_val += 1
        norm_grad = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() in ['rmsprop', 'rmspropintrace']:
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    v_hat = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                norm_grad += ((p.grad.square()* v_hat).sum()).abs().item()
        

        if self.in_trace_sample_scaling in [True, 'True', 'true', 1, '1']:
            scale_of_sample_in_trace = 1.0/norm_grad
        else:
            scale_of_sample_in_trace = 1.0

        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower()=='rmspropintrace':
                    v_hat_in_trace = state["rmsprop_v_hat"] 
                    v_hat_for_trace_norm = 1.0
                elif self.entrywise_normalization.lower() == 'rmsprop':
                    v_hat_in_trace = torch.ones_like(p.data)
                    v_hat_for_trace_norm = state["rmsprop_v_hat"] 
                elif self.entrywise_normalization.lower() == 'none':
                    v_hat_in_trace = torch.ones_like(p.data)
                    v_hat_for_trace_norm = 1.0
                else:
                    raise(ValueError(f'     {self.entrywise_normalization}     not supported'))
                
                e.mul_(group["gamma"] * group["lamda"]).addcdiv_(p.grad, v_hat_in_trace, value=scale_of_sample_in_trace)
                z_sum += ((e.square()* v_hat_for_trace_norm).sum()).abs().item()
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad**(self.sig_power/2.0)-self.sigma)
        norm_normalizer = ((self.sigma/(1-(self.gamma*self.lamda)**self.t_val)) + 1e-16) ** (1.0/self.sig_power)
        z_normalizer = math.sqrt(z_sum/(1-(self.gamma*self.lamda)**self.t_val))
        dot_product =  self.kappa * norm_normalizer * z_normalizer
        step_size = 1 / dot_product
        safe_delta = self.delta_clipper.clip_and_norm(delta)
        #print(delta, '\t', safe_delta)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.mul_(1.0-self.weight_decay/self.kappa)
                if self.entrywise_normalization.lower() == 'rmsprop':
                    p.data.addcdiv_(e, state["rmsprop_v_hat"], value=-safe_delta*step_size)
                elif self.entrywise_normalization.lower() in ['rmspropintrace', 'none']:
                    p.data.add_(e, alpha=-safe_delta*step_size)
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'delta_used':safe_delta, 'abs_delta':abs(delta), 'norm1_eligibility_trace':z_sum}
        return info


    

class ObtN(torch.optim.Optimizer): # same as Obn but also has delta_clipping
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, delta_trace=.01, beta2=0.999, entrywise_normalization='none', sig_power=2, in_trace_sample_scaling=True, weight_decay=0.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, beta2=beta2)
        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.weight_decay = weight_decay
        self.sig_power = sig_power
        self.delta_trace = delta_trace
        self.sigma = 0.0
        self.delta_bar = 0
        self.t_val = 0
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.in_trace_sample_scaling = in_trace_sample_scaling  # if True, z accumulates grads scaled down by their norms. This can be applied with or without entrywise_normalization.
        super(ObtN, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        self.t_val += 1
        norm_grad = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() in ['rmsprop', 'rmspropintrace']:
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    v_hat = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                norm_grad += ((p.grad.square()* v_hat).sum()).abs().item()
        

        if self.in_trace_sample_scaling in [True, 'True', 'true', 1, '1']:
            scale_of_sample_in_trace = 1.0/norm_grad
        else:
            scale_of_sample_in_trace = 1.0

        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower()=='rmspropintrace':
                    v_hat_in_trace = state["rmsprop_v_hat"] 
                    v_hat_for_trace_norm = 1.0
                elif self.entrywise_normalization.lower() == 'rmsprop':
                    v_hat_in_trace = torch.ones_like(p.data)
                    v_hat_for_trace_norm = state["rmsprop_v_hat"] 
                elif self.entrywise_normalization.lower() == 'none':
                    v_hat_in_trace = torch.ones_like(p.data)
                    v_hat_for_trace_norm = 1.0
                else:
                    raise(ValueError(f'     {self.entrywise_normalization}     not supported'))
                
                e.mul_(group["gamma"] * group["lamda"]).addcdiv_(p.grad, v_hat_in_trace, value=scale_of_sample_in_trace)
                z_sum += ((e.square()* v_hat_for_trace_norm).sum()).abs().item()
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad**(self.sig_power/2.0)-self.sigma)
        norm_normalizer = ((self.sigma/(1-(self.gamma*self.lamda)**self.t_val)) + 1e-16) ** (1.0/self.sig_power)
        z_normalizer = math.sqrt(z_sum/(1-(self.gamma*self.lamda)**self.t_val))
        self.delta_bar += self.delta_trace * (delta*delta - self.delta_bar)
        delta_bar = math.sqrt(self.delta_bar/(1-(1-self.delta_trace)**self.t_val))
        #delta_bar = max(abs(delta), 1.0)
        #delta_bar = 1.0
        dot_product = delta_bar * self.kappa * norm_normalizer * z_normalizer
        step_size = 1 / dot_product

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.mul_(1.0-self.weight_decay/self.kappa)
                if self.entrywise_normalization.lower() == 'rmsprop':
                    p.data.addcdiv_(e, state["rmsprop_v_hat"], value=-delta*step_size)
                elif self.entrywise_normalization.lower() in ['rmspropintrace', 'none']:
                    p.data.add_(e, alpha=-delta*step_size)
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'delta_used':delta/delta_bar, 'abs_delta':abs(delta), 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info
    

class ObtCm(torch.optim.Optimizer): # same as Obn but also has delta_clipping
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0,  beta2=0.999, entrywise_normalization='none', sig_power=2, in_trace_sample_scaling=True, weight_decay=0.0, momentum=0.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, beta2=beta2, beta_momentum=momentum)
        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.sigma = 0.0
        self.weight_decay = weight_decay
        self.sig_power = sig_power
        self.t_val = 0
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.in_trace_sample_scaling = in_trace_sample_scaling  # if True, z accumulates grads scaled down by their norms. This can be applied with or without entrywise_normalization.
        super(ObtCm, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        self.t_val += 1
        norm_grad = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() in ['rmsprop', 'rmspropintrace']:
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    v_hat = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                norm_grad += ((p.grad.square()* v_hat).sum()).abs().item()
        

        if self.in_trace_sample_scaling in [True, 'True', 'true', 1, '1']:
            scale_of_sample_in_trace = 1.0/norm_grad
        else:
            scale_of_sample_in_trace = 1.0

        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower()=='rmspropintrace':
                    v_hat_in_trace = state["rmsprop_v_hat"] 
                    v_hat_for_trace_norm = 1.0
                elif self.entrywise_normalization.lower() == 'rmsprop':
                    v_hat_in_trace = torch.ones_like(p.data)
                    v_hat_for_trace_norm = state["rmsprop_v_hat"] 
                elif self.entrywise_normalization.lower() == 'none':
                    v_hat_in_trace = torch.ones_like(p.data)
                    v_hat_for_trace_norm = 1.0
                else:
                    raise(ValueError(f'     {self.entrywise_normalization}     not supported'))
                
                e.mul_(group["gamma"] * group["lamda"]).addcdiv_(p.grad, v_hat_in_trace, value=scale_of_sample_in_trace)
                z_sum += ((e.square()* v_hat_for_trace_norm).sum()).abs().item()
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad**(self.sig_power/2.0)-self.sigma)
        norm_normalizer = ((self.sigma/(1-(self.gamma*self.lamda)**self.t_val)) + 1e-16) ** (1.0/self.sig_power)
        z_normalizer = math.sqrt(z_sum/(1-(self.gamma*self.lamda)**self.t_val))
        delta_bar = max(abs(delta), 1.0)
        #delta_bar = 1.0
        dot_product = delta_bar * self.kappa * norm_normalizer * z_normalizer
        step_size = 1 / dot_product

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p.data)
                m = state["momentum"]
                m.mul_(group["beta_momentum"]).add_(e, alpha=-delta*step_size)
                p.data.mul_(1.0-self.weight_decay/self.kappa)
                # if self.entrywise_normalization.lower() == 'rmsprop':
                #     p.data.addcdiv_(e, state["rmsprop_v_hat"], value=-delta*step_size)
                # elif self.entrywise_normalization.lower() in ['rmspropintrace', 'none']:
                #     p.data.add_(e, alpha=-delta*step_size)
                if self.entrywise_normalization.lower() == 'rmsprop':
                    p.data.addcdiv_(m, state["rmsprop_v_hat"], value=1-group["beta_momentum"])
                elif self.entrywise_normalization.lower() in ['rmspropintrace', 'none']:
                    p.data.add_(m, alpha=1-group["beta_momentum"])
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'abs_delta':abs(delta), 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info
    

class ObtC(torch.optim.Optimizer): # same as Obn but also has delta_clipping
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0,  beta2=0.999, entrywise_normalization='none', sig_power=2, in_trace_sample_scaling=True, weight_decay=0.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, beta2=beta2)
        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.sigma = 0.0
        self.weight_decay = weight_decay
        self.sig_power = sig_power
        self.t_val = 0
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop' or 'RMSPropInTrace' (meaning z accumulates entrywised scaled grads)  (default: RMSProp)
        self.in_trace_sample_scaling = in_trace_sample_scaling  # if True, z accumulates grads scaled down by their norms. This can be applied with or without entrywise_normalization.
        super(ObtC, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        self.t_val += 1
        norm_grad = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower() in ['rmsprop', 'rmspropintrace']:
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    v_hat = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0
                norm_grad += ((p.grad.square()* v_hat).sum()).abs().item()
        

        if self.in_trace_sample_scaling in [True, 'True', 'true', 1, '1']:
            scale_of_sample_in_trace = 1.0/norm_grad
        else:
            scale_of_sample_in_trace = 1.0

        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower()=='rmspropintrace':
                    v_hat_in_trace = state["rmsprop_v_hat"] 
                    v_hat_for_trace_norm = 1.0
                elif self.entrywise_normalization.lower() == 'rmsprop':
                    v_hat_in_trace = torch.ones_like(p.data)
                    v_hat_for_trace_norm = state["rmsprop_v_hat"] 
                elif self.entrywise_normalization.lower() == 'none':
                    v_hat_in_trace = torch.ones_like(p.data)
                    v_hat_for_trace_norm = 1.0
                else:
                    raise(ValueError(f'     {self.entrywise_normalization}     not supported'))
                
                e.mul_(group["gamma"] * group["lamda"]).addcdiv_(p.grad, v_hat_in_trace, value=scale_of_sample_in_trace)
                z_sum += ((e.square()* v_hat_for_trace_norm).sum()).abs().item()
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad**(self.sig_power/2.0)-self.sigma)
        norm_normalizer = ((self.sigma/(1-(self.gamma*self.lamda)**self.t_val)) + 1e-16) ** (1.0/self.sig_power)
        z_normalizer = math.sqrt(z_sum/(1-(self.gamma*self.lamda)**self.t_val))
        delta_bar = max(abs(delta), 1.0)
        #delta_bar = 1.0
        dot_product = delta_bar * self.kappa * norm_normalizer * z_normalizer
        step_size = 1 / dot_product

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.mul_(1.0-self.weight_decay/self.kappa)
                if self.entrywise_normalization.lower() == 'rmsprop':
                    p.data.addcdiv_(e, state["rmsprop_v_hat"], value=-delta*step_size)
                elif self.entrywise_normalization.lower() in ['rmspropintrace', 'none']:
                    p.data.add_(e, alpha=-delta*step_size)
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'abs_delta':abs(delta), 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info
    

class ObnN(torch.optim.Optimizer):  # obn with normalized delta
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, delta_trace=.01, u_trace=.01, beta2=0.999, entrywise_normalization='none'):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, beta2=beta2)
        self.u_bar = 0.0
        self.delta_trace = delta_trace
        self.delta_bar = 0.0
        self.u_trace = u_trace
        self.t_val = 0
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop'
        super(ObnN, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        self.t_val += 1
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower()=='rmsprop':
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    sigma = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    e.mul_(group["gamma"] * group["lamda"]).addcdiv_(p.grad, sigma, value=1.0)
                elif self.entrywise_normalization.lower()=='none':
                    e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                    sigma = torch.ones_like(p.data)
                z_sum += ((e.square()* sigma).sum()).abs().item()
        
        self.u_bar +=  self.u_trace * (z_sum-self.u_bar)
        norm_normalizer = max(z_sum, math.sqrt(z_sum*self.u_bar/(1-(1-self.u_trace)**self.t_val)))
        self.delta_bar += self.delta_trace * (delta*delta - self.delta_bar)
        delta_bar = math.sqrt(self.delta_bar/(1-(1-self.delta_trace)**self.t_val))
        dot_product = delta_bar * norm_normalizer * group["lr"] * group["kappa"]
        step_size = group["lr"] / dot_product

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'abs_delta':abs(delta),  'delta_used':delta/delta_bar, 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info

class Obn(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, u_trace=.01, beta2=0.999, entrywise_normalization='none'):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, beta2=beta2)
        self.u_bar = 0.0
        self.u_trace = u_trace
        self.t_val = 0
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop'
        super(Obn, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        self.t_val += 1
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower()=='rmsprop':
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    sigma = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    e.mul_(group["gamma"] * group["lamda"]).addcdiv_(p.grad, sigma, value=1.0)
                elif self.entrywise_normalization.lower()=='none':
                    e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                    sigma = torch.ones_like(p.data)
                z_sum += ((e.square()* sigma).sum()).abs().item()
        
        self.u_bar +=  self.u_trace * (z_sum-self.u_bar)
        norm_normalizer = max(z_sum, math.sqrt(z_sum*self.u_bar/(1-(1-self.u_trace)**self.t_val)))
        #delta_bar = max(abs(delta), 1.0)
        delta_bar = 1.0
        dot_product = delta_bar * norm_normalizer * group["lr"] * group["kappa"]
        step_size = group["lr"] / dot_product

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta,  'delta_used':delta/delta_bar, 'abs_delta':abs(delta), 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info
    

class ObnC(torch.optim.Optimizer): # same as Obn but also has delta_clipping
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, u_trace=.01, beta2=0.999, entrywise_normalization='none'):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, beta2=beta2)
        self.u_bar = 0.0
        self.u_trace = u_trace
        self.t_val = 0
        self.entrywise_normalization = entrywise_normalization # 'none' or 'RMSprop'
        super(ObnC, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        self.t_val += 1
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                e = state["eligibility_trace"]
                v = state["entrywise_squared_grad"]
                if self.entrywise_normalization.lower()=='rmsprop':
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    sigma = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    e.mul_(group["gamma"] * group["lamda"]).addcdiv_(p.grad, sigma, value=1.0)
                elif self.entrywise_normalization.lower()=='none':
                    e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                    sigma = torch.ones_like(p.data)
                z_sum += ((e.square()* sigma).sum()).abs().item()
        
        self.u_bar +=  self.u_trace * (z_sum-self.u_bar)
        norm_normalizer = max(z_sum, math.sqrt(z_sum*self.u_bar/(1-(1-self.u_trace)**self.t_val)))
        delta_bar = max(abs(delta), 1.0)
        #delta_bar = 1.0
        dot_product = delta_bar * norm_normalizer * group["lr"] * group["kappa"]
        step_size = group["lr"] / dot_product

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'abs_delta':abs(delta),  'delta_used':delta/delta_bar, 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info
    

class ObGD_sq_plain(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD_sq_plain, self).__init__(params, defaults)
    
    def step(self, delta, reset=False):
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                z_sum += (e.square().sum()).abs().item()

        delta_bar = 1.0
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        step_size = group["lr"] / dot_product

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta,  'delta_used':delta/delta_bar, 'abs_delta':abs(delta), 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info


class ObGD_sq(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD_sq, self).__init__(params, defaults)
    
    def step(self, delta, reset=False):
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                z_sum += (e.square().sum()).abs().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta,  'delta_used':delta/delta_bar, 'abs_delta':abs(delta), 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info



class ObGDN(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0, delta_clip='none', delta_norm='none'):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        self.delta_clipper = delta_clipper(clip_type=delta_clip, normalization_type=delta_norm)
        super(ObGDN, self).__init__(params, defaults)
    def step(self, delta, reset=False):
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                z_sum += e.abs().sum().item()

        safe_delta = self.delta_clipper.clip_and_norm(delta)
        dot_product = z_sum * group["lr"] * group["kappa"]
        step_size = 1.0/dot_product
        # delta_bar = max(abs(delta), 1.0)
        # dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        # if dot_product > 1:
        #     step_size = group["lr"] / dot_product
        # else:
        #     step_size = group["lr"]
        

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.add_(safe_delta * e, alpha=-step_size)
                #p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta,  'delta_used':safe_delta, 'abs_delta':abs(delta), 'norm1_eligibility_trace':z_sum}
        return info



class ObGDm(torch.optim.Optimizer):   # same as ObGD but adds momentum
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0, momentum=0.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, beta_momentum=momentum)
        super(ObGDm, self).__init__(params, defaults)
    def step(self, delta, reset=False):
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                z_sum += e.abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p.data)
                m = state["momentum"]
                #p.data.add_(delta * e, alpha=-step_size)
                m.mul_(group["beta_momentum"]).add_(e, alpha=-delta*step_size)
                p.data.add_(m, alpha=1-group["beta_momentum"])
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta,  'delta_used':delta/delta_bar, 'abs_delta':abs(delta), 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info
    

##### ----------------------


class ObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD, self).__init__(params, defaults)
    def step(self, delta, reset=False):
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                z_sum += e.abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta,  'delta_used':delta/delta_bar, 'abs_delta':abs(delta), 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info

class AdaptiveObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, beta2=beta2, eps=eps)
        self.counter = 0
        super(AdaptiveObGD, self).__init__(params, defaults)
    def step(self, delta, reset=False):
        z_sum = 0.0
        self.counter += 1
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                e, v = state["eligibility_trace"], state["v"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)

                v.mul_(group["beta2"]).addcmul_(delta*e, delta*e, value=1.0 - group["beta2"])
                v_hat = v / (1.0 - group["beta2"] ** self.counter)
                z_sum += (e / (v_hat + group["eps"]).sqrt()).abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                v, e = state["v"], state["eligibility_trace"]
                v_hat = v / (1.0 - group["beta2"] ** self.counter)
                p.data.addcdiv_(delta * e, (v_hat + group["eps"]).sqrt(), value=-step_size)
                if reset:
                    e.zero_()
