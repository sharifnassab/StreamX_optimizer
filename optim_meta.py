import torch
import math
from copy import deepcopy
from torch.distributions import Normal
from delta_clipper import delta_clipper
from meta_opt_helper import activation_function, activation_function_inverse, clip_zeta_meta_function
from MetaZero_loss_calculation import error_critic




class RMSPropMetaZero2side(): 
    def __init__(self, network, role, gamma=0.99, lamda=0.0, kappa=2.0, entropy_coeff=0.01, delta_clip='none', delta_norm='none', beta2=0.999,  rmspower=2.0, entrywise_normalization='none', 
            weight_decay=0.0, momentum=0.0, meta_stepsize=1e-3, beta2_meta=0.999, stepsize_parameterization='exp', epsilon_meta=1e-3, meta_loss_type='none', meta_shadow_dist_reg=0.0, clip_zeta_meta='none'):
        self.opt_type = 'OboMetaZero2side'
        self.role = role

        self.net = network
        self.net_shadow_minus = deepcopy(network)
        self.net_shadow_plus = deepcopy(network)

        self.optimizer =              RMSPropCore(self.net.parameters(),              gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm, beta2=beta2, rmspower=rmspower, entrywise_normalization=entrywise_normalization, weight_decay=weight_decay, momentum=momentum)
        self.optimizer_shadow_minus = RMSPropCore(self.net_shadow_minus.parameters(), gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm, beta2=beta2, rmspower=rmspower, entrywise_normalization=entrywise_normalization, weight_decay=weight_decay, momentum=momentum)  
        self.optimizer_shadow_plus =  RMSPropCore(self.net_shadow_plus.parameters(),  gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm, beta2=beta2, rmspower=rmspower, entrywise_normalization=entrywise_normalization, weight_decay=weight_decay, momentum=momentum)  

        self.gamma = gamma
        self.entropy_coeff = entropy_coeff if role=='policy' else None
        self.meta_stepsize = meta_stepsize
        self.beta2_meta = beta2_meta
        self.epsilon_meta = epsilon_meta
        self.stepsize_parameterization = activation_function(stepsize_parameterization)
        self.shadow_distance_regulizer_coeff = meta_shadow_dist_reg
        self.shadow_distance_regulizer_coeff_base = meta_shadow_dist_reg+0.0

        self.update_meta_only_at_the_end_of_episodes = False
        if role=='critic':
            self.error_critic_main  =        error_critic(meta_loss_type=meta_loss_type, gamma=gamma)
            self.error_critic_shadow_minus = error_critic(meta_loss_type=meta_loss_type, gamma=gamma)
            self.error_critic_shadow_plus =  error_critic(meta_loss_type=meta_loss_type, gamma=gamma)
            if meta_loss_type.split('__epEndOnly_')[-1].split('__')[0] in ['True', 'true', '1']:
                self.update_meta_only_at_the_end_of_episodes = True

        self.eta = torch.tensor(1.0/kappa)
        self.zeta = activation_function_inverse(stepsize_parameterization, 1/kappa)
        self.v_meta_p = 1.0
        self.v_meta_m = 1.0
        self.v_meta_diff = 1.0
        self.t_meta = 1
        self.aggregate_beta2_meta = 1.0
        self.end_time_of_last_episode = 0
        self.clip_zeta_meta = clip_zeta_meta_function(clip_zeta_meta)


    def step(self, s, a, r, s_prime, reset, terminated_mask_t=None, delta=None):
        if self.role == 'critic':
            v_s, v_prime, delta =                      value_calculations(self.net,              s, s_prime, r, terminated_mask_t, gamma=self.gamma)
            v_s_shad_m, v_prime_shad_m, delta_shad_m = value_calculations(self.net_shadow_minus, s, s_prime, r, terminated_mask_t, gamma=self.gamma)
            v_s_shad_p, v_prime_shad_p, delta_shad_p = value_calculations(self.net_shadow_plus,  s, s_prime, r, terminated_mask_t, gamma=self.gamma)

        elif self.role == 'policy':
            delta_shad_m, delta_shad_p = delta, delta
            prob_pi, entropy_pi =               policy_calculations(self.net,              s, a, delta,        entropy_coeff=self.entropy_coeff, importance_sampling=False)
            prob_pi_shad_m, entropy_pi_shad_m = policy_calculations(self.net_shadow_minus, s, a, delta_shad_m, entropy_coeff=self.entropy_coeff, importance_sampling=True, target_policy_prob=prob_pi)
            prob_pi_shad_p, entropy_pi_shad_p = policy_calculations(self.net_shadow_plus,  s, a, delta_shad_p, entropy_coeff=self.entropy_coeff, importance_sampling=True, target_policy_prob=prob_pi)

            importance_sampling_m = prob_pi_shad_m/prob_pi
            importance_sampling_p = prob_pi_shad_p/prob_pi

        info = self.optimizer.step(delta, reset=reset)
        info_shad_m = self.optimizer_shadow_minus.step(delta_shad_m, reset=reset)
        info_shad_p = self.optimizer_shadow_plus.step(delta_shad_p, reset=reset)

        # meta errors:
        if self.role == 'critic':
            error_main =   self.error_critic_main.step(v_s, v_prime, r, v_prime, reset)
            error_shad_m = self.error_critic_shadow_minus.step(v_s_shad_m, v_prime_shad_m, r, v_prime, reset)
            error_shad_p = self.error_critic_shadow_plus.step( v_s_shad_p, v_prime_shad_p, r, v_prime, reset)
        elif self.role == 'policy':
            delta_normalized = self.optimizer.delta_normalized
            error_main = -(delta_normalized + self.entropy_coeff*entropy_pi)
            error_shad_m = -(importance_sampling_m*delta_normalized +  self.entropy_coeff*entropy_pi_shad_m)
            error_shad_p = -(importance_sampling_p*delta_normalized +  self.entropy_coeff*entropy_pi_shad_p)


        # meta update:
        if reset or not self.update_meta_only_at_the_end_of_episodes:
            self.t_meta += 1
            meta_grad_m = -(error_shad_m-error_main)/self.epsilon_meta
            self.v_meta_m =  self.v_meta_m + ((1-self.beta2_meta)/(1-self.beta2_meta**(self.t_meta-1))) * (meta_grad_m**2 - self.v_meta_m) 
            
            meta_grad_p = (error_shad_p-error_main)/self.epsilon_meta
            self.v_meta_p =  self.v_meta_p + ((1-self.beta2_meta)/(1-self.beta2_meta**(self.t_meta-1))) * (meta_grad_p**2 - self.v_meta_p) 
            
            meta_grad_diff = meta_grad_p - meta_grad_m
            self.v_meta_diff =  self.v_meta_diff + ((1-self.beta2_meta)/(1-self.beta2_meta**(self.t_meta-1))) * (meta_grad_diff**2 - self.v_meta_diff)

            self.zeta = self.zeta - self.meta_stepsize * meta_grad_diff / (math.sqrt(self.v_meta_diff) + 1e-8)
            self.zeta = self.clip_zeta_meta(self.zeta)

            
            #######
            
                
        self.eta = self.stepsize_parameterization(self.zeta)
        self.optimizer.eta = self.eta
        self.optimizer_shadow_minus.eta = self.stepsize_parameterization(self.zeta - self.epsilon_meta)
        self.optimizer_shadow_plus.eta  = self.stepsize_parameterization(self.zeta + self.epsilon_meta)
        
        if self.shadow_distance_regulizer_coeff>0: # regularization to pull shadow netwrok toward main network
            with torch.no_grad():
                for param, param_shad_m, param_shad_p in zip(self.net.parameters(), self.net_shadow_minus.parameters(), self.net_shadow_plus.parameters()):
                    param_shad_m.add_(param.data - param_shad_m.data, alpha=self.shadow_distance_regulizer_coeff)
                    param_shad_p.add_(param.data - param_shad_p.data, alpha=self.shadow_distance_regulizer_coeff)

        # info update:
        if self.role == 'critic':
            info.update({'v_s':float(v_s), 'v_prime':float(v_prime)})
            info_shad_m.update({'v_s':float(v_s_shad_m), 'v_prime':float(v_prime_shad_m)})
            info_shad_p.update({'v_s':float(v_s_shad_p), 'v_prime':float(v_prime_shad_p)})
        info.update({'meta_error':float(error_main)})
        info_shad_m.update({'meta_error':float(error_shad_m)})
        info_shad_p.update({'meta_error':float(error_shad_p)})

        return {**info, **{f'{key}_shadow_minus':info_shad_m[key] for key in info_shad_m}, **{f'{key}_shadow_plus':info_shad_p[key] for key in info_shad_p}}


class RMSPropMetaZero(): 
    def __init__(self, network, role, gamma=0.99, lamda=0.0, kappa=2.0, entropy_coeff=0.01, delta_clip='none', delta_norm='none', beta2=0.999,  rmspower=2.0, entrywise_normalization='none', 
            weight_decay=0.0, momentum=0.0, meta_stepsize=1e-3, beta2_meta=0.999, stepsize_parameterization='exp', epsilon_meta=1e-3, meta_loss_type='none', meta_shadow_dist_reg=0.0, clip_zeta_meta='none'):
        self.opt_type = 'OboMetaZero'
        self.role = role

        self.net = network
        self.net_shadow = deepcopy(network)

        self.optimizer =        RMSPropCore(self.net.parameters(), gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm, beta2=beta2, rmspower=rmspower, entrywise_normalization=entrywise_normalization, weight_decay=weight_decay, momentum=momentum)
        self.optimizer_shadow = RMSPropCore(self.net_shadow.parameters(), gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm, beta2=beta2, rmspower=rmspower, entrywise_normalization=entrywise_normalization, weight_decay=weight_decay, momentum=momentum)  

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


    def step(self, s, a, r, s_prime, reset, terminated_mask_t=None, delta=None):
        if self.role == 'critic':
            v_s, v_prime, delta = value_calculations(self.net, s, s_prime, r, terminated_mask_t, gamma=self.gamma)
            v_s_shadow, v_prime_shadow, delta_shadow = value_calculations(self.net_shadow, s, s_prime, r, terminated_mask_t, gamma=self.gamma)

        elif self.role == 'policy':
            delta_shadow = delta
            prob_pi, entropy_pi = policy_calculations(self.net, s, a, delta, entropy_coeff=self.entropy_coeff, importance_sampling=False)
            prob_pi_shadow, entropy_pi_shadow = policy_calculations(self.net_shadow, s, a, delta_shadow, entropy_coeff=self.entropy_coeff, importance_sampling=True, target_policy_prob=prob_pi)
            
            importance_sampling = prob_pi_shadow/prob_pi
            
        
        info = self.optimizer.step(delta, reset=reset)
        info_shadow = self.optimizer_shadow.step(delta_shadow, reset=reset)

        # meta update
        if self.role == 'critic':
            error_main = self.error_critic_main.step(v_s, v_prime, r, v_prime, reset)
            error_shadow = self.error_critic_shadow.step(v_s_shadow, v_prime_shadow, r, v_prime, reset)
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

        self.eta = self.stepsize_parameterization(self.zeta)
        self.optimizer.eta = self.eta
        self.optimizer_shadow.eta = self.stepsize_parameterization(self.zeta + self.epsilon_meta)
        
        if self.shadow_distance_regulizer_coeff>0: # regularization to pull shadow netwrok toward main network
            with torch.no_grad():
                for param, param_shadow in zip(self.net.parameters(), self.net_shadow.parameters()):
                    param_shadow.add_(param.data - param_shadow.data, alpha=self.shadow_distance_regulizer_coeff)
            
        
        # info update:
        if self.role == 'critic':
            info.update({'v_s':float(v_s), 'v_prime':float(v_prime)})
            info_shadow.update({'v_s':float(v_s_shadow), 'v_prime':float(v_prime_shadow)})
        info.update({'meta_error':float(error_main)})
        info_shadow.update({'meta_error':float(error_shadow)})

        return {**info, **{f'{key}_shadow':info_shadow[key] for key in info_shadow}}





class OboMetaZero2side(): 
    def __init__(self, network, role, gamma=0.99, lamda=0.0, kappa=2.0, entropy_coeff=0.01, delta_clip='none', delta_norm='none', beta2=0.999,  rmspower=2.0, entrywise_normalization='none', 
            weight_decay=0.0, momentum=0.0, meta_stepsize=1e-3, beta2_meta=0.999, stepsize_parameterization='exp', epsilon_meta=1e-3, meta_loss_type='none', meta_shadow_dist_reg=0.0, clip_zeta_meta='none'):
        self.opt_type = 'OboMetaZero2side'
        self.role = role

        self.net = network
        self.net_shadow_minus = deepcopy(network)
        self.net_shadow_plus = deepcopy(network)

        self.optimizer =              OboCore(self.net.parameters(),              gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm, beta2=beta2, rmspower=rmspower, entrywise_normalization=entrywise_normalization, weight_decay=weight_decay, momentum=momentum)
        self.optimizer_shadow_minus = OboCore(self.net_shadow_minus.parameters(), gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm, beta2=beta2, rmspower=rmspower, entrywise_normalization=entrywise_normalization, weight_decay=weight_decay, momentum=momentum)  
        self.optimizer_shadow_plus =  OboCore(self.net_shadow_plus.parameters(),  gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm, beta2=beta2, rmspower=rmspower, entrywise_normalization=entrywise_normalization, weight_decay=weight_decay, momentum=momentum)  

        self.gamma = gamma
        self.entropy_coeff = entropy_coeff if role=='policy' else None
        self.meta_stepsize = meta_stepsize
        self.beta2_meta = beta2_meta
        self.epsilon_meta = epsilon_meta
        self.stepsize_parameterization = activation_function(stepsize_parameterization)
        self.shadow_distance_regulizer_coeff = meta_shadow_dist_reg
        self.shadow_distance_regulizer_coeff_base = meta_shadow_dist_reg+0.0

        self.update_meta_only_at_the_end_of_episodes = False
        if role=='critic':
            self.error_critic_main  =        error_critic(meta_loss_type=meta_loss_type, gamma=gamma)
            self.error_critic_shadow_minus = error_critic(meta_loss_type=meta_loss_type, gamma=gamma)
            self.error_critic_shadow_plus =  error_critic(meta_loss_type=meta_loss_type, gamma=gamma)
            if meta_loss_type.split('__epEndOnly_')[-1].split('__')[0] in ['True', 'true', '1']:
                self.update_meta_only_at_the_end_of_episodes = True

        self.eta = torch.tensor(1.0/kappa)
        self.zeta = activation_function_inverse(stepsize_parameterization, 1/kappa)
        self.v_meta_p = 1.0
        self.v_meta_m = 1.0
        self.v_meta_diff = 1.0
        self.t_meta = 1
        self.aggregate_beta2_meta = 1.0
        self.end_time_of_last_episode = 0
        self.clip_zeta_meta = clip_zeta_meta_function(clip_zeta_meta)


    def step(self, s, a, r, s_prime, reset, terminated_mask_t=None, delta=None):
        if self.role == 'critic':
            v_s, v_prime, delta =                      value_calculations(self.net,              s, s_prime, r, terminated_mask_t, gamma=self.gamma)
            v_s_shad_m, v_prime_shad_m, delta_shad_m = value_calculations(self.net_shadow_minus, s, s_prime, r, terminated_mask_t, gamma=self.gamma)
            v_s_shad_p, v_prime_shad_p, delta_shad_p = value_calculations(self.net_shadow_plus,  s, s_prime, r, terminated_mask_t, gamma=self.gamma)

        elif self.role == 'policy':
            delta_shad_m, delta_shad_p = delta, delta
            prob_pi, entropy_pi =               policy_calculations(self.net,              s, a, delta,        entropy_coeff=self.entropy_coeff, importance_sampling=False)
            prob_pi_shad_m, entropy_pi_shad_m = policy_calculations(self.net_shadow_minus, s, a, delta_shad_m, entropy_coeff=self.entropy_coeff, importance_sampling=True, target_policy_prob=prob_pi)
            prob_pi_shad_p, entropy_pi_shad_p = policy_calculations(self.net_shadow_plus,  s, a, delta_shad_p, entropy_coeff=self.entropy_coeff, importance_sampling=True, target_policy_prob=prob_pi)

            importance_sampling_m = prob_pi_shad_m/prob_pi
            importance_sampling_p = prob_pi_shad_p/prob_pi

        info = self.optimizer.step(delta, reset=reset)
        info_shad_m = self.optimizer_shadow_minus.step(delta_shad_m, reset=reset)
        info_shad_p = self.optimizer_shadow_plus.step(delta_shad_p, reset=reset)

        # meta errors:
        if self.role == 'critic':
            error_main =   self.error_critic_main.step(v_s, v_prime, r, v_prime, reset)
            error_shad_m = self.error_critic_shadow_minus.step(v_s_shad_m, v_prime_shad_m, r, v_prime, reset)
            error_shad_p = self.error_critic_shadow_plus.step( v_s_shad_p, v_prime_shad_p, r, v_prime, reset)
        elif self.role == 'policy':
            delta_normalized = self.optimizer.delta_normalized
            error_main = -(delta_normalized + self.entropy_coeff*entropy_pi)
            error_shad_m = -(importance_sampling_m*delta_normalized +  self.entropy_coeff*entropy_pi_shad_m)
            error_shad_p = -(importance_sampling_p*delta_normalized +  self.entropy_coeff*entropy_pi_shad_p)


        # meta update:
        if reset or not self.update_meta_only_at_the_end_of_episodes:
            self.t_meta += 1
            meta_grad_m = -(error_shad_m-error_main)/self.epsilon_meta
            self.v_meta_m =  self.v_meta_m + ((1-self.beta2_meta)/(1-self.beta2_meta**(self.t_meta-1))) * (meta_grad_m**2 - self.v_meta_m) 
            
            meta_grad_p = (error_shad_p-error_main)/self.epsilon_meta
            self.v_meta_p =  self.v_meta_p + ((1-self.beta2_meta)/(1-self.beta2_meta**(self.t_meta-1))) * (meta_grad_p**2 - self.v_meta_p) 
            
            meta_grad_diff = meta_grad_p - meta_grad_m
            self.v_meta_diff =  self.v_meta_diff + ((1-self.beta2_meta)/(1-self.beta2_meta**(self.t_meta-1))) * (meta_grad_diff**2 - self.v_meta_diff)

            self.zeta = self.zeta - self.meta_stepsize * meta_grad_diff / (math.sqrt(self.v_meta_diff) + 1e-8)
            self.zeta = self.clip_zeta_meta(self.zeta)

            
            #######
            

            

            
            if False: # for debuggung and analysis of shadows:
                norm_main, norm_m, norm_p, norm_diff, cor, norm_m_1, norm_p_1, norm_diff_1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for param, param_shad_m, param_shad_p in zip(self.net.parameters(), self.net_shadow_minus.parameters(), self.net_shadow_plus.parameters()):
                    norm_main += torch.sum((param.data)**2).item()
                    norm_m += torch.sum((param_shad_m.data - param.data)**2).item()
                    norm_p += torch.sum((param_shad_p.data - param.data)**2).item()
                    norm_diff += torch.sum((param_shad_p.data + param_shad_m.data - 2*param.data)**2).item()
                    norm_m_1 += torch.sum(torch.abs(param_shad_m.data - param.data)).item()
                    norm_p_1 += torch.sum(torch.abs(param_shad_p.data - param.data)).item()
                    norm_diff_1 += torch.sum(torch.abs(param_shad_p.data + param_shad_m.data - 2*param.data)).item()
                    cor += torch.sum((param_shad_p.data - param.data)*(param.data-param_shad_m.data)).item()

                norm_m = math.sqrt(norm_m)
                norm_p = math.sqrt(norm_p)
                norm_main = math.sqrt(norm_main)
                norm_diff = math.sqrt(norm_diff)
                cor_coeff = cor/(norm_m*norm_p+1e-12)
                norm_diff_ratio = norm_diff/(norm_m+norm_p+1e-12)

                if self.t_meta%10_000==0:
                    print(f"\n{self.t_meta},\t\t {cor_coeff:.4f},\t {norm_m:.4f},\t {norm_p:.4f}, \t {norm_main:.4f} e6")

                #####
                meta_grad_ratio = meta_grad_p/(meta_grad_m+1e-12)
                meta_grad_ratio_smooth = (meta_grad_p+math.copysign(1,meta_grad_p)*.1*math.sqrt(self.v_meta_p))/(meta_grad_m+math.copysign(1,meta_grad_m)*0.1*math.sqrt(self.v_meta_m)+1e-12)
                meta_grad_m_noemalized = meta_grad_m/(math.sqrt(self.v_meta_m)+1e-12)
                meta_grad_p_noemalized = meta_grad_p/(math.sqrt(self.v_meta_p)+1e-12)
                meta_grad_min = 0 if meta_grad_p*meta_grad_m<0 else min(abs(meta_grad_p), abs(meta_grad_m))*math.copysign(1, meta_grad_p)
                #if math.log(abs(meta_grad_ratio+1e-12))>math.log(2.0) and abs(meta_grad_p/(math.sqrt(self.v_meta_p)+1e-12))>.5:
                if self.t_meta%10_000==0:
                    print(f"{self.t_meta},\t {self.t_meta-(self.last_t if hasattr(self, 'last_t') else 0)}, \t {meta_grad_ratio_smooth:.4f},\t  {meta_grad_ratio:.4f},\t {math.sqrt(self.v_meta_p)/(math.sqrt(self.v_meta_m)+1e-12):.4f},\t {meta_grad_m_noemalized:.4f},\t {meta_grad_p_noemalized:.4f}")
                    self.last_t = self.t_meta+0
                
                
                if False: # adjust the regilarizer coefficient
                    self.shadow_distance_regulizer_coeff *= (norm_diff_ratio+.95)
                    self.shadow_distance_regulizer_coeff = max(min(self.shadow_distance_regulizer_coeff, 10*self.shadow_distance_regulizer_coeff_base), 0.1*self.shadow_distance_regulizer_coeff_base)
                    # if norm_diff_ratio>0.1:
                    #     self.shadow_distance_regulizer_coeff = min(self.shadow_distance_regulizer_coeff*1.1, 10*self.shadow_distance_regulizer_coeff_base)
                    # if norm_diff_ratio<0.04:
                    #     self.shadow_distance_regulizer_coeff = max(self.shadow_distance_regulizer_coeff*0.9, 0.1*self.shadow_distance_regulizer_coeff_base)
                    
                    if self.t_meta%2000<100:
                        print(self.shadow_distance_regulizer_coeff)
                
                
                if True: # plotting
                    intervals = 50_000
                    points_in_interval = 2_000
                    tt = (self.t_meta-2) % intervals
                    if tt == 0:
                        self.info_plot=[]
                    if tt<points_in_interval-1:
                        self.info_plot.append({
                            'meta_grad_ratio': meta_grad_ratio,
                            'meta_grad_ratio_smooth': meta_grad_ratio_smooth,
                            'meta_grad_m_noemalized': meta_grad_m_noemalized,
                            'meta_grad_p_noemalized': meta_grad_p_noemalized,
                            'meta_grad_m': meta_grad_m,
                            'meta_grad_p': meta_grad_p,
                            'meta_grad_min':meta_grad_min,
                            'cor_coeff': cor_coeff,
                            'norm_m': norm_m,
                            'norm_p': norm_p,
                            'norm_diff': norm_diff,
                            'norm_m_1': norm_m_1,
                            'norm_p_1': norm_p_1,
                            'norm_diff_1': norm_diff_1,
                            'norm_main': norm_main, 
                            'episode_end': reset,
                            'norm_grad_main': info['norm2_grad'],
                            'norm_grad_m': info_shad_m['norm2_grad'],
                            'norm_grad_p': info_shad_p['norm2_grad'],
                        })
                    if tt == points_in_interval-1:
                        import numpy as np
                        import matplotlib.pyplot as plt
                        def binning_for_plot(x,y):
                            n_bins = 20
                            idx = np.argsort(x)
                            xs = x[idx]
                            ys = y[idx]
                            bins = np.array_split(np.arange(len(xs)), n_bins)
                            x_bin = np.array([xs[b].mean() for b in bins])
                            y_bin = np.array([ys[b].mean() for b in bins])
                            y_bin_sq = np.sqrt(np.array([(ys[b]**2).mean() for b in bins]))
                            return x_bin, y_bin, y_bin_sq
                        info_effective = np.array([info for info in self.info_plot  if info['meta_grad_m_noemalized']>0.1 and info['meta_grad_p_noemalized']>0.1])
                        cor_coeff_list = np.array([info['cor_coeff'] for info in info_effective])
                        meta_grad_log_ratio_list = np.array([math.log(abs(info['meta_grad_ratio']))/math.log(2) for info in info_effective])
                        meta_grad_abs_log_ratio_list = np.array([abs(math.log(abs(info['meta_grad_ratio']))/math.log(2)) for info in info_effective])
                        norm_p_over_norm_m_log_list = np.array([(math.log(info['norm_p']+1e-12)/(info['norm_m']+1e-12))/math.log(2) for info in info_effective])
                        norm_diff_ratio_list = np.array([info['norm_diff']/(info['norm_p']+info['norm_m']+1e-12) for info in info_effective]) 
                        norm_diff_list = np.array([info['norm_diff'] for info in info_effective])
                        norm_diff_log_list = np.array([math.log(info['norm_diff']+1e-12)/math.log(2) for info in info_effective])
                        
                        # Plots:
                        lw=1
                        num_cols = 3
                        plt.figure(figsize=(18,9))
                        #---
                        plt.subplot(2,num_cols,1)
                        #plt.scatter([info['cor_coeff'] for info in self.info_plot  if info['meta_grad_m_noemalized']>0.1 and info['meta_grad_p_noemalized']>0.1], [info['meta_grad_ratio_smooth'] for info in self.info_plot if info['meta_grad_m_noemalized']>0.1 and info['meta_grad_p_noemalized']>0.1], alpha=0.5)
                        plt.scatter(cor_coeff_list, meta_grad_abs_log_ratio_list, alpha=0.5)
                        plt.xlabel(r'correlation coeff of $\Delta w^+$ and $\Delta w^-$ (w.r.t. main network)')
                        plt.ylabel('meta grad ratio (abs log_2)')
                        plt.subplot(2,num_cols,num_cols+1)
                        x,y,y_sq = binning_for_plot(cor_coeff_list, meta_grad_abs_log_ratio_list)
                        plt.plot(x, y, 'o-', lw=lw)
                        plt.plot(x, y_sq, '-', color='red', lw=lw)
                        #---
                        plt.subplot(2,num_cols,2)
                        plt.scatter(norm_diff_ratio_list, meta_grad_abs_log_ratio_list, alpha=0.5)
                        plt.xlabel(r'norm_diff_ratio_list')
                        plt.ylabel('meta grad ratio (abs log_2)')
                        plt.subplot(2,num_cols,num_cols+2)
                        x,y,y_sq = binning_for_plot(norm_diff_ratio_list, meta_grad_abs_log_ratio_list) 
                        plt.plot(x, y, 'o-', lw=lw)
                        plt.plot(x, y_sq, '-', color='red', lw=lw)
                        #---
                        plt.subplot(2,num_cols,3)
                        plt.scatter(norm_diff_log_list, meta_grad_abs_log_ratio_list, alpha=0.5)
                        plt.xlabel(r'norm_diff_log_list')
                        plt.ylabel('meta grad ratio (abs log_2)')
                        plt.subplot(2,num_cols,num_cols+3)
                        x,y,y_sq = binning_for_plot(norm_diff_log_list, meta_grad_abs_log_ratio_list) 
                        plt.plot(x, y, 'o-', lw=lw)
                        plt.plot(x, y_sq, '-', color='red', lw=lw)
                        plt.show()

                        #---
                        ep_end_list = [i for i in range(len(self.info_plot)) if self.info_plot[i]['episode_end']]
                        plt.figure(figsize=(18,9))
                        #plt.subplot(2,num_cols,4)
                        plt.plot(np.array([info['meta_grad_m'] for info in self.info_plot]), lw=lw, label='meta_grad_m')
                        plt.plot(np.array([info['meta_grad_p'] for info in self.info_plot]), color='red', lw=lw, label='meta_grad_p')
                        plt.plot(np.array([info['meta_grad_min'] for info in self.info_plot]), color='black', lw=lw, label='meta_grad_min')
                        main_keys = ['meta_grad_m', 'meta_grad_p']
                        aux_keys = ['norm_grad_main', 'norm_grad_p', 'norm_grad_m']
                        range_plot_min, range_plot_max = min([min(info[key] for key in main_keys) for info in self.info_plot]), max([max(info[key] for key in main_keys) for info in self.info_plot])
                        range_plot = range_plot_max - range_plot_min
                        range_2 = max([max(info[key] for key in aux_keys) for info in self.info_plot]) - min([min(info[key] for key in aux_keys) for info in self.info_plot])
                        plt.plot(ep_end_list, [range_plot_min]*len(ep_end_list), 'x', color='green', label='episode end')
                        plt.plot([0,len(self.info_plot)], [range_plot_max,range_plot_max], '-', color='green', lw=lw)
                        plt.plot(np.array([range_plot_max+info['norm_grad_main']*range_plot/range_2/10 for info in self.info_plot]), color='blue', lw=lw, label='norm_grad_m')
                        plt.plot(np.array([range_plot_max+info['norm_grad_p']*range_plot/range_2/10 for info in self.info_plot]), color='red', lw=lw, label='norm_grad_p')
                        plt.plot(np.array([range_plot_max+info['norm_grad_m']*range_plot/range_2/10 for info in self.info_plot]), color='black', lw=lw, label='norm_grad_m')
                        plt.xlabel(r'time')
                        plt.ylabel('meta grads ')
                        plt.legend()
                        plt.show()
                        
                        plt.figure(figsize=(18,9))
                        #plt.subplot(2,num_cols,num_cols+4)
                        norm_type_plot = '' # '_1' for L1 norm, '' for L2 norm
                        plt.plot([info[f'norm_m{norm_type_plot}'] for info in self.info_plot], lw=lw, label=f'norm_m{norm_type_plot}')
                        plt.plot([info[f'norm_p{norm_type_plot}'] for info in self.info_plot], color='red', lw=lw, label=f'norm_p{norm_type_plot}')
                        plt.plot([info[f'norm_diff{norm_type_plot}'] for info in self.info_plot], color='black', lw=lw, label=f'norm_diff{norm_type_plot}')
                        main_keys = [f'norm_m{norm_type_plot}', f'norm_p{norm_type_plot}', f'norm_diff{norm_type_plot}']
                        aux_keys = ['norm_grad_main', 'norm_grad_p', 'norm_grad_m']
                        range_plot_min, range_plot_max = min([min(info[key] for key in main_keys) for info in self.info_plot]), max([max(info[key] for key in main_keys) for info in self.info_plot])
                        range_plot = range_plot_max - range_plot_min
                        range_2 = max([max(info[key] for key in aux_keys) for info in self.info_plot]) - min([min(info[key] for key in aux_keys) for info in self.info_plot])
                        plt.plot(ep_end_list, [range_plot_min]*len(ep_end_list), 'x', color='green', label='episode end')
                        plt.plot([0,len(self.info_plot)], [range_plot_max,range_plot_max], '-', color='green', lw=lw)
                        plt.plot(np.array([range_plot_max+info['norm_grad_main']*range_plot/range_2/10 for info in self.info_plot]), color='blue', lw=lw, label='norm_grad_main')
                        plt.plot(np.array([range_plot_max+info['norm_grad_p']*range_plot/range_2/10 for info in self.info_plot]), color='red', lw=lw, label='norm_grad_p')
                        plt.plot(np.array([range_plot_max+info['norm_grad_m']*range_plot/range_2/10 for info in self.info_plot]), color='black', lw=lw, label='norm_grad_m')
                        plt.xlabel(r'time')
                        plt.ylabel('norms')
                        plt.legend()
                        plt.show()

                

        self.eta = self.stepsize_parameterization(self.zeta)
        self.optimizer.eta = self.eta
        self.optimizer_shadow_minus.eta = self.stepsize_parameterization(self.zeta - self.epsilon_meta)
        self.optimizer_shadow_plus.eta  = self.stepsize_parameterization(self.zeta + self.epsilon_meta)
        
        if self.shadow_distance_regulizer_coeff>0: # regularization to pull shadow netwrok toward main network
            with torch.no_grad():
                for param, param_shad_m, param_shad_p in zip(self.net.parameters(), self.net_shadow_minus.parameters(), self.net_shadow_plus.parameters()):
                    param_shad_m.add_(param.data - param_shad_m.data, alpha=self.shadow_distance_regulizer_coeff)
                    param_shad_p.add_(param.data - param_shad_p.data, alpha=self.shadow_distance_regulizer_coeff)

        # info update:
        if self.role == 'critic':
            info.update({'v_s':float(v_s), 'v_prime':float(v_prime)})
            info_shad_m.update({'v_s':float(v_s_shad_m), 'v_prime':float(v_prime_shad_m)})
            info_shad_p.update({'v_s':float(v_s_shad_p), 'v_prime':float(v_prime_shad_p)})
        info.update({'meta_error':float(error_main)})
        info_shad_m.update({'meta_error':float(error_shad_m)})
        info_shad_p.update({'meta_error':float(error_shad_p)})

        return {**info, **{f'{key}_shadow_minus':info_shad_m[key] for key in info_shad_m}, **{f'{key}_shadow_plus':info_shad_p[key] for key in info_shad_p}}





class OboMetaZero(): 
    def __init__(self, network, role, gamma=0.99, lamda=0.0, kappa=2.0, entropy_coeff=0.01, delta_clip='none', delta_norm='none', beta2=0.999,  rmspower=2.0, entrywise_normalization='none', 
            weight_decay=0.0, momentum=0.0, meta_stepsize=1e-3, beta2_meta=0.999, stepsize_parameterization='exp', epsilon_meta=1e-3, meta_loss_type='none', meta_shadow_dist_reg=0.0, clip_zeta_meta='none'):
        self.opt_type = 'OboMetaZero'
        self.role = role

        self.net = network
        self.net_shadow = deepcopy(network)

        self.optimizer =        OboCore(self.net.parameters(), gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm, beta2=beta2, rmspower=rmspower, entrywise_normalization=entrywise_normalization, weight_decay=weight_decay, momentum=momentum)
        self.optimizer_shadow = OboCore(self.net_shadow.parameters(), gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm, beta2=beta2, rmspower=rmspower, entrywise_normalization=entrywise_normalization, weight_decay=weight_decay, momentum=momentum)  

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


    def step(self, s, a, r, s_prime, reset, terminated_mask_t=None, delta=None):
        if self.role == 'critic':
            v_s, v_prime, delta = value_calculations(self.net, s, s_prime, r, terminated_mask_t, gamma=self.gamma)
            v_s_shadow, v_prime_shadow, delta_shadow = value_calculations(self.net_shadow, s, s_prime, r, terminated_mask_t, gamma=self.gamma)

        elif self.role == 'policy':
            delta_shadow = delta
            prob_pi, entropy_pi = policy_calculations(self.net, s, a, delta, entropy_coeff=self.entropy_coeff, importance_sampling=False)
            prob_pi_shadow, entropy_pi_shadow = policy_calculations(self.net_shadow, s, a, delta_shadow, entropy_coeff=self.entropy_coeff, importance_sampling=True, target_policy_prob=prob_pi)
            
            importance_sampling = prob_pi_shadow/prob_pi
            
        
        info = self.optimizer.step(delta, reset=reset)
        info_shadow = self.optimizer_shadow.step(delta_shadow, reset=reset)

        # meta update
        if self.role == 'critic':
            error_main = self.error_critic_main.step(v_s, v_prime, r, v_prime, reset)
            error_shadow = self.error_critic_shadow.step(v_s_shadow, v_prime_shadow, r, v_prime, reset)
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
            
        
        # info update:
        if self.role == 'critic':
            info.update({'v_s':float(v_s), 'v_prime':float(v_prime)})
            info_shadow.update({'v_s':float(v_s_shadow), 'v_prime':float(v_prime_shadow)})
        info.update({'meta_error':float(error_main)})
        info_shadow.update({'meta_error':float(error_shadow)})

        return {**info, **{f'{key}_shadow':info_shadow[key] for key in info_shadow}}




class OboMetaOpt(torch.optim.Optimizer): 
    def __init__(self, network, role, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', beta2=0.999,  rmspower=2.0, entrywise_normalization='none', 
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


class OboBase(): 
    def __init__(self, network, role, gamma=0.99, lamda=0.0, kappa=2.0, entropy_coeff=0.01, delta_clip='none', delta_norm='none', 
                 beta2=0.999, rmspower=2.0, entrywise_normalization='none', weight_decay=0.0, momentum=0.0):
        self.role = role
        self.net = network
        self.optimizer = OboCore(self.net.parameters(), gamma=gamma, lamda=lamda, kappa=kappa, delta_clip=delta_clip, delta_norm=delta_norm, beta2=beta2, rmspower=rmspower, entrywise_normalization=entrywise_normalization, weight_decay=weight_decay, momentum=momentum)
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.eta = torch.tensor(1.0/kappa)
        self.v_meta = 1.0

    def step(self, s, a, r, s_prime, reset, terminated_mask_t=None, delta=None):
        if self.role == 'critic':
            v_s, v_prime, delta = value_calculations(self.net, s, s_prime, r, terminated_mask_t, gamma=self.gamma)
        elif self.role == 'policy':
            _,_ = policy_calculations(self.net, s, a, delta, entropy_coeff=self.entropy_coeff, importance_sampling=False)            
        
        info = self.optimizer.step(delta, reset=reset)

        if self.role == 'critic':
            info.update({'v_s':float(v_s), 'v_prime':float(v_prime)})
        
        return info


class RMSPropCore(torch.optim.Optimizer): 
    def __init__(self, params, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', 
                 beta2=0.999, rmspower=2.0, entrywise_normalization='none', weight_decay=0.0, momentum=0.0):
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
        super(RMSPropCore, self).__init__(params, defaults)

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
        
        normalizer_decay_rate = 0.999
        #normalizer_decay_rate = self.gamma*self.lamda

        self.sigma +=   (1-normalizer_decay_rate) * (norm_grad-self.sigma)
        norm_normalizer = math.sqrt((self.sigma/(1-normalizer_decay_rate**self.t_val)) + 1e-12) 
        z_normalizer = math.sqrt(z_sum/(1-(self.gamma*self.lamda)**self.t_val))
        normalizer_ = norm_normalizer * z_normalizer
        step_size = 1/norm_normalizer**2
        #print(self.eta/step_size)

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

        info = {'clipped_step_size':float(step_size), 'delta':float(delta), 'delta_used':float(safe_delta), 'abs_delta':float(abs(delta)), 'norm2_eligibility_trace':float(z_sum), 'norm2_grad':float(norm_grad)**.5}
        return info
    


class RMSPropCore_test(torch.optim.Optimizer): 
    def __init__(self, params, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', 
                 beta2=0.999, rmspower=2.0, entrywise_normalization='none', weight_decay=0.0, momentum=0.0):
        defaults = dict(gamma=gamma, lamda=lamda, beta2=beta2, beta_momentum=momentum, rmspower=rmspower)
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
        super(RMSPropCoretest, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        safe_delta = self.delta_clipper.clip_and_norm(delta)
        self.delta_normalized = safe_delta
        self.t_val += 1

        # base update
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad)
                
                if self.entrywise_normalization.lower() in ['rmsprop']:
                    v.mul_(group["beta2"]).add_(torch.pow(torch.abs(p.grad), group["rmspower"]), alpha=1.0 - group["beta2"])
                    v_hat = torch.pow(v / (1.0 - group["beta2"] ** self.t_val), 1.0/group["rmspower"]) + 1e-8
                elif self.entrywise_normalization.lower()=='none':
                    v_hat = 1.0

                if group["beta_momentum"]>0:
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p.data)
                    m = state["momentum"]
                    if self.entrywise_normalization.lower() == 'rmsprop':
                        m.mul_(group["beta_momentum"]).addcdiv_(e, v_hat, value= safe_delta * (1-group["beta_momentum"]))
                    else:
                        m.mul_(group["beta_momentum"]).add_(e, alpha= safe_delta * (1-group["beta_momentum"]))
                    delta_w = m
                else:
                    delta_w = safe_delta * e / v_hat

                if self.weight_decay > 0.0:
                    p.data.mul_(1.0-self.eta*self.weight_decay)

                p.data.add_(delta_w, alpha=self.eta)

                if reset:
                    e.zero_()

        info = {'delta':float(delta), 'delta_used':float(safe_delta), 'abs_delta':float(abs(delta))}
        return info


class OboCore(torch.optim.Optimizer): 
    def __init__(self, params, gamma=0.99, lamda=0.0, kappa=2.0, delta_clip='none', delta_norm='none', 
                 beta2=0.999, rmspower=2.0, entrywise_normalization='none', weight_decay=0.0, momentum=0.0):
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
        super(OboCore, self).__init__(params, defaults)

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
        #print(self.eta/step_size)

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

        info = {'clipped_step_size':float(step_size), 'delta':float(delta), 'delta_used':float(safe_delta), 'abs_delta':float(abs(delta)), 'norm2_eligibility_trace':float(z_sum), 'norm2_grad':float(norm_grad)**.5}
        return info
    








def policy_calculations(local_net, s, a, delta, entropy_coeff=0.01, importance_sampling=False, target_policy_prob=None):
    mu, std = local_net(s)
    dist = Normal(mu, std)

    log_prob_pi = (dist.log_prob(a)).sum()
    prob_pi = torch.exp(log_prob_pi)
    if importance_sampling and target_policy_prob is not None:
        IS_ = prob_pi.item()/target_policy_prob
    else:            
        IS_ = 1.0
    pure_entropy = dist.entropy().sum()
    entropy_pi = entropy_coeff * dist.entropy().sum() * torch.sign(torch.tensor(delta)).item()
    local_net.zero_grad()
    (IS_*log_prob_pi + entropy_pi).backward()

    return prob_pi.item(), pure_entropy.item()

def value_calculations(local_net, s, s_prime, r, terminated_mask_t, gamma=0.99):
    with torch.no_grad():
        v_prime = local_net(s_prime)
    v_s = local_net(s)
    with torch.no_grad():
        td_target_critic = r + gamma * v_prime * terminated_mask_t
        delta = td_target_critic - v_s
    local_net.zero_grad()
    v_s.backward()
    return v_s.item(), v_prime.item(), delta.item()
