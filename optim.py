import torch
import math
from delta_clipper import delta_clipper


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
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad-self.sigma**(self.sig_power/2.0))
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
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad-self.sigma**(self.sig_power/2.0))
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
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad-self.sigma**(self.sig_power/2.0))
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
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad-self.sigma**(self.sig_power/2.0))
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
        
        self.sigma +=  (1-self.gamma*self.lamda) * (norm_grad-self.sigma**(self.sig_power/2.0))
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
