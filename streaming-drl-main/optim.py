import torch
import math

class Obn(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, eta=1.0, u_trace=.99, beta2=0.999, entryise_normalization='none'):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, eta=eta, beta2=beta2)
        self.u_bar = 0.0
        self.u_trace = u_trace
        self.t_val = 0
        self.entryise_normalization = entryise_normalization # 'none' or 'RMSprop'
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
                if self.entryise_normalization.lower()=='rmsprop':
                    v.mul_(group["beta2"]).addcmul_(p.grad, p.grad, value=1.0 - group["beta2"])
                    sigma = (v / (1.0 - group["beta2"] ** self.t_val)).sqrt() + 1e-8
                    e.mul_(group["gamma"] * group["lamda"]).addcdiv_(p.grad, sigma, value=1.0)
                elif self.entryise_normalization.lower()=='none':
                    e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                    sigma = torch.ones_like(p.data)
                z_sum += ((e.square()* sigma).sum()).abs().item()
        
        self.u_bar +=  self.u_trace * (z_sum-self.u_bar)
        norm_normalizer = max(z_sum, math.sqrt(z_sum*self.u_bar/(1-self.u_trace**self.t_val)))
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

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'abs_delta':abs(delta), 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info
    



class ObGD_sq_plain(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD_sq_plain, self).__init__(params, defaults)
    
    def step(self, delta, reset=False):
        lgging_info = {}
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

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'abs_delta':abs(delta), 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info


class ObGD_sq(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD_sq, self).__init__(params, defaults)
    
    def step(self, delta, reset=False):
        lgging_info = {}
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

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'abs_delta':abs(delta), 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
        return info





# --------------------------------


class TDNots(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, eta=1.0):
        defaults = dict(lr=lr, gamma=gamma, kappa=kappa, eta=eta)
        super(TDNots, self).__init__(params, defaults)
    def policy_step(self, delta, reset=False):
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                z_sum += p.grad.abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                p.data.add_(delta * p.grad, alpha=-step_size)

    def value_step(self, delta, reset=False, v_prime_grads=None):
        z_sum = 0.0; v_v_prime_product = 0.0; v_prime_grad_norm_squared = 0.0; v_grad_norm_squared = 0.0
        for group_idx, group in enumerate(self.param_groups):
            group_grads = v_prime_grads[group_idx]
            for p_idx, p in enumerate(group["params"]):
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                v_prime_grad = group_grads[p_idx]
                v_v_prime_product += (p.grad * v_prime_grad).sum().item()
                v_prime_grad_norm_squared += v_prime_grad.square().sum().item()
                v_grad_norm_squared += p.grad.square().sum().item()

        cosine_similarity = v_v_prime_product / (math.sqrt(v_grad_norm_squared) * math.sqrt(v_prime_grad_norm_squared))
        #coeff = 1.0 / math.sqrt(1.0 - group["eta"] * (cosine_similarity**2))
        coeff = 1.0 / (1.0 - group["eta"] * (cosine_similarity**2))

        for group_idx, group in enumerate(self.param_groups):
            group_grads = v_prime_grads[group_idx]
            for p_idx, p in enumerate(group["params"]):
                state = self.state[p]
                v_prime_grad = group_grads[p_idx]
                z_sum += (p.grad.square().sum()).abs().item()

        #delta_bar = max(abs(delta), 1.0)
        delta_bar = 1.0
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        # if dot_product > 1:
        #     step_size = group["lr"] / dot_product
        # else:
        #     step_size = group["lr"]
        step_size = group["lr"] / dot_product

        for group_idx, group in enumerate(self.param_groups):
            group_grads = v_prime_grads[group_idx]
            for p_idx, p in enumerate(group["params"]):
                state = self.state[p]
                v_prime_grad = group_grads[p_idx]
                u = p.grad - (1.0-reset) * group["eta"] * v_prime_grad * v_v_prime_product / v_prime_grad_norm_squared
                p.data.add_(delta * u, alpha=-step_size * coeff)


class ObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD, self).__init__(params, defaults)
    def step(self, delta, reset=False):
        lgging_info = {}
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

        info = {'M':dot_product, 'clipped_step_size':step_size, 'delta':delta, 'abs_delta':abs(delta), 'delta_bar':delta_bar, 'norm1_eligibility_trace':z_sum}
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
