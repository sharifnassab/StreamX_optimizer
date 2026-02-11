import math, torch, time

class OboPolicy(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=20.0, beta2=0.999, momentum=0.9):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, beta2=beta2, beta_momentum=momentum)
        super().__init__(params, defaults)

        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.beta2 = beta2
        self.momentum = momentum

        self.u_trace = 1.0
        self.u_bar = 0.0
        self.sigma = 0.0
        self.t_step = 0

        # state for delta clipping / normalization
        self.clip_mult = 20.0
        self.clip_cap_kavg = 20.0
        self.clip_eta = 0.9998
        self.norm_eta = 0.9998
        self.clip_t = 0
        self.clip_ema_sq = 0.0
        self.norm_t = 0
        self.delta_abs_ema = 0.0

    # ----- internal helpers -----
    def _clip_and_normalize_delta(self, delta: float) -> float:
        # clipping: 20_avg_sq_max_20avg__dec_0.9998
        a = abs(delta)

        prev_avg = 0.0
        if self.clip_t > 0:
            denom = 1.0 - (self.clip_eta ** self.clip_t)
            if denom > 0.0:
                prev_avg = math.sqrt(max(self.clip_ema_sq / denom, 0.0))

        trace_abs = a
        if self.clip_cap_kavg is not None and prev_avg > 0.0:
            trace_abs = min(trace_abs, self.clip_cap_kavg * prev_avg)

        x_clip = trace_abs * trace_abs
        self.clip_t += 1
        self.clip_ema_sq += (1.0 - self.clip_eta) * (x_clip - self.clip_ema_sq)

        curr_avg = 0.0
        denom2 = 1.0 - (self.clip_eta ** self.clip_t)
        if denom2 > 0.0:
            curr_avg = math.sqrt(max(self.clip_ema_sq / denom2, 0.0))

        cap = float("inf")
        if self.clip_mult is not None and curr_avg > 0.0:
            cap = self.clip_mult * curr_avg
        clipped = math.copysign(min(a, cap), delta)

        self.norm_t += 1
        x_norm = abs(clipped)
        self.delta_abs_ema = self.norm_eta * self.delta_abs_ema + (1.0 - self.norm_eta) * x_norm
        denom_norm = 1.0 - self.norm_eta ** self.norm_t
        if denom_norm <= 0.0:
            norm = 1.0
        else:
            norm = max(self.delta_abs_ema / denom_norm, 0.0)

        return clipped / max(norm, 1e-12)

    def step(self, delta, reset=False, avg_grad=None):
        self.t_step += 1

        # 1) entrywise RMSProp stats and norm_grad
        norm_grad = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                v.mul_(self.beta2).addcmul_(p.grad, p.grad, value=1.0 - self.beta2)
                v_hat = (v / (1.0 - self.beta2 ** self.t_step)).sqrt() + 1e-8
                state["rmsprop_v_hat"] = v_hat
                norm_grad += ((p.grad.square() / v_hat).sum()).abs().item()
        if avg_grad is not None  and avg_grad != 0:
            norm_grad = avg_grad

        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                v_hat_for_trace_norm = state["rmsprop_v_hat"]
                e.mul_(self.gamma * self.lamda).add_(p.grad, alpha=1.0)
                z_sum += ((e.square() / v_hat_for_trace_norm).sum()).abs().item()

        # 3) global normalizer (sigma, u, u_bar) and step size
        self.sigma += (1 - self.gamma * self.lamda) * (norm_grad - self.sigma)
        norm_normalizer = math.sqrt((self.sigma / (1 - (self.gamma * self.lamda) ** self.t_step)))
        z_normalizer = math.sqrt(z_sum / (1 - (self.gamma * self.lamda) ** self.t_step))
        u = norm_normalizer * z_normalizer
        self.u_bar += self.u_trace * (u - self.u_bar)
        normalizer_ = max(u, math.sqrt(u * self.u_bar / (1 - (1 - self.u_trace) ** self.t_step)))
        dot_product = self.kappa * normalizer_
        
        #if u==0:
        #    print('u', u,  'norm_normalizer', norm_normalizer, 'norm_grad', norm_grad)
        #    time.sleep(3)
        step_size = 1.0 / (dot_product+1e-12)

        # 4) delta clipping and normalization (policy)
        safe_delta = self._clip_and_normalize_delta(delta)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p.data)
                m = state["momentum"]
                m.mul_(self.momentum).add_(e, alpha=-safe_delta * step_size)
                p.data.addcdiv_(m, state["rmsprop_v_hat"], value=1 - self.momentum)
                if reset:
                    e.zero_()

class OboValue(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.0, kappa=2.0, beta2=0.999, momentum=0.9):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, beta2=beta2, beta_momentum=momentum)
        super().__init__(params, defaults)

        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.beta2 = beta2
        self.momentum = momentum

        self.u_trace = 1.0
        self.u_bar = 0.0
        self.sigma = 0.0
        self.t_step = 0

        # state for delta clipping (value; no normalization)
        self.clip_mult = 20.0
        self.clip_cap_kavg = 20.0
        self.clip_eta = 0.9998
        self.clip_t = 0
        self.clip_ema_sq = 0.0

    # ----- internal helpers -----
    def _clip_delta(self, delta: float) -> float:
        # clipping: 20_avg_sq_max_20avg__dec_0.9998 (no normalization)
        a = abs(delta)

        prev_avg = 0.0
        if self.clip_t > 0:
            denom = 1.0 - (self.clip_eta ** self.clip_t)
            if denom > 0.0:
                prev_avg = math.sqrt(max(self.clip_ema_sq / denom, 0.0))

        trace_abs = a
        if self.clip_cap_kavg is not None and prev_avg > 0.0:
            trace_abs = min(trace_abs, self.clip_cap_kavg * prev_avg)

        x_clip = trace_abs * trace_abs
        self.clip_t += 1
        self.clip_ema_sq += (1.0 - self.clip_eta) * (x_clip - self.clip_ema_sq)

        curr_avg = 0.0
        denom2 = 1.0 - (self.clip_eta ** self.clip_t)
        if denom2 > 0.0:
            curr_avg = math.sqrt(max(self.clip_ema_sq / denom2, 0.0))

        cap = float("inf")
        if self.clip_mult is not None and curr_avg > 0.0:
            cap = self.clip_mult * curr_avg
        return math.copysign(min(a, cap), delta)

    def step(self, delta, reset=False):
        self.t_step += 1

        # 1) entrywise RMSProp stats and norm_grad
        norm_grad = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["entrywise_squared_grad"] = torch.ones_like(p.data)
                v = state["entrywise_squared_grad"]
                v.mul_(self.beta2).addcmul_(p.grad, p.grad, value=1.0 - self.beta2)
                v_hat = (v / (1.0 - self.beta2 ** self.t_step)).sqrt() + 1e-8
                state["rmsprop_v_hat"] = v_hat
                norm_grad += ((p.grad.square() / v_hat).sum()).abs().item()

        # 2) eligibility traces and z_sum
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                v_hat_for_trace_norm = state["rmsprop_v_hat"]
                e.mul_(self.gamma * self.lamda).add_(p.grad, alpha=1.0)
                z_sum += ((e.square() / v_hat_for_trace_norm).sum()).abs().item()

        # 3) global normalizer (sigma, u, u_bar) and step size
        self.sigma += (1 - self.gamma * self.lamda) * (norm_grad - self.sigma)
        norm_normalizer = math.sqrt((self.sigma / (1 - (self.gamma * self.lamda) ** self.t_step)))
        z_normalizer = math.sqrt(z_sum / (1 - (self.gamma * self.lamda) ** self.t_step))
        u = norm_normalizer * z_normalizer
        self.u_bar += self.u_trace * (u - self.u_bar)
        normalizer_ = max(u, math.sqrt(u * self.u_bar / (1 - (1 - self.u_trace) ** self.t_step)))
        dot_product = self.kappa * normalizer_
        step_size = 1.0 / dot_product

        # 4) delta clipping (value; no normalization)
        safe_delta = self._clip_delta(delta)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p.data)
                m = state["momentum"]
                m.mul_(self.momentum).add_(e, alpha=-safe_delta * step_size)
                p.data.addcdiv_(m, state["rmsprop_v_hat"], value=1 - self.momentum)
                if reset:
                    e.zero_()


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
