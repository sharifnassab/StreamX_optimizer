import torch
import math

def activation_function(name):
    if name.lower() in ['identity', 'none']:
        return lambda x: x
    elif name.lower() in ['sigmoid']:
        return lambda x: 1/(1+torch.exp(-torch.clamp(x, -500, 500)))
    elif name.lower() in ['exp']:
        return lambda x: torch.exp(torch.clamp(x, -500, 500))
    else:
        raise(ValueError(f'     {name}     not supported'))
    
def activation_function_inverse(name, y):
    y = torch.tensor(y)
    if name.lower() in ['identity', 'none']:
        return y
    elif name.lower() in ['sigmoid']:
        return -torch.log(1/y -1)
    elif name.lower() in ['exp']:
        return torch.log(torch.clamp(y, 1e-20, 1e20))
    else:
        raise(ValueError(f'     {name}     not supported'))
    
def clip_zeta_meta_function(name='none'):  # 'minmax_10'
    name_lower = name.lower()
    if name_lower in ['none', 'no_clip']:
        return lambda x: x
    elif ('etamin' in name_lower) or ('etamax' in name_lower):
        th_min = math.log(float((name_lower+'_').split('etamin_')[-1].split('_')[0])) if 'etamin' in name_lower else None
        th_max = math.log(float((name_lower+'_').split('etamax_')[-1].split('_')[0])) if 'etamax' in name_lower else None
        return lambda x: torch.clamp(x, min=th_min, max=th_max)
    elif name_lower.startswith('minmax_'):
        th = float(name.split('_')[1])
        return lambda x: torch.clamp(x, min=-th, max=th)
    elif name_lower.startswith('min_'):
        th = float(name.split('_')[1])
        return lambda x: torch.clamp(x, min=th)
    elif name_lower.startswith('max_'):
        th = float(name.split('_')[1])
        return lambda x: torch.clamp(x, max=th)
    else:
        raise(ValueError(f'     {name}     not supported'))

