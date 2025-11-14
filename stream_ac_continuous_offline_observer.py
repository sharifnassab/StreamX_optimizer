import time
start_time = time.time()
import os, pickle, argparse
os.environ["GYM_DISABLE_PLUGIN_AUTOLOAD"] = "1"
os.environ["WANDB_DISABLE_SYSTEM_METRICS"] = "true"
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import wandb  # For wandb-specific tweaks if needed
import glob

# --- Optimizers ---
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
from optim import Obtnnzm as Obtnnzm_Optimizer
from optim import Obtm as Obtm_Optimizer

# --- Utilities ---
from sparse_init import sparse_init
from logger import get_logger  # For wandb / other backends
print('All imports done.')


def initialize_weights(m, sparsity=0.9):
    """Initializes network weights with sparsity."""
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=sparsity)
        m.bias.data.fill_(0.0)


class ValueNetwork(nn.Module):
    """Value Network (used for the Observer)."""
    def __init__(self, n_obs=11, hidden_depth=2, hidden_width=128, initialization_sparsity=0.9):
        super(ValueNetwork, self).__init__()
        self.fc_layer = nn.Linear(n_obs, hidden_width)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_width, hidden_width) for _ in range(hidden_depth - 1)])
        self.linear_layer = nn.Linear(hidden_width, 1)
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
    """Computes the L2 norm of model parameters."""
    total = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total += torch.sum(p ** 2)
    return torch.sqrt(total).item()


def spec_to_name(spec: dict) -> str:
    """Converts a spec dictionary to a short string name."""
    short_hand_mapping = {
        'optimizer': '',
        'hidden_depth': '_net',
        'hidden_width': 'x',
        'initialization_sparsity': 'sp',
        'lamda': '_lam',
        'kappa': 'k',
        'gamma': 'gam',
        'entrywise_normalization': 'en',
        'beta2': 'b2',
        'u_trace': 'u',
        'momentum': 'm',
        'lr': 'lr',
        'weight_decay': 'wd',
        'delta_trace': 'delTr',
        'sig_power': 'sigP',
        'in_trace_sample_scaling': 'itss',
        'delta_clip': 'delClip',
        'delta_norm': 'delNorm',
    }
    list_params = []
    for key in short_hand_mapping:
        if key in spec:
            list_params.append(f"{short_hand_mapping[key]}_{spec[key]}")
    return '_'.join(list_params) if list_params else ''


class OfflineObserver(nn.Module):
    """
    An agent that only contains an observer, to be trained on offline data.
    """
    def __init__(self, n_obs: int, observer_spec: dict):
        super().__init__()

        self.observer_spec = dict(observer_spec or {})
        gamma_val = self.observer_spec.get('gamma', None)
        if gamma_val is None:
            raise ValueError("observer_spec must contain 'gamma' for OfflineObserver.")
        self.gamma_observer = float(gamma_val)

        if self.observer_spec.get('optimizer', '').lower() == 'monte_carlo':
            self.is_monte_carlo = True
            self.observer_net = None
            self.optimizer_observer = None
            print("Running in 'monte_carlo' mode. No network will be trained.")
        else:
            self.is_monte_carlo = False
            self.observer_net = ValueNetwork(
                n_obs=n_obs,
                hidden_depth=self.observer_spec['hidden_depth'],
                hidden_width=self.observer_spec['hidden_width'],
                initialization_sparsity=self.observer_spec['initialization_sparsity'],
            )
            self.optimizer_observer = self._build_optimizer(self.observer_net.parameters(), self.observer_spec, role="observer")

    def _build_optimizer(self, params, spec: dict, role: str):
        spec = spec or {}
        opt_name = spec.get('optimizer', 'none').strip().lower()

        lr = float(spec.get('lr', 3e-4))
        gamma = float(spec.get('gamma'))
        lamda = float(spec.get('lamda'))
        kappa = float(spec.get('kappa'))
        weight_decay = float(spec.get('weight_decay', 0.0))
        momentum = float(spec.get('momentum', 0.0))
        u_trace = float(spec.get('u_trace', 0.01))
        entrywise_normalization = spec.get('entrywise_normalization', 'none')
        beta2 = float(spec.get('beta2', 0.999))
        delta_trace = float(spec.get('delta_trace', 0.01))
        sig_power = float(spec.get('sig_power', 2))
        
        # --- FIX 3: Corrected boolean parsing ---
        # Convert string 'True'/'False' from argparse to boolean
        in_trace_sample_scaling = spec.get('in_trace_sample_scaling', 'False') == 'True'
        
        delta_clip = spec.get('delta_clip', 'none')
        delta_norm = spec.get('delta_norm', 'none')

        if opt_name == 'obgd':
            return ObGD_Optimizer(params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        if opt_name in ('adaptiveobgd', 'adaptive_obgd'):
            return AdaptiveObGD_Optimizer(params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        if opt_name == 'obgd_sq':
            return ObGDsq_Optimizer(params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        if opt_name == 'obgd_sq_plain':
            return ObGDsqPlain_Optimizer(params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        if opt_name == 'obn':
            return Obn_Optimizer(params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, u_trace=u_trace,
                                 entrywise_normalization=entrywise_normalization, beta2=beta2)
        if opt_name == 'obnc':
            return ObnC_Optimizer(params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, u_trace=u_trace,
                                  entrywise_normalization=entrywise_normalization, beta2=beta2)
        if opt_name == 'obnn':
            return ObnN_Optimizer(params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, delta_trace=delta_trace,
                                  u_trace=u_trace, entrywise_normalization=entrywise_normalization, beta2=beta2)
        if opt_name == 'obtc':
            return ObtC_Optimizer(params, gamma=gamma, lamda=lamda, kappa=kappa, weight_decay=weight_decay,
                                  sig_power=sig_power, entrywise_normalization=entrywise_normalization, beta2=beta2,
                                  in_trace_sample_scaling=in_trace_sample_scaling)
        if opt_name == 'obtn':
            return ObtN_Optimizer(params, gamma=gamma, lamda=lamda, kappa=kappa, weight_decay=weight_decay,
                                  sig_power=sig_power, delta_trace=delta_trace,
                                  entrywise_normalization=entrywise_normalization, beta2=beta2,
                                  in_trace_sample_scaling=in_trace_sample_scaling)
        if opt_name == 'obt':
            return Obt_Optimizer(params, gamma=gamma, lamda=lamda, kappa=kappa, weight_decay=weight_decay,
                                 sig_power=sig_power, delta_clip=delta_clip, delta_norm=delta_norm,
                                 entrywise_normalization=entrywise_normalization, beta2=beta2,
                                 in_trace_sample_scaling=in_trace_sample_scaling)
        if opt_name == 'obtnnz':
            return Obtnnz_Optimizer(params, gamma=gamma, lamda=lamda, kappa=kappa, weight_decay=weight_decay,
                                    delta_clip=delta_clip, delta_norm=delta_norm, u_trace=u_trace,
                                    entrywise_normalization=entrywise_normalization, beta2=beta2)
        if opt_name == 'obtnnzm':
            return Obtnnzm_Optimizer(params, gamma=gamma, lamda=lamda, kappa=kappa, weight_decay=weight_decay,
                                     delta_clip=delta_clip, delta_norm=delta_norm, momentum=momentum, u_trace=u_trace,
                                     entrywise_normalization=entrywise_normalization, beta2=beta2)
        if opt_name == 'obgdn':
            return ObGDN_Optimizer(params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa,
                                   delta_clip=delta_clip, delta_norm=delta_norm)
        if opt_name == 'obgdm':
            return ObGDm_Optimizer(params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, momentum=momentum)
        if opt_name == 'obtcm':
            return ObtCm_Optimizer(params, gamma=gamma, lamda=lamda, kappa=kappa, weight_decay=weight_decay,
                                   sig_power=sig_power, momentum=momentum,
                                   entrywise_normalization=entrywise_normalization, beta2=beta2,
                                   in_trace_sample_scaling=in_trace_sample_scaling)
        if opt_name == 'obtm':
            return Obtm_Optimizer(params, gamma=gamma, lamda=lamda, kappa=kappa, weight_decay=weight_decay,
                                  sig_power=sig_power, delta_clip=delta_clip, delta_norm=delta_norm,
                                  momentum=momentum, entrywise_normalization=entrywise_normalization,
                                  beta2=beta2, in_trace_sample_scaling=in_trace_sample_scaling)

        raise ValueError(f"Unknown optimizer '{spec.get('optimizer')}' for role '{role}'.")

    def v_observer(self, x):
        """Get the observer's value prediction."""
        return self.observer_net(x)

    def update_observer(self, s, r, s_prime, done):
        """Performs a single observer update from an offline transition."""
        done_mask = 0.0 if done else 1.0

        s = torch.tensor(np.array(s), dtype=torch.float)
        r = torch.tensor(np.array(r), dtype=torch.float)
        s_prime = torch.tensor(np.array(s_prime), dtype=torch.float)
        done_mask_t = torch.tensor(done_mask, dtype=torch.float)

        self.optimizer_observer.zero_grad()

        v_s_obs = self.v_observer(s)
        with torch.no_grad():
            v_prime_obs = self.v_observer(s_prime)

        delta_obs = r + self.gamma_observer * v_prime_obs * done_mask_t - v_s_obs

        (-v_s_obs).backward()

        info_from_optim = self.optimizer_observer.step(delta_obs.item(), reset=done)

        info_to_return = info_from_optim or {}
        info_to_return.update({
            "delta": float(delta_obs.item()),
            "v(s)": float(v_s_obs.item()),
        })

        return info_to_return


def compute_monte_carlo_v_at_the_end_of_episode(list_ep_R, gamma):
    """Calculates discounted returns (G) for an episode."""
    G = 0.0
    list_v = []
    for r in reversed(list_ep_R):
        G = r + gamma * G
        list_v.append(G)
    return [v for v in reversed(list_v)]


def compute_prediction_MSE(list_ep_R, list_ep_v, gamma):
    """Computes Mean Squared Error against Monte-Carlo returns."""
    G = 0.0
    ep_sse = 0.0
    ep_abss = 0.0
    for v, r in zip(reversed(list_ep_v), reversed(list_ep_R)):
        G = r + gamma * G
        err = G - v
        ep_sse += err * err
        ep_abss += abs(err)
    n = len(list_ep_R)
    if n == 0:
        return {'ep_MSE_error': 0.0, 'ep_abs_error': 0.0}
    return {'ep_MSE_error': ep_sse / n, 'ep_abs_error': ep_abss / n}


def compute_prediction_MSE_end_of_episode_W(list_ep_R, list_ep_S, observer_net, gamma):
    """Computes MSE at the end of an episode with fresh V-estimates."""
    if not list_ep_S:
        return {'ep_MSE_error': 0.0, 'ep_abs_error': 0.0}

    with torch.no_grad():
        s_batch = torch.tensor(np.array(list_ep_S), dtype=torch.float)
        v_batch = observer_net(s_batch)
        list_ep_v = v_batch.squeeze(-1).cpu().numpy().tolist()

    return compute_prediction_MSE(list_ep_R, list_ep_v, gamma)


class BinnedNpzDataset:
    """
    Stream transitions from a directory of .npz bins created by save_dataset_bin_npz.

    Exposes:
        - env_name
        - data_seed
        - n_obs
        - total_steps
        - step()  -> (s, a, r_scaled, r_actual, s_prime, done)
    """
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.file_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
        if not self.file_paths:
            raise FileNotFoundError(f"No .npz files found in directory: {dataset_dir}")

        # --- Read metadata from first bin ---
        with np.load(self.file_paths[0]) as meta:
            # env_name and seed were saved as 0-d arrays
            env_name_arr = meta["env_name"]
            seed_arr = meta["seed"]
            states_arr = meta["states"]

            self.env_name = str(env_name_arr.item() if np.ndim(env_name_arr) == 0 else env_name_arr)
            self.data_seed = int(seed_arr.item() if np.ndim(seed_arr) == 0 else seed_arr)
            self.n_obs = int(states_arr.shape[1])

        # --- Count total transitions across all bins (cheap: only read shapes) ---
        self.total_steps = 0
        for path in self.file_paths:
            with np.load(path) as d:
                self.total_steps += int(d["states"].shape[0])

        # --- Streaming state ---
        self._current_file_idx = 0
        self._idx_in_file = 0
        self._current_npz = None
        self._file_length = 0

        self._states = None
        self._actions = None
        self._r_scaled = None
        self._r_actual = None
        self._next_states = None
        self._dones = None

        # Load the first file
        self._load_file(self._current_file_idx)

    def _load_file(self, file_idx: int):
        # Close previous npz file if open
        if self._current_npz is not None:
            self._current_npz.close()

        path = self.file_paths[file_idx]
        self._current_npz = np.load(path)
        self._states = self._current_npz["states"]
        self._actions = self._current_npz["actions"]
        self._r_scaled = self._current_npz["rewards_scaled"]
        self._r_actual = self._current_npz["rewards_actual"]
        self._next_states = self._current_npz["next_states"]
        self._dones = self._current_npz["dones"]

        self._file_length = int(self._states.shape[0])
        self._idx_in_file = 0

        print(f"Loaded bin {file_idx} from '{os.path.basename(path)}' with {self._file_length} transitions.")

    def step(self):
        """
        Return the next transition:
            (s, a, r_scaled, r_actual, s_prime, done)

        Raises StopIteration when the dataset is exhausted.
        """
        # End of current file: move to next one
        if self._idx_in_file >= self._file_length:
            self._current_file_idx += 1
            if self._current_file_idx >= len(self.file_paths):
                # No more bins
                raise StopIteration
            self._load_file(self._current_file_idx)

        i = self._idx_in_file
        self._idx_in_file += 1

        s = self._states[i]
        a = self._actions[i]
        r_scaled = float(self._r_scaled[i])
        r_actual = float(self._r_actual[i])
        s_prime = self._next_states[i]
        done = bool(self._dones[i])

        return s, a, r_scaled, r_actual, s_prime, done


class LegacyPickleDataset:
    """
    Backward-compatible wrapper for the old pickled format:

        data = {
            'dataset': list of (s, a, r_scaled, r_actual, s_prime, done),
            'env_name': ...,
            'seed': ...
        }

    Exposes same API as BinnedNpzDataset.
    """
    def __init__(self, dataset_path: str):
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

        dataset_list = data["dataset"]
        if not dataset_list:
            raise ValueError("Legacy pickle dataset is empty.")

        self._dataset = dataset_list
        self.env_name = data.get("env_name", "unknown_env")
        self.data_seed = data.get("seed", "unknown")

        s0, _, _, _, _, _ = dataset_list[0]
        self.n_obs = int(np.array(s0).shape[0])
        self.total_steps = len(dataset_list)
        self._cursor = 0

    def step(self):
        if self._cursor >= self.total_steps:
            raise StopIteration
        transition = self._dataset[self._cursor]
        self._cursor += 1
        return transition


def load_offline_dataset(dataset_path: str):
    """
    Helper that hides whether we are reading new-style .npz bins
    or old-style pickle files.
    """
    if os.path.isdir(dataset_path):
        print(f"Detected directory '{dataset_path}', assuming binned .npz format.")
        ds = BinnedNpzDataset(dataset_path)
        print(f"Found {len(ds.file_paths)} bins with total {ds.total_steps} transitions.")
        return ds

    # Fallback: old pickled dataset
    print(f"Detected file '{dataset_path}', assuming legacy pickle format.")
    ds = LegacyPickleDataset(dataset_path)
    print(f"Loaded legacy dataset with {ds.total_steps} transitions from env '{ds.env_name}' (data seed: {ds.data_seed}).")
    return ds


def main(dataset_path, total_steps, seed, observer_spec, logging_spec, debug=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_offline_dataset(dataset_path)
    except FileNotFoundError:
        print(f"Error: dataset path not found: {dataset_path}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if dataset.total_steps <= 0:
        print("Error: Dataset is empty (total_steps == 0).")
        return

    # We now get n_obs directly from the dataset object
    try:
        n_obs = int(dataset.n_obs)
        print(f"Inferred n_obs = {n_obs}")
    except Exception as e:
        print(f"Error inferring n_obs from dataset: {e}")
        return

    env_name = dataset.env_name
    data_seed = dataset.data_seed

    agent = OfflineObserver(n_obs=n_obs, observer_spec=observer_spec)
    gamma_obs = agent.gamma_observer  # unused here but kept for consistency

    log_freq = int(logging_spec.get('log_freq', 20_000))
    bin_size = int(logging_spec.get('bin_size', 50_000))
    dataset_name = logging_spec.get('dataset_name') + f'__seed{seed}'

    if log_freq <= 0:
        print("Warning: log_freq <= 0 detected. Group 2 step-level logging will be disabled.")
    if bin_size <= 0:
        print("Warning: bin_size <= 0 detected. Group 1 binning may behave unexpectedly.")

    run_name = (f'{env_name}____dataset_{dataset_name}____-Observer_{spec_to_name(observer_spec)}' +
                (f"____{logging_spec.get('run_name', '')}" if (logging_spec.get('run_name', '') != '') else ''))


    config = {
        "env_name": env_name,
        "observer_init_seed": seed,
        "data_seed": data_seed,
        "data_dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "observer": observer_spec,
        "run_name": run_name,
        "log_freq": log_freq,
        "bin_size": bin_size,
    }
    print(f"Run Name: {run_name}")
    print(f"Observer Init Seed: {seed}")

    logger = get_logger(
        backend=logging_spec.get('backend', 'wandb'),
        log_dir=logging_spec.get('dir', 'runs'),
        project=logging_spec.get('project', 'StreamX_OptDesign'),
        run_name=run_name + f'__seed{seed}',
        config=config,
    )

    try:
        if wandb.run is not None:
            pass
    except Exception:
        pass

    start_time = time.time()

    # Unified queue
    final_log_queue = []  # (step, payload_dict)

    # Group 1 episodic storage
    episodic_log_data = []

    # Group 1 accumulators
    list_ep_R_scaled, list_ep_v, list_ep_S = [], [], []

    # Group 2 settings
    desired_to_log = ['v(s)', 'delta']
    next_log_trigger_step = 0
    is_new_episode = True
    logging_this_episode = False
    current_log_milestone_name = ""

    # Relative step counter per episode
    current_episode_step = 0

    episode_number = 0
    total_steps = dataset.total_steps

    print(f"Starting observer training on {total_steps} transitions...")

    t = -1  # global step index
    while t<total_steps:
        t += 1
        try:
            s, a, r_scaled, r_actual, s_prime, done = dataset.step()
        except StopIteration:
            # We have exhausted all bins / transitions
            break

        if is_new_episode:
            is_new_episode = False
            current_episode_step = 0  # Reset relative step counter

            if log_freq > 0 and t >= next_log_trigger_step:
                logging_this_episode = True
                current_log_milestone_name = f"{np.round(next_log_trigger_step / 1_000_000.0, 2):.2f}m"
                next_log_trigger_step += log_freq
                if debug:
                    print(f"\nStep {t}: [Debug] Starting to log episode for milestone {current_log_milestone_name}")
            else:
                logging_this_episode = False

        if agent.is_monte_carlo:
            step_info = {}
        else:
            step_info = agent.update_observer(s, r_scaled, s_prime, done)

        if logging_this_episode and not agent.is_monte_carlo and log_freq > 0:
            payload_for_this_step = {}
            for key, value in step_info.items():
                if key in desired_to_log:
                    payload_for_this_step[f"{key}/{current_log_milestone_name}"] = value
            if payload_for_this_step:
                # Log against relative step, not global step 't'
                final_log_queue.append((int(current_episode_step), payload_for_this_step))

        list_ep_R_scaled.append(r_scaled)
        list_ep_S.append(s)
        if not agent.is_monte_carlo:
            list_ep_v.append(step_info.get('v(s)', 0.0))

        # Episode end or end-of-dataset
        if done or (t == total_steps - 1):

            if logging_this_episode and agent.is_monte_carlo and log_freq > 0:
                monte_carlo_v = compute_monte_carlo_v_at_the_end_of_episode(list_ep_R_scaled, agent.gamma_observer)
                ep_len = len(list_ep_R_scaled)

                for i in range(ep_len):
                    relative_step = i
                    payload_for_this_step = {
                        f"v(s)/{current_log_milestone_name}": monte_carlo_v[i],
                        f"delta/{current_log_milestone_name}": list_ep_R_scaled[i] / 10.0
                    }
                    final_log_queue.append((int(relative_step), payload_for_this_step))

            # Make sure Group2 logging does not carry over episodes
            logging_this_episode = False

            # --- Group 1 (Binned) Logic ---
            if agent.is_monte_carlo:
                log_payload = {
                    "prediction/episode_MSE": None,
                    "prediction/episode_abs": None,
                    "prediction/episode_RMSE": None,
                    "prediction/episode_MSE_end_of_episode_W": None,
                    "prediction/episode_abs_end_of_episode_W": None,
                    "prediction/episode_RMSE_end_of_episode_W": None,
                    "network/observer_w_norm": None
                }
            else:
                ep_pred_error = compute_prediction_MSE(list_ep_R_scaled, list_ep_v, agent.gamma_observer)
                ep_pred_error_end_of_episode_W = compute_prediction_MSE_end_of_episode_W(
                    list_ep_R_scaled, list_ep_S, agent.observer_net, agent.gamma_observer
                )

                log_payload = {
                    "prediction/episode_MSE": float(ep_pred_error['ep_MSE_error']),
                    "prediction/episode_abs": float(ep_pred_error['ep_abs_error']),
                    "prediction/episode_RMSE": float(np.sqrt(ep_pred_error['ep_MSE_error'])),
                    "prediction/episode_MSE_end_of_episode_W": float(ep_pred_error_end_of_episode_W['ep_MSE_error']),
                    "prediction/episode_abs_end_of_episode_W": float(ep_pred_error_end_of_episode_W['ep_abs_error']),
                    "prediction/episode_RMSE_end_of_episode_W": float(
                        np.sqrt(ep_pred_error_end_of_episode_W['ep_MSE_error'])
                    ),
                }

                if episode_number % 40 == 0 and agent.observer_net is not None:
                    log_payload.update({"network/observer_w_norm": compute_weight_norm(agent.observer_net)})

            # Group 1 logs still use global step
            log_payload['step'] = int(t + 1)
            episodic_log_data.append(log_payload)

            if debug:
                rmse_val = log_payload.get('prediction/episode_RMSE')
                rmse_str = f"{rmse_val:.4f}" if rmse_val is not None else "N/A (MC)"
                print(f"[Debug] End of Ep {episode_number} at Step {t+1} | RMSE: {rmse_str}")

            list_ep_R_scaled, list_ep_v, list_ep_S = [], [], []
            episode_number += 1
            is_new_episode = True

        # Increment the relative step counter at the end of every step
        current_episode_step += 1

    print(f"--- Finished processing all {total_steps} steps ---")
    print(f"Total time = {time.gmtime(int(time.time() - start_time))}")

    print("\n--- Processing and logging data to wandb ---")
    # ... (rest of your logging / flush code unchanged)








    # --- Group 1 Binning (Unchanged) ---
    if episodic_log_data:
        print("Processing Group 1 (Episodic Binned) data...")
        max_step = max(d['step'] for d in episodic_log_data)
        if bin_size > 0:
            num_bins = int(np.ceil(max_step / bin_size))
        else:
            num_bins = 1

        for i in range(1, num_bins + 1):
            bin_start = (i - 1) * bin_size + 1 if bin_size > 0 else 1
            bin_end = i * bin_size if bin_size > 0 else max_step

            bin_data = [d for d in episodic_log_data if bin_start <= d['step'] <= bin_end]

            if bin_data:
                binned_payload = {}
                all_keys = set()
                for d in bin_data:
                    all_keys.update(d.keys())

                for key in all_keys:
                    if key == 'step':
                        continue
                    values = [d[key] for d in bin_data if key in d and d.get(key) is not None]
                    if values:
                        binned_payload[key] = float(np.mean(values))

                if binned_payload:
                    print(f"  Processing bin {i}/{num_bins} (Step {bin_end})...")
                    # Group 1 plots are logged against their global bin_end step
                    final_log_queue.append((int(bin_end), binned_payload))
        print("Group 1 processing complete.")
    
    # --- Final Sync Loop (Unchanged) ---
    # This logic is now correct. It will sort all logs by their step.
    # Group 1 (binned) plots will be at steps 50k, 100k, etc.
    # Group 2 (relative) plots will be at steps 0, 1, 2, 3...
    print(f"Sorting {len(final_log_queue)} log entries...")
    final_log_queue.sort(key=lambda x: int(x[0]))

    # Merge payloads per step so each step is logged exactly once
    merged_by_step = {}
    for step, payload in final_log_queue:
        s = int(step)
        if s not in merged_by_step:
            merged_by_step[s] = {}
        merged_by_step[s].update(payload)

    sorted_steps = sorted(merged_by_step.keys())
    print(f"Logging {len(sorted_steps)} unique steps to wandb in order...")

    for s in sorted_steps:
        raw_payload = merged_by_step[s]
        clean_payload = {}
        for k, v in raw_payload.items():
            if isinstance(v, (np.floating, np.integer)):
                clean_payload[k] = float(v)
            elif isinstance(v, np.ndarray):
                if v.shape == ():
                    clean_payload[k] = float(v)
                else:
                    clean_payload[k] = v
            else:
                clean_payload[k] = v

        logger.log(clean_payload, step=int(s))

    print("Finishing wandb run (this may take a moment)...")
    logger.finish()
    print("--- Wandb logging finished and sync complete ---")
    

    if logging_spec['dir_pickle'] != 'none':
        save_dir = os.path.join(logging_spec['dir_pickle'], dataset_name,  run_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
            pickle.dump((merged_by_step, sorted_steps, env_name), f)


if __name__ == '__main__':
    optimizer_choices = [
        'ObGD', 'ObGD_sq', 'ObGD_sq_plain', 'Obn', 'ObnC', 'ObnN',
        'AdaptiveObGD', 'ObtC', 'ObtN', 'Obt', 'Obtnnz', 'Obtnnzm',
        'ObGDN', 'ObGDm', 'ObtCm', 'Obtm', 'monte_carlo'
    ]

    parser = argparse.ArgumentParser(description='Offline Observer Training')

    # Core Arguments
    parser.add_argument('--dataset_path', type=str, default='/Users/arsalan/Desktop/Codes/StreamX/offline_dataset/ObGD_ObGD_lam_0.8__', help='Path to the offline dataset .pkl file to load.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the observer network initialization.')
    parser.add_argument('--total_steps', type=int, default=5_000_000, help='Total number of steps to train the observer.')
    parser.add_argument('--max_time', type=str, default="", help='Not effective in this code.')
    parser.add_argument('--env_name', type=str, default='Ant')  # Only affects saving firectory
    
    # Observer Arguments
    parser.add_argument('--observer_hidden_depth', type=int, default=2)
    parser.add_argument('--observer_hidden_width', type=int, default=128)
    parser.add_argument('--observer_initialization_sparsity', type=float, default=0.9)
    parser.add_argument('--observer_optimizer', type=str, default='ObGD', choices=optimizer_choices)
    parser.add_argument('--observer_kappa', type=float, default=2.0)
    parser.add_argument('--observer_gamma', type=float, default=0.99)
    parser.add_argument('--observer_lamda', type=float, default=0.0)
    parser.add_argument('--observer_lr', type=float, default=1.0)
    parser.add_argument('--observer_momentum', type=float, default=0.0)
    parser.add_argument('--observer_u_trace', type=float, default=0.01)
    parser.add_argument('--observer_entrywise_normalization', type=str, default='RMSProp')
    parser.add_argument('--observer_beta2', type=float, default=0.999)
    parser.add_argument('--observer_delta_trace', type=float, default=0.01)
    parser.add_argument('--observer_weight_decay', type=float, default=0.0)
    parser.add_argument('--observer_in_trace_sample_scaling', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--observer_sig_power', type=float, default=2)
    parser.add_argument('--observer_delta_clip', type=str, default='none')
    parser.add_argument('--observer_delta_norm', type=str, default='none')

    # Logging Arguments
    parser.add_argument('--log_backend', type=str, default='wandb', choices=['none', 'tensorboard', 'wandb', 'wandb_offline'])
    parser.add_argument('--log_dir', type=str, default='./wandb_offline', help='WandB offline log dir (if backend=wandb_offline)')
    parser.add_argument('--log_dir_for_pickle', type=str, default='none', help='/home/asharif/StreamX_optimizer/pickle')
    parser.add_argument('--project', type=str, default='StreamX_Offline_Observer', help='WandB project')
    parser.add_argument('--run_name', type=str, default='', help='Run name for logger')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_freq', type=int, default=20_000, help='Frequency for logging Group 2 plots.')
    parser.add_argument('--bin_size', type=int, default=50_000, help='Bin size for averaging Group 1 plots.')
    parser.add_argument('--uID', type=str, default='', help='')  # not used
    parser.add_argument('--dataset_name', type=str, default='', help='Name of the dataset for logging purposes (pickle save).')

    args = parser.parse_args()

    shared_params = ['optimizer', 'kappa', 'gamma', 'lamda', 'weight_decay', 'hidden_depth', 'hidden_width', 'initialization_sparsity']
    required_optimizer_params = {
        'monte_carlo': ['gamma'],
        'ObGD': shared_params + ['lr'],
        'ObGD_sq': shared_params + ['lr'],
        'ObGD_sq_plain': shared_params + ['lr'],
        'AdaptiveObGD': shared_params + ['lr'],
        'Obn': shared_params + ['lr', 'entrywise_normalization', 'beta2', 'u_trace'],
        'ObnC': shared_params + ['lr', 'entrywise_normalization', 'beta2', 'u_trace'],
        'ObnN': shared_params + ['lr', 'entrywise_normalization', 'beta2', 'u_trace', 'delta_trace'],
        'ObtC': shared_params + ['entrywise_normalization', 'beta2', 'sig_power', 'in_trace_sample_scaling'],
        'ObtCm': shared_params + ['entrywise_normalization', 'beta2', 'sig_power', 'in_trace_sample_scaling', 'momentum'],
        'ObtN': shared_params + ['entrywise_normalization', 'beta2', 'sig_power', 'in_trace_sample_scaling', 'delta_trace'],
        'Obt': shared_params + ['entrywise_normalization', 'beta2', 'sig_power', 'in_trace_sample_scaling', 'delta_clip', 'delta_norm'],
        'Obtm': shared_params + ['entrywise_normalization', 'beta2', 'sig_power', 'in_trace_sample_scaling', 'delta_clip', 'delta_norm', 'momentum'],
        'Obtnnz': shared_params + ['entrywise_normalization', 'beta2', 'u_trace', 'delta_clip', 'delta_norm'],
        'Obtnnzm': shared_params + ['entrywise_normalization', 'beta2', 'u_trace', 'delta_clip', 'delta_norm', 'momentum'],
        'ObGDN': shared_params + ['lr', 'delta_clip', 'delta_norm'],
        'ObGDm': shared_params + ['lr', 'momentum'],
    }

    def build_spec(kind, args, required_optimizer_params) -> dict:
        opt = getattr(args, f'{kind}_optimizer')
        required_params = required_optimizer_params.get(opt, [])
        spec = {'optimizer': opt}
        for key in required_params:
            if hasattr(args, f'{kind}_{key}'):
                spec[key] = getattr(args, f'{kind}_{key}')
        return spec

    observer_spec = build_spec(kind='observer', args=args, required_optimizer_params=required_optimizer_params)

    logging_spec = {
        'backend': args.log_backend,
        'dir_pickle': args.log_dir_for_pickle,
        'dir': f'{args.log_dir}_{args.env_name + ("" if args.env_name.endswith("v5") else "-v5")}', # if env_name ends with 'v5', do not add extra '-v5'
        'project': args.project,
        'run_name': args.run_name,
        'log_freq': args.log_freq,
        'bin_size': args.bin_size,
        'dataset_name': args.dataset_name,
    }

    main(dataset_path=args.dataset_path, total_steps=args.total_steps, seed=args.seed, observer_spec=observer_spec, logging_spec=logging_spec, debug=args.debug)