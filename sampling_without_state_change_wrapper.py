import copy
from collections import deque
from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, ScaleReward
 

import numpy as np
import mujoco


def _walk_wrappers(env):
    cur = env
    while True:
        yield cur
        if not hasattr(cur, "env"):
            break
        cur = cur.env


def _copy_np(x):
    return None if x is None else np.array(x, copy=True)


def _copy_deque(x):
    return deque(list(x), maxlen=x.maxlen)


def _get_np_random_state(obj):
    rng = getattr(obj, "np_random", None)
    if rng is None:
        return None
    try:
        return copy.deepcopy(rng.bit_generator.state)
    except Exception:
        return None


def _set_np_random_state(obj, state):
    if state is None:
        return
    rng = getattr(obj, "np_random", None)
    if rng is None:
        return
    try:
        rng.bit_generator.state = copy.deepcopy(state)
    except Exception:
        pass


def _snapshot_sample_mean_std(stats):
    return {
        "mean": _copy_np(stats.mean),
        "var": _copy_np(stats.var),
        "p": _copy_np(stats.p),
        "count": int(stats.count),
    }


def _restore_sample_mean_std(stats, snap):
    stats.mean = _copy_np(snap["mean"])
    stats.var = _copy_np(snap["var"])
    stats.p = _copy_np(snap["p"])
    stats.count = int(snap["count"])


def _snapshot_wrapper_state(env):
    """
    Snapshot all mutable Python-side state in the wrapper stack that can change on step().
    """
    snap = []

    for w in _walk_wrappers(env):
        ws = {
            "__class__": type(w).__name__,
            "__np_random_state__": _get_np_random_state(w),
        }

        # Your custom wrappers
        if isinstance(w, AddTimeInfo):
            ws["epi_time"] = float(w.epi_time)

        if isinstance(w, NormalizeObservation):
            ws["obs_stats"] = _snapshot_sample_mean_std(w.obs_stats)

        if isinstance(w, ScaleReward):
            ws["reward_trace"] = _copy_np(w.reward_trace)
            ws["reward_stats"] = _snapshot_sample_mean_std(w.reward_stats)

        # Gymnasium TimeLimit
        if hasattr(w, "_elapsed_steps"):
            ws["_elapsed_steps"] = copy.deepcopy(w._elapsed_steps)

        # Gymnasium RecordEpisodeStatistics (single-env and vector-safe fields)
        if hasattr(w, "episode_count"):
            ws["episode_count"] = copy.deepcopy(w.episode_count)
        if hasattr(w, "episode_start_times"):
            ws["episode_start_times"] = _copy_np(w.episode_start_times)
        if hasattr(w, "episode_returns"):
            ws["episode_returns"] = _copy_np(w.episode_returns)
        if hasattr(w, "episode_lengths"):
            ws["episode_lengths"] = _copy_np(w.episode_lengths)
        if hasattr(w, "return_queue"):
            ws["return_queue"] = _copy_deque(w.return_queue)
        if hasattr(w, "length_queue"):
            ws["length_queue"] = _copy_deque(w.length_queue)
        if hasattr(w, "time_queue"):
            ws["time_queue"] = _copy_deque(w.time_queue)

        snap.append((w, ws))

    return snap


def _restore_wrapper_state(snapshot):
    for w, ws in snapshot:
        _set_np_random_state(w, ws.get("__np_random_state__"))

        if isinstance(w, AddTimeInfo) and "epi_time" in ws:
            w.epi_time = float(ws["epi_time"])

        if isinstance(w, NormalizeObservation) and "obs_stats" in ws:
            _restore_sample_mean_std(w.obs_stats, ws["obs_stats"])

        if isinstance(w, ScaleReward):
            if "reward_trace" in ws:
                w.reward_trace = _copy_np(ws["reward_trace"])
            if "reward_stats" in ws:
                _restore_sample_mean_std(w.reward_stats, ws["reward_stats"])

        if "_elapsed_steps" in ws:
            w._elapsed_steps = copy.deepcopy(ws["_elapsed_steps"])

        if "episode_count" in ws:
            w.episode_count = copy.deepcopy(ws["episode_count"])
        if "episode_start_times" in ws:
            w.episode_start_times = _copy_np(ws["episode_start_times"])
        if "episode_returns" in ws:
            w.episode_returns = _copy_np(ws["episode_returns"])
        if "episode_lengths" in ws:
            w.episode_lengths = _copy_np(ws["episode_lengths"])
        if "return_queue" in ws:
            w.return_queue = _copy_deque(ws["return_queue"])
        if "length_queue" in ws:
            w.length_queue = _copy_deque(ws["length_queue"])
        if "time_queue" in ws:
            w.time_queue = _copy_deque(ws["time_queue"])


def _snapshot_mujoco_state(env):
    """
    Save MuJoCo integration state from env.unwrapped.
    """
    base = env.unwrapped
    if not (hasattr(base, "model") and hasattr(base, "data")):
        raise TypeError("env.unwrapped does not look like a MuJoCo env (missing model/data).")

    model = base.model
    data = base.data
    sig = mujoco.mjtState.mjSTATE_INTEGRATION
    n = mujoco.mj_stateSize(model, sig)
    state = np.empty(n, dtype=np.float64)
    mujoco.mj_getState(model, data, state, sig)

    debug = {
        "time": float(data.time),
        "qpos": _copy_np(data.qpos),
        "qvel": _copy_np(data.qvel),
        "act": _copy_np(getattr(data, "act", None)),
        "ctrl": _copy_np(getattr(data, "ctrl", None)),
    }
    return base, sig, state, debug


def _restore_mujoco_state(base, sig, state):
    mujoco.mj_setState(base.model, base.data, state, sig)
    mujoco.mj_forward(base.model, base.data)


def _print_internal_state(tag, env, base):
    print(f"\n=== {tag} ===")
    print(f"mujoco time: {base.data.time:.8f}")
    print("qpos[:5] =", np.array(base.data.qpos[:5], copy=True))
    print("qvel[:5] =", np.array(base.data.qvel[:5], copy=True))

    for w in _walk_wrappers(env):
        name = type(w).__name__

        if isinstance(w, AddTimeInfo):
            print(f"{name}.epi_time =", w.epi_time)

        if isinstance(w, NormalizeObservation):
            print(f"{name}.obs_stats.count =", w.obs_stats.count)
            print(f"{name}.obs_stats.mean[:3] =", np.array(w.obs_stats.mean[:3], copy=True))

        if isinstance(w, ScaleReward):
            print(f"{name}.reward_trace =", np.array(w.reward_trace, copy=True))
            print(f"{name}.reward_stats.count =", w.reward_stats.count)

        if hasattr(w, "_elapsed_steps"):
            print(f"{name}._elapsed_steps =", w._elapsed_steps)

        if hasattr(w, "episode_count"):
            print(f"{name}.episode_count =", w.episode_count)
        if hasattr(w, "episode_returns") and w.episode_returns is not None:
            print(f"{name}.episode_returns =", np.array(w.episode_returns, copy=True))
        if hasattr(w, "episode_lengths") and w.episode_lengths is not None:
            print(f"{name}.episode_lengths =", np.array(w.episode_lengths, copy=True))
        if hasattr(w, "episode_start_times") and w.episode_start_times is not None:
            print(f"{name}.episode_start_times =", np.array(w.episode_start_times, copy=True))


def step_without_changing_env(env, action, verbose=True):
    """
    Take a hypothetical env.step(action), return its outputs,
    and restore MuJoCo + wrapper state so the environment is unchanged afterward.

    Returns:
        next_obs, reward, terminated, truncated, info
    """
    wrapper_snap = _snapshot_wrapper_state(env)
    base, sig, mujoco_state, mujoco_debug_before = _snapshot_mujoco_state(env)

    if verbose:
        _print_internal_state("BEFORE hypothetical step", env, base)

    try:
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Deep-copy outputs so they remain valid after restore.
        next_obs_out = _copy_np(next_obs) if isinstance(next_obs, np.ndarray) else copy.deepcopy(next_obs)
        reward_out = copy.deepcopy(reward)
        terminated_out = copy.deepcopy(terminated)
        truncated_out = copy.deepcopy(truncated)
        info_out = copy.deepcopy(info)

        if verbose:
            _print_internal_state("AFTER hypothetical step", env, base)

    finally:
        _restore_wrapper_state(wrapper_snap)
        _restore_mujoco_state(base, sig, mujoco_state)

    if verbose:
        _print_internal_state("AFTER restore", env, base)

        print("\nSanity checks:")
        print("time restored:", np.isclose(base.data.time, mujoco_debug_before["time"]))
        print("qpos restored:", np.allclose(base.data.qpos, mujoco_debug_before["qpos"]))
        print("qvel restored:", np.allclose(base.data.qvel, mujoco_debug_before["qvel"]))
        if mujoco_debug_before["act"] is not None:
            print("act restored:", np.allclose(base.data.act, mujoco_debug_before["act"]))
        if mujoco_debug_before["ctrl"] is not None:
            print("ctrl restored:", np.allclose(base.data.ctrl, mujoco_debug_before["ctrl"]))

    return next_obs_out, reward_out, terminated_out, truncated_out, info_out


def verify_step_without_changing_env(env, action, atol=1e-10, rtol=1e-8, verbose=True):
    """
    Strong verification:
    1) do hypothetical step,
    2) do real step with the same action,
    3) compare outputs.

    If restoration is correct, the two outputs should match.
    """
    hyp = step_without_changing_env(env, action, verbose=verbose)
    real = env.step(action)

    hyp_obs, hyp_rew, hyp_term, hyp_trunc, hyp_info = hyp
    real_obs, real_rew, real_term, real_trunc, real_info = real

    ok = True

    if isinstance(hyp_obs, np.ndarray) and isinstance(real_obs, np.ndarray):
        same_obs = np.allclose(hyp_obs, real_obs, atol=atol, rtol=rtol)
    else:
        same_obs = hyp_obs == real_obs
    ok &= bool(same_obs)

    same_rew = np.allclose(np.asarray(hyp_rew), np.asarray(real_rew), atol=atol, rtol=rtol)
    ok &= bool(same_rew)

    same_term = hyp_term == real_term
    ok &= bool(same_term)

    same_trunc = hyp_trunc == real_trunc
    ok &= bool(same_trunc)

    print("\nVerification against real env.step(action):")
    print("obs match:", same_obs)
    print("reward match:", same_rew)
    print("terminated match:", same_term)
    print("truncated match:", same_trunc)

    # Optional useful comparisons if your wrappers provide these.
    hyp_raw = hyp_info.get("obs_original", None) if isinstance(hyp_info, dict) else None
    real_raw = real_info.get("obs_original", None) if isinstance(real_info, dict) else None
    if hyp_raw is not None and real_raw is not None:
        same_raw = np.allclose(np.asarray(hyp_raw), np.asarray(real_raw), atol=atol, rtol=rtol)
        print("raw obs match:", same_raw)
        ok &= bool(same_raw)

    hyp_immediate_rew = hyp_info.get("reward_immediate", None) if isinstance(hyp_info, dict) else None
    real_immediate_rew = real_info.get("reward_immediate", None) if isinstance(real_info, dict) else None
    if hyp_immediate_rew is not None and real_immediate_rew is not None:
        same_immediate_rew = np.allclose(
            np.asarray(hyp_immediate_rew), np.asarray(real_immediate_rew), atol=atol, rtol=rtol
        )
        print("immediate reward match:", same_immediate_rew)
        ok &= bool(same_immediate_rew)

    print("overall verification:", ok)
    return ok, hyp, real