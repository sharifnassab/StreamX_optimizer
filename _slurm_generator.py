# slurm_generator.py

import itertools, shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from textwrap import dedent

SLURM_DIR = Path("__slurm")  # Global location for all local outputs

def fresh_slurm_dir() -> None:
    if SLURM_DIR.exists():
        shutil.rmtree(SLURM_DIR)
    SLURM_DIR.mkdir()


def cart_prod(d: dict[str, list]) -> list[dict]:
    keys, vals = zip(*d.items())
    return [dict(zip(keys, v)) for v in itertools.product(*vals)]


def derive_resources(run: dict, resource_defaults: dict, resource_overrides: dict) -> dict:
    # Build a consistent, sorted tuple key from run (excluding reserved keys)
    reserved = {"max_time", "cpus", "gpus", "account", "mem", "constraint"}
    run_key = tuple(sorted((k, v) for k, v in run.items() if k not in reserved))

    # Start with defaults, apply override if any
    res = resource_defaults.copy()
    res.update(resource_overrides.get(run_key, {}))

    # 1. max_time from sweep overrides everything
    mt = run.get("max_time")
    if mt and mt != "default":
        res["max_time"] = mt
    elif "max_time" not in res:
        res["max_time"] = resource_defaults.get("max_time")

    run["max_time"] = res["max_time"]  # keep in sync

    # Slurm expects this as --time=
    res["time"] = res["max_time"]

    # Normalize memory
    if isinstance(res.get("mem"), str) and res["mem"].lower().endswith("g"):
        num = int(float(res["mem"][:-1]))
        res["mem"] = f"{num * 1024}M"

    # Ensure Slurm-compatible strings
    res["cpus"] = str(res.get("cpus", 1))
    res["gpus"] = str(res.get("gpus", 0))

    return res


def export_line(run: dict, fixed: dict, uid: str) -> str:
    all_vars = {**fixed, **run, "uID": uid}
    return "export " + " ".join(f"{k}={v}" for k, v in all_vars.items())


def arg_string(first_export: str) -> str:
    toks = first_export.replace("export ", "").replace("=", " ").split()
    return " ".join(f"--{toks[i]}=${toks[i]}" for i in range(0, len(toks), 2))


def slurm_header(arr_sz: int, res: dict, remote_dir: Path, sweep_id: int, group_id: int) -> str:
    out_path = remote_dir / f"{sweep_id}_{group_id}" / "output_%j.txt"
    hdr = dedent(f"""\
        #!/bin/bash
        #SBATCH --account={res['account']}
        #SBATCH --time={res['time']}
        #SBATCH --cpus-per-task={res['cpus']}
        #SBATCH --mem={res['mem']}
        #SBATCH --array=1-{arr_sz}
        #SBATCH --output={out_path}
        #SBATCH --constraint={res.get('constraint','')}
    """)
    if not res["gpus"] == '0':
        hdr += f"#SBATCH --gres=gpu:{res['gpus']}\n"
    return hdr



def slurm_body(py_args: str, export_path: Path, venv_activate: str, py_entry: str, res: dict) -> str:
    try: 
        num_gpus = int(res["gpus"].split('.')[-1])
    except:
        num_gpus = 0
    cuda = "module load cuda\n" if num_gpus>0 else ""
    return dedent(f"""\
        
        module purge
        module load StdEnv/2023
        module load python/3.10
        module load mujoco/3.1.6

        source {venv_activate}
        export PYTHONNOUSERSITE=1

        {cuda}
        $(sed -n "${{SLURM_ARRAY_TASK_ID}}p" < {export_path})
        echo "Task ${{SLURM_ARRAY_TASK_ID}} started on $(hostname) at $(date)"
        python3 {py_entry} {py_args}
        echo "Program test finished with exit code $? at: `date`"
    """)


def generate_slurm(
    common_env: dict,
    hyper_sweeps: list[dict],
    resource_defaults: dict,
    resource_overrides: dict,
    remote_results_root: str,
    python_entrypoint: str,
    venv_activate: str,
    uid_description: str = '',
) -> str:
    """
    Creates export_*.dat and _submitter_*.sh files in ./slurm/
    Returns: uid (timestamp string)
    """
    fresh_slurm_dir()
    uid = datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + ('__' + uid_description if not uid_description=='' else '')
    remote_save = Path(remote_results_root) / uid

    for sweep_id, sweep in enumerate(hyper_sweeps):
        runs = cart_prod(sweep)
        export_lines, resources = [], []

        for r in runs:
            res = derive_resources(r, resource_defaults, resource_overrides)
            export_lines.append(export_line(r, common_env, uid))
            resources.append(res)

        # Group by resources
        groups: defaultdict[tuple, list[int]] = defaultdict(list)
        for i, res in enumerate(resources, 1):
            groups[tuple(res.items())].append(i)

        for group_id, (res_items, line_idxs) in enumerate(groups.items()):
            res = dict(res_items)
            lines = [export_lines[i - 1] for i in line_idxs]

            export_name = f"export_{sweep_id}_{group_id}.dat"
            export_local = SLURM_DIR / export_name
            export_remote = remote_save / export_name
            export_local.write_text("\n".join(lines) + "\n")

            py_args = arg_string(lines[0])
            submitter = SLURM_DIR / f"_submitter_{sweep_id}_{group_id}.sh"
            header = slurm_header(len(line_idxs), res, remote_save, sweep_id, group_id)
            body = slurm_body(py_args, export_remote, venv_activate, python_entrypoint, res)
            submitter.write_text(header + body)
            submitter.chmod(0o755)

            print(f"• {submitter.name:<22} jobs:{len(line_idxs):>3}  time:{res['time']}")

    # Write UID for run_remote.sh
    (SLURM_DIR / "last_uid.txt").write_text(uid + "\n")
    #print(f"\nAll files written to {SLURM_DIR.resolve()}")
    #print(f"Remote results dir: {remote_save}\n")
    return uid
