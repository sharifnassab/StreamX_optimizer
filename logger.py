# logger.py
import os
from typing import Dict, Optional

class _NullLogger:
    def __init__(self, *args, **kwargs): pass
    def log(self, metrics: Dict, step: Optional[int] = None): pass
    def finish(self): pass
    def watch(self, *args, **kwargs): pass

class _TBLogger:
    def __init__(self, log_dir: str = "runs", run_name: Optional[str] = None, config: Optional[Dict] = None, **_):
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name or "run"))
        if config:
            try:
                # Write config as text; TB doesn't have a native config pane like wandb
                self.writer.add_text("config", "\n".join(f"{k}: {v}" for k,v in config.items()))
            except Exception:
                pass

    def log(self, metrics: Dict, step: Optional[int] = None):
        for k, v in metrics.items():
            if v is None:  # skip missing metrics
                continue
            try:
                self.writer.add_scalar(k, float(v), global_step=step if step is not None else 0)
            except Exception:
                # fall back for non-scalars
                pass

    def finish(self):
        self.writer.flush()
        self.writer.close()

    def watch(self, *args, **kwargs):
        # No-op for TensorBoard to keep API parity with WandB
        pass

class _WandbLogger:
    def __init__(self, project: str = "rl", run_name: Optional[str] = None, config: Optional[Dict] = None, **_):
        import wandb
        self.wandb = wandb
        self.wandb.init(project=project, name=run_name, config=config or {})
    def log(self, metrics: Dict, step: Optional[int] = None):
        self.wandb.log(metrics, step=step)
    def finish(self):
        self.wandb.finish()
    def watch(self, models, **kwargs):
        try:
            self.wandb.watch(models, **kwargs)
        except Exception:
            pass

def get_logger(backend: str = "tensorboard", **kwargs):
    backend = (backend or "tensorboard").lower()
    if backend == "tensorboard":
        return _TBLogger(**kwargs)
    elif backend == "wandb":
        return _WandbLogger(**kwargs)
    elif backend in ("none", "null", "off"):
        return _NullLogger()
    else:
        raise ValueError(f"Unknown logger backend: {backend}")
