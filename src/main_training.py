"""
Training script for EZSpecificity (SS model).

Usage:
  python src/main_training.py --config path/to/train_config.yml

Hyperparameters from the paper:
1 .Binary classification with cross-entropy -- BCEWithLogitsLoss
2. AdamW optimizer, lr=0.0003
3. LR scheduler: linear from 6e-6 to 0.0003 for initial epochs
4. BUT reduce LR by half if no improvement for 10 epochs -- ReduceLROnPlateau
5. Stop when lr < 6e-6 -- MinLRStoppingCallback
6. Batch size 32
7. SE(3) GNN 3 layers, hidden_dim 128 -- SS model
8. MLP predictor: 3 feedforward layers, hidden_dim 128

Hyperparameters from the paper (overridable via --config):
  - AdamW, lr=0.0003, warmup, ReduceLROnPlateau, batch_size=32
  - 3 GNN layers, 3 MLP layers, hidden_dim 128
"""

import argparse
import os
import sys
import yaml
from time import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from easydict import EasyDict

import torch
import torch.multiprocessing
from rdkit import RDLogger

from Datasets.brenda import Singledataset
from Models.ss import SS

RDLogger.DisableLog("rdApp.*")
torch.multiprocessing.set_sharing_strategy("file_descriptor")

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


def _to_edict(obj):
    """Recursively convert dict to EasyDict."""
    if isinstance(obj, dict):
        return EasyDict({k: _to_edict(v) for k, v in obj.items()})
    return obj


def load_config(cfg_path):
    """Load config from YAML. Returns EasyDict. No defaults or transformations applied."""
    with open(cfg_path, "r") as f:
        return _to_edict(yaml.safe_load(f))


_PATH_KEYS = frozenset(
    {
        "train_data_path",
        "val_data_path",
        "test_data_path",
        "enzyme_lmdb_path",
        "reaction_lmdb_path",
        "grover_path",
        "morgan_path",
        "structure_processed_path",
        "sequence_processed_path",
        "high_quality_id_path",
        "pdb_dir",
        "ligand_dir",
        "log_dir",
        "skip_log_file",
        "ckpt_path",
        "root_dir",
        "output_dir",
        "resume",
    }
)


def _resolve_one(base: str, s: str) -> str:
    if not s or s == "null":
        return s
    if os.path.isabs(s):
        return s
    return os.path.normpath(os.path.join(base, s))


def resolve_config_paths(config, repo_root: str) -> None:
    """
    If config has ``data_root`` (relative to repo_root), join it with all known path fields.
    Absolute path strings in the YAML are left unchanged. Mutates config in place.
    """
    data_root = getattr(config, "data_root", None)
    if data_root is None:
        return
    base = os.path.normpath(
        os.path.join(os.path.abspath(repo_root), str(data_root).strip())
    )

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if k in _PATH_KEYS:
                    if isinstance(v, str):
                        obj[k] = _resolve_one(base, v)
                    elif isinstance(v, list):
                        obj[k] = [
                            _resolve_one(base, x) if isinstance(x, str) else x for x in v
                        ]
                elif isinstance(v, dict):
                    walk(v)
        elif hasattr(obj, "keys") and callable(getattr(obj, "keys")):
            # EasyDict
            for k in list(obj.keys()):
                v = obj[k]
                if k in _PATH_KEYS:
                    if isinstance(v, str):
                        obj[k] = _resolve_one(base, v)
                    elif isinstance(v, list):
                        obj[k] = [
                            _resolve_one(base, x) if isinstance(x, str) else x for x in v
                        ]
                elif isinstance(v, dict) or (
                    hasattr(v, "keys") and callable(getattr(v, "keys"))
                ):
                    walk(v)

    walk(config)


class MinLRStoppingCallback(pl.Callback):
    """Stop training when learning rate drops below min_lr (paper: 0.000006)."""

    def __init__(self, min_lr=6e-6):
        self.min_lr = min_lr

    def on_train_epoch_end(self, trainer, pl_module):
        lrs = [pg["lr"] for pg in trainer.optimizers[0].param_groups]
        if lrs and min(lrs) < self.min_lr:
            trainer.should_stop = True
            print(f"\n[INFO] Stopping: LR {min(lrs):.2e} < min_lr {self.min_lr:.2e}", flush=True)


class EpochLogCallback(pl.Callback):
    """Write epoch progress to a log file in real time (for SLURM/job monitoring)."""

    def __init__(self, log_path):
        self.log_path = log_path

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        msg = f"Epoch {epoch}/{max_epochs} completed, lr={lr:.2e}\n"
        with open(self.log_path, "a") as f:
            f.write(msg)
            f.flush()


def main():
    parser = argparse.ArgumentParser(description="Train EZSpecificity (SS model)")
    parser.add_argument("--config", required=True, help="Path to config YAML (all arguments come from config)")
    parser.add_argument(
        "--repo_root",
        type=str,
        default=None,
        help="Directory that config ``data_root`` and relative paths are resolved against (default: current working directory)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Override: resume from checkpoint path")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    repo_root = args.repo_root if args.repo_root is not None else os.getcwd()

    config = load_config(args.config)
    resolve_config_paths(config, repo_root)
    if args.resume:
        config.resume = args.resume

    # Output dir: config.output_dir or config.root_dir (both valid in config format)
    output_dir = config.log_dir if hasattr(config, "log_dir") else config.root_dir
    num_gpus = config.num_gpus if hasattr(config, "num_gpus") else 1

    dm = Singledataset(config)
    model = SS(config)

    print(f"Output directory: {output_dir}", flush=True)

    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "training.log")
    with open(log_path, "w") as f:
        f.write(f"Training started at {time()}\n")
        f.flush()

    callbacks = [
        EpochLogCallback(log_path),
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="best-{epoch:03d}",
            monitor="aupr/val",
            mode="max",
            save_top_k=1,
        ),
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="last-{epoch:03d}",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="aupr/val",
            patience=config.training.sched_patience,
            mode="max",
        ),
        MinLRStoppingCallback(min_lr=config.training.min_lr),
    ]

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if num_gpus > 0 else "cpu",
        devices=num_gpus if num_gpus > 0 else 1,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=callbacks,
        default_root_dir=output_dir
    )

    ckpt_path = getattr(config, "resume", None)

    print(f"Training started at {time()}", flush=True)
    if ckpt_path and os.path.isfile(ckpt_path):
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, datamodule=dm)
    print(f"Training completed at {time()}", flush=True)
    print(f"\nTraining complete. Best checkpoint saved to {output_dir}/checkpoints/", flush=True)
    with open(log_path, "a") as f:
        f.write(f"Training completed at {time()}\n")
        f.flush()

if __name__ == "__main__":
    main()
