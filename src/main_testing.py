import sys

root_dir = "/projects/bbto/suyufeng/enzyme_specificity"
sys.path.append(f"{root_dir}/src")

import pytorch_lightning as pl
from Datasets.brenda import Singledataset
import torch, sys, glob
import torch.multiprocessing
from rdkit import RDLogger
import warnings
import pandas as pd
import numpy as np
from Models.ss import SS

RDLogger.DisableLog("rdApp.*")
torch.multiprocessing.set_sharing_strategy("file_descriptor")
warnings.filterwarnings("ignore")


def generate_prediction(config):
    log_dirs = config.data.log_dir

    full_logits = None
    full_labels = None
    full_logits_dict = {}
    full_labels_dict = {}

    num_runs = 0
    if config.data.ckpt_path is not None:
        resume_checkpoint_path = config.data.ckpt_path

        config.num_cpus = 16
        dm = Singledataset(config)
        model = SS.load_from_checkpoint(resume_checkpoint_path, config=config)

        trainer_paras = {
            "accelerator": "gpu",
            "devices": 1,
            "num_nodes": 1,
            "gradient_clip_val": config.training.gradient_clip_val,
            "reload_dataloaders_every_n_epochs": 0,
            "accumulate_grad_batches": config.training.accumulate_grad_batches,
            "strategy": "auto",
        }

        trainer = pl.Trainer(**trainer_paras)
        trainer.test(model, datamodule=dm)
        logits = model.logits
        label = model.labels
        logits_dict = model.logits_dict
        labels_dict = model.labels_dict

        if full_logits is None:
            full_logits = logits
            full_labels = label
        else:
            full_logits = full_logits + logits
            full_labels = label

        num_runs += 1

        for key in logits_dict:
            if key not in full_logits_dict:
                full_logits_dict[key] = np.array([_.item() for _ in logits_dict[key]])
                full_labels_dict[key] = np.array(labels_dict[key])
            else:
                full_logits_dict[key] += np.array([_.item() for _ in logits_dict[key]])
        test_prediction_df = dm.test_prediction_df

    for key in logits_dict:
        full_logits_dict[key] /= num_runs
        test_prediction_df["score"] = full_logits_dict[key]
        return test_prediction_df
