from Datasets.create_features import (
    generate_esm_embedding,
    convert_protein_sequence_to_number,
)
import lmdb
from multiprocessing.pool import Pool
from Datasets.create_features import get_reaction_feature_single
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import os
import argparse

from Bio.PDB import *

import glob
from tqdm import tqdm
from ray.util.multiprocessing import Pool
import ray

from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, rdDepictor
from rdkit import Chem
from rdkit import RDLogger
import ray

from easydict import EasyDict as edict
from Datasets.Structure.structure import str_process, seq_process

from easydict import EasyDict as edict
from main_testing import generate_prediction
from utils import load_config

from pathlib import Path
from datetime import datetime
import os
import subprocess
import sys
import pandas as pd

############################################################
### 0 Generate enzyme features
############################################################


## Running the og model
#data_root_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/"
#src_root_dir = "/vast/projects/pranam/lab/emahood/repos/ezspec/"
#model_config_path = f"{data_root_dir}/saved_model/model/run_0/complete-full-random-all-0-complex.yml"
#logging_dir = f"{data_root_dir}/saved_model/model"
#ckpt_path = f"{logging_dir}/run_0/models/best-checkpoint.ckpt"

## Running the FT model
#src_root_dir = "/vast/projects/pranam/lab/emahood/repos/ezspec/"
#logging_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/cyp_benchmark/data_inputs/train/processed_data/logs"
#ckpt_path = f"{logging_dir}/checkpoints/ft_ezspec_best_epoch_6.ckpt"

## The preprocessing steps have been extracted to preprocess.py

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root-dir", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, required=True)
    parser.add_argument("--features-output-dir", type=str, required=True)
    parser.add_argument("--prepared-data-dir", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)

    return parser.parse_args()

def main():

    ############################################################
    ### Generate prediction!!!
    ############################################################

    args = argparser()
    src_root_dir = args.src_root_dir
    logging_dir = args.logging_dir
    features_output_dir = args.features_output_dir
    prepared_data_dir = args.prepared_data_dir
    ckpt_path = args.ckpt_path

    ## Constants
    metadata_csv_path = f"{prepared_data_dir}/data.csv"
    structure_dir = f"{prepared_data_dir}/Structure"

    model_config_path = f"{src_root_dir}/configs/train_example.yml"

    test_csv_path = metadata_csv_path

    dict = {
        "data": {
            "tag": "example",
            "representer": "structure_sequence",
            "log_dir": [
                logging_dir,
            ],
            "train_data_path": test_csv_path,
            "val_data_path": test_csv_path,
            "test_data_path": test_csv_path,
            "enzyme_lmdb_path": f"{features_output_dir}/enzyme_features.lmdb",
            "reaction_lmdb_path": f"{features_output_dir}/reaction_features.lmdb",
            "grover_path": f"{features_output_dir}/grover_fingerprint.lmdb",
            "morgan_path": f"{features_output_dir}/morgan_fingerprint.npy",
            "structure_processed_path": f"{features_output_dir}/structure_features.lmdb",
            "sequence_processed_path": f"{features_output_dir}/str_features.lmdb",
            "pdb_dir": f"{structure_dir}/str_tmp_data/pocket",
            "ligand_dir": f"{structure_dir}/str_tmp_data/ligand",
            "high_quality_id_path": f"{features_output_dir}/high_quality_id.txt",
            "full_data": True,
            "fake_sequence_ratio": 0,
            "sample_weight": [1.0, 1.0],
            "batch_size": 8,
            "max_substrate_length": 280,
            "max_enzyme_length": 1450,
            "features": ["morgan", 1024, "grover_mean", 4885],
            "atom_features": ["grover", 2400],
            "skip_timeout": 30,
            "skip_log_file": f"{features_output_dir}/skip_log.txt",
            "ckpt_path": ckpt_path,
        }
    }
    config = load_config(
        model_config_path
    )
    config.data = edict(dict).data

    score = generate_prediction(config)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = f"{prepared_data_dir}/example_out_{timestamp}.csv"
    score.to_csv(out_path, sep=",")
    print(f"Saved prediction to {out_path}")


if __name__ == "__main__":
    main()
