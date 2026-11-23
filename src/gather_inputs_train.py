## Convert train/val df to their format
## Include these cols: 
##  pdb_file_name,
##  enzyme_index,
##  substrate_index,
##  label
##  uniprot ID
##  dock_index (this is the index of the pdb file in the Structure/complex directory)

## Need to get all enzymes and substrates across training/val 
## Including negative substrates (though they should be included in positives)
## Inputs: train/val df, negative_cache, pdb dirs for train/val pos/neg uniprot2seq

## Collect all enzyme UniprotIDs and sequences across train/val pos/neg --> output
## Same for substrate SMILES --> output
## For each pdb file in train/val:  ()
##      Probably the way to do this is get the pdb files for each enzyme
##      Loop through these, if "correct" is NOT in the name then the index if 
##      the index of the negative substrate in the negative cache

##      If it is correct, substrate is in the train/val df

## build the substrate and enzyme files first, then do pdbs 

import pandas as pd
from pathlib import Path
import argparse
from typing import Dict
import pickle
import json
from tqdm import tqdm

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_df", type=Path, required=True)
    parser.add_argument("--val_df", type=Path, required=True)
    parser.add_argument("--negative_cache", type=Path, required=True)
    parser.add_argument("--pdb_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--u2s", type=Path, required=True)
    return parser.parse_args()

def process_one_pdb(
    pdb: Path,
    parent_folder: Path,
    ftype: str,
    u2s: Dict[str, str],
    negative_cache: Dict = None,
    df: pd.DataFrame = None,
):
    """
    Hacky and redundant but I am of limited time

    If positive, get sequence and substrate from df (train/val df)
    If negative, get these from negative cache

    Args:
        pdb: Path to the pdb file
        parent_folder: Path to the parent folder of the pdb file (e.g. train_pos)
        type: "pos" or "neg"
        u2s: Dictionary mapping uniprot IDs to sequences
        negative_cache: Dictionary mapping enzymes to negative substrates
        df: DataFrame containing the train/val df

    Returns:
        sequence: Sequence of the enzyme
        substrate: Substrate of the enzyme
        label: Label of the enzyme
        uniprot: Uniprot ID of the enzyme
    """

    if ftype == "pos":
        pdb_id = pdb.stem
        sample_id = "_".join(pdb_id.split("_")[0:3])
        uniprot = sample_id.split("_")[0]
        sequence = u2s[uniprot]
        substrate = df[df["sample_id"] == sample_id]["substrate"].values[0]
        label = 1

    elif ftype == "neg":
        uniprot = pdb.stem.split("_")[0]
        all_neg_subs = negative_cache[uniprot]
        all_neg_pdbs = sorted(Path(parent_folder).glob(f"{uniprot}*.pdb"))
        neg_index = all_neg_pdbs.index(pdb)
        substrate = all_neg_subs[neg_index]
        label = 0
        sequence = u2s[uniprot]

    return sequence, substrate, label, uniprot

def map_pdb_to_sequence_substrate(
    cifs_dir: Path, 
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    negative_cache: Dict, 
    u2s: Dict[str, str]
):
    """
    Collect all pdb files, get name and label of each pdb file
    For each enzyme, get all pdb files for it, sort by name
    Enzyme comes from the name of the pdb
    Substrate comes from the df (positive) and negative cache
    Negative cache index for that enzyme is the index of the pdb file in its list
    
    Collect unique substrates and sequences, assign an index to each
    Find the sequence and substrate for each remaining pdb file
    Map the pdb file to the sequence and substrate index
    Save the mapping to a csv file

    Args:
        cifs_dir: Path to the directory containing the pdb files
        holdout_df: holdout dataframe
        rxn_files: Path to the directory containing the rxn files

    Returns:
        pdb_to_sequence_substrate_df: DataFrame mapping pdb file to sequence and substrate index
        unique_substrates: List of unique substrates
        unique_sequences: List of unique sequences
    """

    pdb_dirs = ["train_pos", "train_neg", "val_pos", "val_neg"]

    unique_substrates = set()
    unique_sequences = set()
    unique_uniprots = set()
    pdb_to_sequence_substrate = {}

    for folder in pdb_dirs:
        pdbs = Path(cifs_dir / folder).glob("*.pdb")

        if "train" in folder:
            df = train_df
        else:
            df = val_df
        ft = "pos" if "pos" in folder else "neg" 

        for pdb in pdbs:
            seq, sub, lab, uni = process_one_pdb(pdb, Path(cifs_dir) / folder, ft, u2s, negative_cache, df)
            unique_substrates.add(sub)
            unique_sequences.add(seq)
            unique_uniprots.add(uni)
            pdb_to_sequence_substrate[pdb.stem] = (seq, sub, lab, uni, folder)

    print(f"Found {len(unique_substrates)} unique substrates")
    print(f"Found {len(unique_sequences)} unique sequences")

    substrate_to_index = {
        substrate: index for index, substrate in enumerate(unique_substrates)
    }
    sequence_to_index = {
        sequence: index for index, sequence in enumerate(unique_sequences)
    }

    for pdb_id, (
        sequence,
        substrate,
        label,
        uniprot,
        folder
    ) in pdb_to_sequence_substrate.items():
        substrate_index = substrate_to_index[substrate]
        sequence_index = sequence_to_index[sequence]
        pdb_to_sequence_substrate[pdb_id] = (
            sequence_index,
            substrate_index,
            label,
            uniprot,
            substrate,
            folder
        )

    pdb_to_sequence_substrate_list = [
        {
            "pdb_id": pdb_id,
            "Enzyme Index": seq_idx,
            "Substrate Index": sub_idx,
            "Label": label,
            "uniprot": uniprot,
            "substrate": substrate,
            "folder": folder
        }
        for pdb_id, (
            seq_idx,
            sub_idx,
            label,
            uniprot,
            substrate,
            folder
        ) in pdb_to_sequence_substrate.items()
    ]
    pdb_to_sequence_substrate_df = pd.DataFrame(pdb_to_sequence_substrate_list)

    pdb_to_sequence_substrate_df["Dock Index"] = pd.Series(
        range(len(pdb_to_sequence_substrate_df))
    )

    return (
        pdb_to_sequence_substrate_df,
        unique_substrates,
        unique_sequences,
        unique_uniprots,
    )


def convert_pdb(
    pdb: Path, df: pd.DataFrame, output_dir: Path
):
    """
    Convert pdb file into acceptable format for pocket extraction
    Save the converted pdb file to a new directory

    Args:
        pdb: Path to the pdb file
        df: DataFrame containing the Dock Index, substrate and pdb_id
        output_dir: Path to the directory to save the converted pdb file

    Returns:
        None
    """

    pdb_id = pdb.stem
    pdb_index = df[df["pdb_id"] == pdb_id]["Dock Index"].values[0]
    substrate = df[df["pdb_id"] == pdb_id]["substrate"].values[0]
    output_path = output_dir / f"{pdb_index}.pdb"

    protein_lines = []
    ligand_lines = []
    ligand_atoms = []
    with open(pdb, "r") as f:
        lines = f.readlines()

    ## Remove first 2 lines
    lines = lines[2:]

    ## Collect all lines until "HEM" is found
    ## Change "ATOM" to "HETATM" in all lines with "HEM"
    lig_found = False
    ligand_break = len(lines)  # Default to end of file if no ligand found
    for i, line in enumerate(lines):
        if "HEM B" in line:
            line = line.replace("ATOM", "HETATM")
            protein_lines.append(line)

        elif "LIG C" in line and not lig_found:
            lig_found = True
            ligand_break = i
            ligand_lines.append(line)
            ligand_atoms.append(line[7:12].strip())

        elif "LIG C" in line and lig_found:
            ligand_lines.append(line)
            ligand_atoms.append(line[7:12].strip())

        elif "ATOM" in line:
            protein_lines.append(line)

    ## Final loop through lines to collect CONECT lines for ligand atoms
    ligand_atoms = list(set(ligand_atoms))
    for i, line in enumerate(lines[ligand_break:]):
        if "CONECT" in line:
            atoms = [atom.strip() for atom in line.split(" ")]
            if any(atom in ligand_atoms for atom in atoms):
                ligand_lines.append(line)
        elif "MASTER" in line:
            ligand_lines.append(line)

    ## Write to a new pdb file
    with open(output_path, "w") as f:
        for line in protein_lines:
            f.write(line)

        f.write("END\n")
        f.write(f"REMARK SMILES {substrate}\n")
        f.write(f"COMPND {pdb_index}\n")
        f.write("AUTHOR GENERATED BY OPEN BABEL 3.1.0\n")
        for line in ligand_lines:
            f.write(line)
        f.write("END\n")


def main():
    args = argparser()
    pdbs_dir = args.pdb_dir
    train_df = args.train_df
    val_df = args.val_df
    negative_cache = args.negative_cache
    u2s = args.u2s
    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_output_dir = output_dir / "Structure" / "complex"
    pdb_output_dir.mkdir(parents=True, exist_ok=True)

    with open(u2s, "rb") as f:
        u2s = pickle.load(f)
        
    with open(negative_cache, "r") as f:
        negative_cache = json.load(f)

    train_df = pd.read_csv(train_df)
    val_df = pd.read_csv(val_df)

    print("Mapping pdb files to sequence and substrate")
    (
        pdb_to_sequence_substrate_df,
        unique_substrates,
        unique_sequences,
        unique_uniprots,
    ) = map_pdb_to_sequence_substrate(pdbs_dir, train_df, val_df, negative_cache, u2s)
    
    print("Converting pdb files to acceptable format")
    
    for _, row in tqdm(
        pdb_to_sequence_substrate_df.iterrows(),
        desc="Converting pdb files to acceptable format",
        total=len(pdb_to_sequence_substrate_df),
    ):
        pdb_id = row["pdb_id"]
        folder = row["folder"]
        convert_pdb(
            pdbs_dir / folder / f"{pdb_id}.pdb",
            pdb_to_sequence_substrate_df,
            pdb_output_dir,
        )

    print("Saving data.csv to output directory", output_dir)
    pdb_to_sequence_substrate_df.to_csv(output_dir / "data.csv", index=False)

    ## Need to split data into train and val
    train_df = pdb_to_sequence_substrate_df[pdb_to_sequence_substrate_df["folder"].str.startswith("train")]
    val_df = pdb_to_sequence_substrate_df[pdb_to_sequence_substrate_df["folder"].str.startswith("val")]
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)

    substrate_df = pd.DataFrame(list(unique_substrates), columns=["Substrate_SMILES"])
    substrate_df.to_csv(output_dir / "Substrates.csv", index=False)
    sequence_df = pd.DataFrame(list(unique_sequences), columns=["Protein sequence"])
    # sequence_df["uniprot"] = unique_uniprots
    sequence_df.to_csv(output_dir / "Enzymes.csv", index=False)


if __name__ == "__main__":
    main()
