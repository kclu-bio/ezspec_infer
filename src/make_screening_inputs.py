## give it screening dir, training enzymes file and substrates file
## Needs to import these, remove any overlap
## Remove any enzyme that has 0 positive substrates

## Then, generate ID for remaining substrates, ID for enzy are Uniprot IDs
## Generate MSA for each enzyme; save in msa dir
## For remaining enz/subs pairs, make input fasta file a la boltz
## Make the data/enzyme/reactions csv files for EZSpecificity preprocessing

import argparse
import json
import os
import glob
import pandas as pd
import re
from typing import Optional, Set, Tuple
from tqdm import tqdm
from rdkit import Chem
import warnings
import requests
from tqdm import tqdm
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

PROCESSED_OUTPUT_FILES = ("data.csv", "Enzymes.csv", "Substrates.csv")


def has_processed_outputs(screening_set_dir: str) -> bool:
    """Return whether a screening-set directory has a complete set of outputs."""
    screening_set_path = Path(screening_set_dir)
    csv_outputs_exist = all(
        (screening_set_path / filename).is_file()
        for filename in PROCESSED_OUTPUT_FILES
    )
    msa_outputs_exist = any((screening_set_path / "msa").glob("*.a3m"))
    boltz_inputs_exist = any(
        (screening_set_path / "boltz_inputs").glob("*.yaml")
    )
    return csv_outputs_exist and msa_outputs_exist and boltz_inputs_exist


def remove_nonspecific_smiles(df: pd.DataFrame, sub_col: str):
    """
    Remove substrates with * and/or no Carbons

    Args:
        df: pd.DataFrame
        sub_col: str, Column name for substrates

    Returns:
        pd.DataFrame, DataFrame with non-specific substrates removed
    """
    df = df.drop(df[df[sub_col].isna()].index)
    df = df.drop(df[df[sub_col].str.contains("*", na=False, regex=False)].index)

    pattern = r"c|C[^roudlsa]"
    df = df[df[sub_col].apply(lambda x: bool(re.search(pattern, x)))]

    return df

def standardize_smiles(smi: str, unmap: bool = False):
    """
    Canonicalize SMILES

    Args:
        smi: str, SMILES string
        unmap: bool, Whether to remove atom mapping

    Returns:
        str, Canonicalized SMILES string
    """
    mol = Chem.MolFromSmiles(smi)
    if mol:
        if unmap:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True)
    return None

def make_msa(sequence: str, output_dir: str, enzyme_id, key: str) -> str:
    """
    A3M style
    You will need to make an NVIDIA_API_KEY with NVIDIA's website if
    you don't have one already.
    Site: https://build.nvidia.com/colabfold/msa-search?snippet_tab=Python
    and press "Get API Key"

    This will prompt for API when run from CLI

    Returns:
        str: Path to the written .a3m file
    """
    url = os.getenv(
        "URL", "https://health.api.nvidia.com/v1/biology/colabfold/msa-search/predict"
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    a3m_out = Path(output_dir) / f"{enzyme_id}.a3m"

    if a3m_out.is_file() and a3m_out.stat().st_size > 0:
        print(f"        Reusing existing MSA: {a3m_out}")
        return str(a3m_out.resolve())

    r = requests.post(
        json={
            "sequence": sequence,
            "output_alignment_formats": ["a3m"],
        },
        url=url,
        headers={"Authorization": f"Bearer {key}"},
    )

    if r.status_code != 200:
        raise ValueError(f"Failed to make MSA: {r.status_code} {r.text}")

    r = r.json()
    saved = False
    for db, formats in r["alignments"].items():
        for fmt, alignment_obj in formats.items():
            aln = alignment_obj["alignment"]
            fmt_name = alignment_obj["format"]

            if db == "colabfold" and fmt_name == "a3m":
                a3m_out.write_text(aln)
                print(f"        Saved to {a3m_out}")
                saved = True

    if not saved:
        raise ValueError(
            f"No colabfold a3m alignment in API response for enzyme_id={enzyme_id}"
        )

    return str(a3m_out.resolve())

def make_boltz_yaml(
    sequence: str,
    output_dir: str,
    enzyme_id,
    dock_index: int,
    ligand: str,
    msa_dir: str,
) -> None:
    """
    Inputs to v0.4.1 boltz
    """
    msa_path = (Path(msa_dir).resolve() / f"{enzyme_id}.a3m").as_posix()
    # Quote SMILES for valid YAML (leading [, ], #, :, etc. are special in YAML)
    smiles_yaml = json.dumps(ligand)
    outstr = (
        f"version: 1\n"
        f"sequences:\n"
        f"  - protein:\n"
        f"      id: A\n"
        f"      sequence: {sequence}\n"
        f"      msa: {msa_path}\n"
        f"  - ligand:\n"
        f"      id: B\n"
        f"      smiles: {smiles_yaml}\n"
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/{dock_index}.yaml", "w") as f:
        f.write(outstr)

def load_training_data(training_dir):
    """
    Load all CSV files from training directory and extract unique sequences and molecules.

    Args:
        training_dir (str): Path to training directory containing CSV files

    Returns:
        tuple: (set of unique sequences, set of unique molecules)
    """
    csv_files = glob.glob(os.path.join(training_dir, "*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in training directory: {training_dir}")

    all_sequences = set()
    all_molecules = set()

    for file_path in csv_files:
        print(f"Loading training file: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path)

            if "sequences" in df.columns:
                sequences = df["sequences"].dropna().unique()
                all_sequences.update(sequences)

            if "substrates" in df.columns:
                molecules = df["substrates"].dropna().unique()
                all_molecules.update(molecules)

        except Exception as e:
            print(f"Error loading {file_path}: {e}. Skipping...")
            continue

    print(f"Found {len(all_sequences)} unique training sequences")
    print(f"Found {len(all_molecules)} unique training molecules")

    return all_sequences, all_molecules

def drop_enzymes_without_positives(df: pd.DataFrame, activity_col: str) -> pd.DataFrame:
    """
    Remove enzymes that have no positive substrate rows.
    """

    enzymes_with_positives = df.groupby("SEQ")[activity_col].max()
    keep_enzymes = enzymes_with_positives[enzymes_with_positives > 0].index
    n_before = df["SEQ"].nunique()
    filtered = df[df["SEQ"].isin(keep_enzymes)].copy()
    n_after = filtered["SEQ"].nunique()

    if n_after < n_before:
        print(
            f"Dropped {n_before - n_after} enzymes with no positive substrates "
            f"({n_after} enzymes remaining)"
        )
    return filtered

def process_screening_file(
    file_path: str,
    training_sequences: Set,
    training_molecules: Set,
) -> Optional[Tuple[Set, Set, pd.DataFrame]]:
    """
    Process a single screening file by filtering and standardizing molecules,
    removing training data overlaps, and generating embeddings for novel sequences/molecules.

    Returns:
        (novel_sequences, novel_molecules, filtered_df) or None if skipped
    """
    print(f"\nProcessing screening file: {os.path.basename(file_path)}")

    try:
        df = pd.read_csv(file_path)
        original_count = len(df)

        required = {"SEQ", "SUBSTRATES", "Activity"}
        missing = required - set(df.columns)
        if missing:
            print(
                f"Missing required columns {sorted(missing)} in {file_path}. Skipping..."
            )
            return None

        print("Filtering and standardizing molecules...")
        df = remove_nonspecific_smiles(df, "SUBSTRATES")

        df["standardized_smiles"] = df["SUBSTRATES"].apply(
            lambda x: standardize_smiles(x, unmap=True) if pd.notna(x) else None
        )
        df = df.dropna(subset=["standardized_smiles"])

        print(f"After filtering: {len(df)}/{original_count} rows remaining")

        unique_sequences = set(df["SEQ"].dropna().unique())
        unique_molecules = set(df["standardized_smiles"].dropna().unique())

        print(f"File contains {len(unique_sequences)} unique sequences")
        print(f"File contains {len(unique_molecules)} unique molecules")

        seq_intersection = unique_sequences.intersection(training_sequences)
        mol_intersection = unique_molecules.intersection(training_molecules)

        print(f"Found {len(seq_intersection)} sequences overlapping with training")
        print(f"Found {len(mol_intersection)} molecules overlapping with training")

        mask = ~(
            df["SEQ"].isin(training_sequences)
            | df["standardized_smiles"].isin(training_molecules)
        )
        filtered_df = df[mask].copy()

        print(f"After removing training overlaps: {len(filtered_df)}/{len(df)} rows remaining")

        filtered_df = drop_enzymes_without_positives(filtered_df, "Activity")

        if filtered_df.empty:
            print(f"No rows remaining after filtering for {file_path}. Skipping...")
            return None

        output_path = file_path.replace(".csv", "_filtered.csv")
        filtered_df.to_csv(output_path, index=False)
        print(f"Saved filtered file: {os.path.basename(output_path)}")

        novel_sequences = set(filtered_df["SEQ"].dropna().unique()) - training_sequences
        novel_molecules = (
            set(filtered_df["standardized_smiles"].dropna().unique()) - training_molecules
        )

        print(f"Found {len(novel_sequences)} novel sequences")
        print(f"Found {len(novel_molecules)} novel molecules")

    except Exception as e:
        print(f"Error processing {file_path}: {e}. Skipping...")
        return None

    return novel_sequences, novel_molecules, filtered_df

def process_screening_sets(
    screening_dir,
    training_sequences,
    training_molecules,
    key,
    force=False,
):
    """
    Process all screening sets in the given directory.

    Each subdir gets its own msa/, cif/, boltz_inputs/, and ezspec CSV outputs.
    """
    subdirs = [
        d for d in os.listdir(screening_dir)
        if os.path.isdir(os.path.join(screening_dir, d))
    ]

    if not subdirs:
        print(f"No subdirectories found in {screening_dir}")
        return

    print(f"Found {len(subdirs)} screening subdirectories")

    for subdir in subdirs:
        print(f"\nProcessing screening set: {subdir}")
        subdir_path = os.path.join(screening_dir, subdir)

        if not force and has_processed_outputs(subdir_path):
            print(
                f"  Processed outputs already exist in {subdir_path}. Skipping... "
                "Use --force to process it again."
            )
            continue

        subdir_msa_dir = os.path.join(subdir_path, "msa")
        boltz_input_dir = os.path.join(subdir_path, "boltz_inputs")

        os.makedirs(subdir_msa_dir, exist_ok=True)
        os.makedirs(boltz_input_dir, exist_ok=True)

        print(f"  msa dir:   {subdir_msa_dir}")

        binary_files = glob.glob(os.path.join(subdir_path, "*_binary.csv"))

        if not binary_files:
            print(f"No *_binary.csv files found in {subdir}. Skipping...")
            continue

        for binary_file in binary_files:
            result = process_screening_file(
                binary_file, training_sequences, training_molecules
            )
            if result is None:
                continue

            _, _, filt = result

            unique_seqs = sorted(filt["SEQ"].dropna().unique())
            unique_subs = sorted(filt["standardized_smiles"].dropna().unique())
            seqs = {seq: i for i, seq in enumerate(unique_seqs)}
            subs = {sub: i for i, sub in enumerate(unique_subs)}

            data_out = {
                "Substrate Index": [],
                "Enzyme Index": [],
                "Label": [],
                "Dock Index": [],
            }

            seen_enz = set()
            dock_index = 0
            for enz, mol, label in tqdm(zip(
                filt["SEQ"], filt["standardized_smiles"], filt["Activity"]
            ), desc="Processing screening set", total=len(filt)):
                enz_id = seqs[enz]
                sub_id = subs[mol]
                if enz not in seen_enz:
                    try:
                        make_msa(enz, subdir_msa_dir, enz_id, key)
                        seen_enz.add(enz)
                    except Exception as e:
                        print(f"Error making MSA for enzyme index {enz_id}: {e}. Skipping...")
                        continue

                make_boltz_yaml(
                    enz,
                    boltz_input_dir,
                    enz_id,
                    dock_index,
                    mol,
                    subdir_msa_dir,
                )

                data_out["Substrate Index"].append(sub_id)
                data_out["Enzyme Index"].append(enz_id)
                data_out["Label"].append(label)
                data_out["Dock Index"].append(dock_index)
                dock_index += 1

            pd.DataFrame(data_out).to_csv(
                os.path.join(subdir_path, "data.csv"), index=False
            )

            enzyme_df = pd.DataFrame(
                [seq for seq, _ in sorted(seqs.items(), key=lambda x: x[1])],
                columns=["Protein sequence"],
            )
            enzyme_df.to_csv(os.path.join(subdir_path, "Enzymes.csv"), index=False)

            sub_df = pd.DataFrame(
                [sub for sub, _ in sorted(subs.items(), key=lambda x: x[1])],
                columns=["Substrate_SMILES"],
            )
            sub_df.to_csv(os.path.join(subdir_path, "Substrates.csv"), index=False)

            print(f"  Saved ezspec inputs and boltz yamls under {subdir_path}")


def main():
    parser = argparse.ArgumentParser(description="Process FusionESP training and screening data")
    parser.add_argument(
        "-training-dir",
        type=str,
        required=True,
        help="Directory containing training CSV files",
    )
    parser.add_argument(
        "-screening-dir",
        type=str,
        required=True,
        help="Directory containing screening set subdirectories",
    )
    parser.add_argument(
        "-api-key",
        type=str,
        required=True,
        help="API key for NVIDIA API",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all screening sets even when complete outputs already exist",
    )

    args = parser.parse_args()

    if not os.path.exists(args.training_dir):
        raise ValueError(f"Training directory does not exist: {args.training_dir}")

    if not os.path.exists(args.screening_dir):
        raise ValueError(f"Screening directory does not exist: {args.screening_dir}")

    print("Starting FusionESP data processing...")
    print(f"Training directory: {args.training_dir}")
    print(f"Screening directory: {args.screening_dir}")

    print("\n" + "=" * 50)
    print("LOADING TRAINING DATA")
    print("=" * 50)
    training_sequences, training_molecules = load_training_data(args.training_dir)

    print("\n" + "=" * 50)
    print("PROCESSING SCREENING SETS")
    print("=" * 50)
    process_screening_sets(
        args.screening_dir,
        training_sequences,
        training_molecules,
        args.api_key,
        force=args.force,
    )

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
