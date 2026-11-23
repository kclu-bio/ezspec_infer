## Cifs must already be converted to pdb files

import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pdbs-dir", type=Path, required=True)
    parser.add_argument("-holdout-df", type=Path, required=True)
    parser.add_argument("-rxn-files-dir", type=Path, required=True)
    parser.add_argument("-output-dir", type=Path, required=True)

    return parser.parse_args()


def add_sequence_to_holdout_df(holdout_df: Path, rxn_files: Path):
    """
    Add sequence to holdout dataframe

    Args:
        holdout_df: Path to the holdout dataframe
        rxn_files: Path to the directory containing the rxn files

    Returns:
        df: Holdout dataframe with sequence column
    """
    df = pd.read_csv(holdout_df)
    df["sequence"] = None

    for rxn_file in rxn_files.glob("*.csv"):
        rxn_id = rxn_file.stem.replace("r_", "")
        rxn_df = pd.read_csv(rxn_file)
        uniprot = rxn_id.split("_")[0]
        sequence = rxn_df.loc[rxn_df["protein_id"] == uniprot, "sequence"].values[0]
        df.loc[df["sample_id"] == rxn_id, "sequence"] = sequence

    return df


def map_pdb_to_sequence_substrate(
    cifs_dir: Path, holdout_df: pd.DataFrame, rxn_files: Path
):
    """
    Collect all pdb files, filter out pdb files not in holdout_df
    Get label of each pdb file
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

    rxn_files = list(rxn_files.glob("*.csv"))

    pdb_files = list(cifs_dir.glob("*.pdb"))
    holdout = holdout_df

    unique_substrates = set()
    unique_sequences = set()
    unique_uniprots = set()
    pdb_to_sequence_substrate = {}

    for pdb in pdb_files:
        pdb_id = pdb.stem
        label = int("correct" in pdb_id)
        sample_id = "_".join(pdb_id.split("_")[0:3])

        if sample_id not in holdout["sample_id"].values:
            continue

        rxn_file = next(
            (rxn_file for rxn_file in rxn_files if sample_id in rxn_file.stem), None
        )
        if rxn_file is None:
            continue

        rxn_df = pd.read_csv(rxn_file)

        if label:
            uniprot = sample_id.split("_")[0]
            sequence = rxn_df.loc[rxn_df["protein_id"] == uniprot, "sequence"].values[0]

        else:
            sequence_index = int(pdb_id.split("_")[3])
            sequence = rxn_df.iloc[sequence_index]["sequence"]
            uniprot = rxn_df.iloc[sequence_index]["protein_id"]

        holdout_df_row = holdout[holdout["sample_id"] == sample_id]
        substrate = holdout_df_row["substrate"].values[0]

        unique_substrates.add(substrate)
        unique_sequences.add(sequence)
        unique_uniprots.add(uniprot)
        pdb_to_sequence_substrate[pdb_id] = (sequence, substrate, label, uniprot)

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
    ) in pdb_to_sequence_substrate.items():
        substrate_index = substrate_to_index[substrate]
        sequence_index = sequence_to_index[sequence]
        pdb_to_sequence_substrate[pdb_id] = (
            sequence_index,
            substrate_index,
            label,
            uniprot,
        )

    pdb_to_sequence_substrate_list = [
        {
            "pdb_id": pdb_id,
            "Enzyme Index": seq_idx,
            "Substrate Index": sub_idx,
            "Label": label,
            "uniprot": uniprot,
        }
        for pdb_id, (
            seq_idx,
            sub_idx,
            label,
            uniprot,
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
    pdb: Path, df: pd.DataFrame, holdout_df: pd.DataFrame, output_dir: Path
):
    """
    Convert pdb file into acceptable format for pocket extraction
    Save the converted pdb file to a new directory

    Args:
        pdb: Path to the pdb file
        df: DataFrame containing the Dock Index and pdb_id
        output_dir: Path to the directory to save the converted pdb file

    Returns:
        None
    """

    pdb_id = pdb.stem
    sample_id = "_".join(pdb_id.split("_")[0:3])
    pdb_index = df[df["pdb_id"] == pdb_id]["Dock Index"].values[0]
    substrate = holdout_df[holdout_df["sample_id"] == sample_id]["substrate"].values[0]
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
    pdbs_dir = args.pdbs_dir
    holdout_df = args.holdout_df
    rxn_files_dir = args.rxn_files_dir
    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_output_dir = output_dir / "Structure" / "complex"
    pdb_output_dir.mkdir(parents=True, exist_ok=True)

    holdout_df = add_sequence_to_holdout_df(holdout_df, rxn_files_dir)

    print("Mapping pdb files to sequence and substrate")
    (
        pdb_to_sequence_substrate_df,
        unique_substrates,
        unique_sequences,
        unique_uniprots,
    ) = map_pdb_to_sequence_substrate(pdbs_dir, holdout_df, rxn_files_dir)
    
    print("Converting pdb files to acceptable format")
    
    for _, row in tqdm(
        pdb_to_sequence_substrate_df.iterrows(),
        desc="Converting pdb files to acceptable format",
        total=len(pdb_to_sequence_substrate_df),
    ):
        pdb_id = row["pdb_id"]
        convert_pdb(
            pdbs_dir / f"{pdb_id}.pdb",
            pdb_to_sequence_substrate_df,
            holdout_df,
            pdb_output_dir,
        )

    print("Saving data.csv to output directory", output_dir)
    pdb_to_sequence_substrate_df.to_csv(output_dir / "data.csv", index=False)
    substrate_df = pd.DataFrame(list(unique_substrates), columns=["Substrate_SMILES"])
    substrate_df.to_csv(output_dir / "Substrates.csv", index=False)
    sequence_df = pd.DataFrame(list(unique_sequences), columns=["Protein sequence"])
    # sequence_df["uniprot"] = unique_uniprots
    sequence_df.to_csv(output_dir / "Enzymes.csv", index=False)


if __name__ == "__main__":
    main()
