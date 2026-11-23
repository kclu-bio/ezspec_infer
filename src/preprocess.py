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
import os
import subprocess
import sys
import pandas as pd

############################################################
### 0 Generate enzyme features
############################################################

## my actual data; no O48786 b/c OOM issues
#features_output_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/cyp_benchmark/no_O48786"
#prepared_data_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/cyp_benchmark/no_O48786"

## my TRAINING data
#features_output_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/cyp_benchmark/data_inputs/train/processed_data"
#prepared_data_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/cyp_benchmark/data_inputs/train/processed_data"

## my total full data
#features_output_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/cyp_benchmark/data_inputs/full_data_benchmark"
#prepared_data_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/cyp_benchmark/data_inputs/full_data_benchmark"

## original example data, came with repo
# features_output_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/data/toy_example"
# prepared_data_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/data/toy_example"

## my teeny benchmark
# features_output_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/cyp_benchmark/data_inputs/test_cifs/test_inputs"
# prepared_data_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/cyp_benchmark/data_inputs/test_cifs/test_inputs"

## my larger benchmark
#features_output_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/cyp_benchmark/data_inputs/test_cifs/test_inputs/large_run"
#prepared_data_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/cyp_benchmark/data_inputs/test_cifs/test_inputs/large_run"

#src_root_dir = "/vast/projects/pranam/lab/emahood/repos/ezspec/"
#data_root_dir = "/vast/projects/pranam/lab/emahood/penn_work/ezspecificity/"

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_data_dir", type=Path, required=True)
    parser.add_argument("--features_output_dir", type=Path, required=True)
    parser.add_argument("--src_root_dir", type=Path, required=True)
    parser.add_argument("--data_root_dir", type=Path, required=True)
    return parser.parse_args()

def single_extract_pocket(names, structure_dir: Path):
    bad_mol = 0
    for name in names:
        id = int(name.split("/")[-1].split(".")[0])

        pocket_out_path = f"{structure_dir}/str_tmp_data/pocket/{id}.pdb"
        ligand_out_path = f"{structure_dir}/str_tmp_data/raw_ligand/{id}.sdf"
        lines = []
        ed = 0
        for index, line in enumerate(open(name, "r")):
            lines.append(line)
            if "COMPND" in line:
                ed = index

        if ed == 0:
            raise ValueError(f"No COMPND found in {name}")

        protein_lines = lines[:ed]
        ligand_lines = lines[ed + 1 :]
        ftmp_out = open(f"{structure_dir}/str_tmp_data/raw_ligand/tmp_{id}.pdb", "w")
        for line in ligand_lines:
            ftmp_out.write(line)
        ftmp_out.close()

        mol = Chem.MolFromPDBFile(
            f"{structure_dir}/str_tmp_data/raw_ligand/tmp_{id}.pdb",
            flavor=1,
            sanitize=False,
        )
        # os.system(f"rm -r {structure_dir}/str_tmp_data/raw_ligand/tmp_{id}.pdb")
        if mol is None:
            print(f"!!!!!{id}!!!!!")
            bad_mol += 1
            raise ValueError(f"Mol is None for {name}")

        ligand_coords = []
        # mol = Chem.RemoveHs(mol, sanitize=False)
        for i, atom in enumerate(mol.GetAtoms()):
            positions = mol.GetConformer().GetAtomPosition(i)
            ligand_coords.append((positions.x, positions.y, positions.z))

        if len(ligand_coords) == 0:
            raise ValueError(f"No ligand coordinates found for {name}")

        try:
            writer = Chem.SDWriter(ligand_out_path)
            writer.write(mol, confId=0)
            writer.close()
        except:
            bad_mol += 1
            raise ValueError(f"Writer is None for {name}")

        fin = open(pocket_out_path, "w")

        for line in protein_lines:
            if "ATOM" in line:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    for ligand_coord in ligand_coords:
                        if (
                            "H" not in line[12:16].strip()
                            and distance(
                                x,
                                y,
                                z,
                                ligand_coord[0],
                                ligand_coord[1],
                                ligand_coord[2],
                            )
                            < 10
                        ):
                            fin.write(line)
                            break
                except:
                    continue

            elif "HETATM" in line or "ENDMDL" in line:
                fin.write(line)
    return bad_mol


def distance(x1, y1, z1, x2, y2, z2):
    return (1.0 * (x2 - x1) ** 2 + 1.0 * (y2 - y1) ** 2 + 1.0 * (z2 - z1) ** 2) ** 0.5


def AssignBondOrdersFromTemplate(refmol, mol):
    """assigns bond orders to a molecule based on the
        bond orders in a template molecule
    Revised from AllChem.AssignBondOrderFromTemplate(refmol, mol)
    """
    AllChem.AssignBondOrdersFromTemplate
    refmol2 = Chem.rdchem.Mol(refmol)
    mol2 = Chem.rdchem.Mol(mol)
    # do the molecules match already?
    matching = mol2.GetSubstructMatch(refmol2)
    if not matching:  # no, they don't match
        # check if bonds of mol are SINGLE
        for b in mol2.GetBonds():
            if b.GetBondType() != Chem.BondType.SINGLE:
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False)
        # set the bonds of mol to SINGLE
        for b in refmol2.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        # set atom charges to zero;
        for a in refmol2.GetAtoms():
            a.SetFormalCharge(0)
        for a in mol2.GetAtoms():
            a.SetFormalCharge(0)

        matching = mol2.GetSubstructMatches(refmol2, uniquify=False)
        # do the molecules match now?
        if matching:
            if len(matching) > 1:
                # logger.warning("More than one matching pattern found - picking one")
                pass
            matchings = matching[:]
            for matching in matchings:
                # matching = matching[0] ## use each matching
                # apply matching: set bond properties
                for b in refmol.GetBonds():
                    atom1 = matching[b.GetBeginAtomIdx()]
                    atom2 = matching[b.GetEndAtomIdx()]
                    b2 = mol2.GetBondBetweenAtoms(atom1, atom2)
                    b2.SetBondType(b.GetBondType())
                    b2.SetIsAromatic(b.GetIsAromatic())
                # apply matching: set atom properties
                for a in refmol.GetAtoms():
                    a2 = mol2.GetAtomWithIdx(matching[a.GetIdx()])
                    a2.SetHybridization(a.GetHybridization())
                    a2.SetIsAromatic(a.GetIsAromatic())
                    a2.SetNumExplicitHs(a.GetNumExplicitHs())
                    a2.SetFormalCharge(a.GetFormalCharge())
                try:
                    Chem.SanitizeMol(mol2)
                    if hasattr(mol2, "__sssAtoms"):
                        mol2.__sssAtoms = None  # we don't want all bonds highlighted
                    break
                except ValueError:
                    pass
                    # print("More than one matching pattern, Fail at this matching. Try next.")
        else:
            raise ValueError("No matching found")
    return mol2


def alignment_number_system(sdf, smile_mol):

    template = smile_mol
    query = sdf

    mcs = rdFMCS.FindMCS([template, query], timeout=120)
    patt = Chem.MolFromSmarts(mcs.smartsString)

    query_match = query.GetSubstructMatch(patt)
    template_match = template.GetSubstructMatch(patt)

    result = [-1] * query.GetNumAtoms()

    for query_atom_id, template_atom_id in zip(query_match, template_match):
        result[query_atom_id] = template_atom_id

    # Check if there is any atom not matched
    for atom in query.GetAtoms():
        assert atom.GetAtomicNum() == 1 or result[atom.GetIdx()] != -1

    return result


def assign_idx(mol, idxs):
    for atom, idx in zip(mol.GetAtoms(), idxs):
        atom.SetAtomMapNum(idx)
    return mol


def mol_get_atomic_number(mol, atom_map=False):
    result = [0] * mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() != -1:
            if atom_map:
                result[atom.GetAtomMapNum()] = atom.GetAtomicNum()
            else:
                result[atom.GetIdx()] = atom.GetAtomicNum()
    return result


def check(mol, mol2):
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() != -1:
            id = atom.GetAtomMapNum()

            atom2 = mol2.GetAtomWithIdx(id)
            if atom.GetAtomicNum() != atom2.GetAtomicNum():
                return False
    return True


def view_difference(mol1, mol2):
    mcs = rdFMCS.FindMCS([mol1, mol2])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    match1 = mol1.GetSubstructMatch(mcs_mol)
    target_atm1 = []
    for atom in mol1.GetAtoms():
        if atom.GetIdx() not in match1:
            target_atm1.append(atom.GetIdx())
    match2 = mol2.GetSubstructMatch(mcs_mol)
    target_atm2 = []
    for atom in mol2.GetAtoms():
        if atom.GetIdx() not in match2:
            target_atm2.append(atom.GetIdx())
    return Draw.MolsToGridImage(
        [mol1, mol2], highlightAtomLists=[target_atm1, target_atm2]
    )


def single_match(item, structure_dir: Path):
    id, smile = item
    sdf_path = f"{structure_dir}/str_tmp_data/raw_ligand/{id}.sdf"
    RDLogger.DisableLog("rdApp.*")

    try:
        mol = next(iter(Chem.SDMolSupplier(sdf_path, sanitize=True)))
        mol = Chem.RemoveHs(mol)
    except Exception as e:
        try:
            mol = next(iter(Chem.SDMolSupplier(sdf_path, sanitize=False)))
            mol = Chem.RemoveHs(mol, sanitize=False)
        except Exception as e:
            return 1
    # mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    try:
        smile_mol = Chem.MolFromSmiles(smile)
    except Exception as e:
        return 1
    # print(mol_get_atomic_number(smile_mol))
    # print(len(smile_mol.GetAtoms()))
    try:
        aligned_idx = alignment_number_system(mol, smile_mol)
    except:

        try:
            mol = AssignBondOrdersFromTemplate(smile_mol, mol)
        except:
            return 1

        try:
            aligned_idx = alignment_number_system(mol, smile_mol)
        except:
            return 1
    mol = assign_idx(mol, aligned_idx)
    # print(mol_get_atomic_number(mol, atom_map=True))
    if not check(mol, smile_mol):
        return 1

    w = Chem.SDWriter(f"{structure_dir}/str_tmp_data/ligand/{id}.sdf")
    try:
        w.write(mol)
        w.close()
    except:
        w.close()
        return 1

    return 0


def batched_match(items, structure_dir: Path):
    ans = 0
    for item in items:
        id, smile = item
        try:
            result = single_match(item, structure_dir)
            ans += result
        except:
            ans += 1

        if not result:
            os.system(f"rm -r {structure_dir}/str_tmp_data/raw_ligand/{id}.sdf")

    return ans


def run(cmd, *, cwd: Path, env: dict | None = None):
    """Run a command with nice errors and streaming output."""
    print(f"\n[cmd] {' '.join(cmd)}\n[cwd] {cwd}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main():

    args = argparser()
    prepared_data_dir = args.prepared_data_dir
    features_output_dir = args.features_output_dir
    src_root_dir = args.src_root_dir
    data_root_dir = args.data_root_dir

    enzyme_csv_path = f"{prepared_data_dir}/Enzymes.csv"
    substrate_csv_path = f"{prepared_data_dir}/Substrates.csv"
    metadata_csv_path = f"{prepared_data_dir}/data.csv"
    structure_dir = f"{prepared_data_dir}/Structure"

    df = pd.read_csv(enzyme_csv_path, sep=",")
    env = lmdb.open(
        f"{features_output_dir}/enzyme_features.lmdb",
        map_size=600 * (1024 * 1024 * 1024),  # 600GB
        create=True,
        subdir=False,
        readonly=False,  # Writable
    )
    data = []
    sequences = []
    uniprot_dict = {}
    for index, sequence in enumerate(df["Protein sequence"]):
        fake_uniprot_id = str(index)
        if fake_uniprot_id in uniprot_dict:
            print(f"{index}th protein already in lmdb")
        if len(sequence) > 1450:
            print(f"{index}th protein sequence too long")
            continue
        try:
            convert_protein_sequence_to_number(sequence)
        except:
            print(f"{index}th proteinsequence contain non-standard amino acid")
            continue
        sequences.append(sequence)
        uniprot_dict[fake_uniprot_id] = (len(uniprot_dict), 1)
        data.append((fake_uniprot_id, sequence))

    print(f"Generating enzyme features for {len(data)} enzymes")
    generate_esm_embedding(env, data, uniprot_dict)
    env.close()

    ############################################################
    ### 1 Generate graph features
    ############################################################

    reactions = []
    substrates = []

    df = pd.read_csv(substrate_csv_path, sep=",").dropna(subset=["Substrate_SMILES"])
    parameters_dict = {}
    for substrate in df["Substrate_SMILES"]:
        left = substrate
        right = substrate
        if substrate not in parameters_dict and len(substrate) < 275:
            parameters_dict[substrate] = (
                len(parameters_dict) - 1,
                (f"{substrate}>>{substrate}", substrate, right, True),
            )

    parameters = [parameter[1] for parameter in parameters_dict.values()]
    db = lmdb.open(
        f"{features_output_dir}/reaction_features.lmdb",
        map_size=600 * (1024 * 1024 * 1024),  # 600GB
        create=True,
        subdir=False,
        readonly=False,  # Writable
    )
    # with Pool(60) as pool:
    results = [get_reaction_feature_single(parameter) for parameter in parameters]
    print(f"Saving reaction features for {len(results)} reactions")
    for full_reaction, substrate, tag, data in results:
        if (data is not None) or (tag is None):
            with db.begin(write=True, buffers=True) as txn:
                txn.put(key=str(len(reactions)).encode(), value=pickle.dumps(data))
                reactions.append(full_reaction)
                substrates.append(substrate)
        else:
            print("Skipped %s %s" % (full_reaction, tag))
    db.close()

    ############################################################
    ### 2 Generate morgan fingerprint features
    ############################################################

    df = pd.read_csv(substrate_csv_path, sep=",").dropna(subset=["Substrate_SMILES"])
    results = []
    for substrate in df["Substrate_SMILES"]:
        mol = AllChem.MolFromSmiles(substrate)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        results.append(arr)
    np.save(f"{features_output_dir}/morgan_fingerprint.npy", np.array(results))

    ############################################################
    ### 3 Generate grover embeddings
    ############################################################

    df = pd.read_csv(substrate_csv_path, sep=",").dropna(subset=["Substrate_SMILES"])

    results = [smile for smile in df["Substrate_SMILES"]]
    data = {"Substrate_SMILES": results}
    data = pd.DataFrame(data)
    data.to_csv(f"{features_output_dir}/grover_substrates.csv", index=False)

    grover_dir = (Path(src_root_dir) / "src" / "other_softwares" / "grover_software").resolve()
    # Subprocess cwd is grover_dir; relative data paths would resolve under grover_software.
    feat_abs = Path(features_output_dir).resolve()
    data_abs = Path(data_root_dir).resolve()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{grover_dir}{os.pathsep}{env.get('PYTHONPATH','')}"

    # Script argv is relative to cwd=grover_dir; file paths must be absolute.
    run(
        [
            sys.executable,
            "scripts/save_features.py",
            "--data_path",
            str(feat_abs / "grover_substrates.csv"),
            "--save_path",
            str(feat_abs / "grover_substrates.npz"),
            "--features_generator",
            "fgtasklabel",
            "--restart",
        ],
        cwd=grover_dir,
        env=env,
    )

    vocab_dir = feat_abs / "grover_vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "scripts/build_vocab.py",
            "--data_path",
            str(feat_abs / "grover_substrates.csv"),
            "--vocab_save_folder",
            str(vocab_dir.resolve()),
            "--dataset_name",
            "test",
        ],
        cwd=grover_dir,
        env=env,
    )

    checkpoint = data_abs / "data" / "pretrain_model" / "grover_large.pt"
    out_fp = feat_abs / "grover_fingerprint.npz"
    out_lmdb = feat_abs / "grover_fingerprint.lmdb"

    run(
        [
            sys.executable,
            "main.py",
            "fingerprint",
            "--data_path",
            str(feat_abs / "grover_substrates.csv"),
            "--features_path",
            str(feat_abs / "grover_substrates.npz"),
            "--checkpoint_path",
            str(checkpoint),
            "--fingerprint_source",
            "both",
            "--output",
            str(out_fp),
            "--save_lmdb_path",
            str(out_lmdb),
        ],
        cwd=grover_dir,
        env=env,
    )

    print(
        f"export PYTHONPATH={src_root_dir}/src/other_softwares/grover_software:$PYTHONPATH"
    )
    print(
        f"python {src_root_dir}/src/other_softwares/grover_software/scripts/save_features.py --data_path {features_output_dir}/grover_substrates.csv  \
                                        --save_path {features_output_dir}/grover_substrates.npz   \
                                        --features_generator fgtasklabel \
                                        --restart"
    )

    print(f"mkdir {features_output_dir}/grover_vocab")
    print(
        f"python {src_root_dir}/src/other_softwares/grover_software/scripts/build_vocab.py --data_path {features_output_dir}/grover_substrates.csv \
                                --vocab_save_folder {features_output_dir}/grover_vocab  \
                                --dataset_name test"
    )
    print(
        f"python main.py fingerprint --data_path {features_output_dir}/grover_substrates.csv --features_path {features_output_dir}/grover_substrates.npz --checkpoint_path {data_root_dir}/data/pretrain_model/grover_large.pt --fingerprint_source both --output {features_output_dir}/grover_fingerprint.npz  --save_lmdb_path {features_output_dir}/grover_fingerprint.lmdb --fingerprint_source both"
    )

    ############################################################
    ### 4 Preprocess generated pocket
    ############################################################

    os.makedirs(f"{structure_dir}/str_tmp_data", exist_ok=True)
    os.makedirs(f"{structure_dir}/str_tmp_data/raw_ligand", exist_ok=True)
    os.makedirs(f"{structure_dir}/str_tmp_data/pocket", exist_ok=True)

    mol = None

    print(
        f"Extracting pockets for {len(glob.glob(f'{structure_dir}/complex/*.pdb'))} complexes"
    )
    for name in tqdm(glob.glob(f"{structure_dir}/complex/*.pdb")):
        single_extract_pocket([name], structure_dir)

    print(
        f"After extracting pockets! {len(glob.glob(f'{structure_dir}/str_tmp_data/pocket/*.pdb'))} pockets extracted"
    )
    print(
        f"After extracting pockets! {len(glob.glob(f'{structure_dir}/str_tmp_data/raw_ligand/*.sdf'))} raw ligands extracted"
    )

    ############################################################
    ### 5 Align substrate in docking pocket and substrate in smiles
    ############################################################

    os.makedirs(f"{structure_dir}/str_tmp_data/ligand", exist_ok=True)

    df = pd.read_csv(metadata_csv_path)
    print(df)
    sub_index_dict = {
        int(index): int(substrate_id)
        for index, substrate_id in zip(
            df["Dock Index"].values, df["Substrate Index"].values
        )
    }

    sub_df = pd.read_csv(substrate_csv_path)
    substrates = sub_df["Substrate_SMILES"].values

    bad_molecular = 0

    parameters = []
    for index, sdf_path in tqdm(
        enumerate(glob.glob(f"{structure_dir}/str_tmp_data/raw_ligand/*.sdf"))
    ):

        id = os.path.basename(sdf_path).split(".")[0]
        if int(id) not in sub_index_dict:
            continue

        substrate_id = int(sub_index_dict[int(id)])
        smile = substrates[substrate_id]
        parameters.append((int(id), smile))

    results = [
        batched_match(parameters[i * 100 : (i + 1) * 100], structure_dir)
        for i in range(len(parameters) // 100 + 1)
    ]
    bad_molecular = 0
    for result in results:
        bad_molecular += result
    print(bad_molecular)

    ############################################################
    ### 6 create structure features
    ############################################################

    dict = {
        "data_df_path": metadata_csv_path,
        "ligand_df_path": substrate_csv_path,
        "data": {
            "structure_processed_path": f"{features_output_dir}/structure_features.lmdb",
            "sequence_processed_path": f"{features_output_dir}/str_features.lmdb",
            "pdb_dir": f"{structure_dir}/str_tmp_data/pocket",
            "ligand_dir": f"{structure_dir}/str_tmp_data/ligand",
        },
    }
    config = edict(dict)
    df = pd.read_csv(config.data_df_path)
    str_process(config, df)
    df = pd.read_csv(config.ligand_df_path)
    seq_process(config, df)

    ############################################################
    ### 7 Generate high quality structure id
    ############################################################

    ligand_ids = []
    for name in tqdm(glob.glob(f"{structure_dir}/str_tmp_data/ligand/*.sdf")):
        ligand_ids.append(name.split("/")[-1].split(".")[0])

    pocket_ids = []
    for name in tqdm(glob.glob(f"{structure_dir}/str_tmp_data/pocket/*.pdb")):
        pocket_ids.append(name.split("/")[-1].split(".")[0])

    ids = set(ligand_ids).intersection(set(pocket_ids))
    fout = open(f"{features_output_dir}/high_quality_id.txt", "w")
    for id in ids:
        fout.write(id + "\n")
    fout.close()

    print(f"Prepared data saved to {prepared_data_dir}")
    print("Done!")

if __name__ == "__main__":
    main()