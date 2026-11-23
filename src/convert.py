from pathlib import Path
import os
import gemmi
import argparse
from tqdm import tqdm

# python src/convert.py --input-dir /home/kechenlu/kechenlu_link/enzymelm/baseline_model/ezspec/data_dir/screen/vergne_nitrilase/boltz_outputs/boltz_results_boltz_inputs/predictions --output-dir /home/kechenlu/kechenlu_link/enzymelm/baseline_model/ezspec/data_dir/screen/vergne_nitrilase/boltz_outputs/boltz_results_boltz_inputs/pdb_files 
# transfer boltz output cif into pdb file

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)


args = parser.parse_args()
src = args.input_dir
dst = args.output_dir

os.makedirs(dst, exist_ok = True)
count = 0
for cif in tqdm(src.glob("*/*_model_0.cif")):
    output = dst / f"{cif.stem}.pdb"
    gemmi.read_structure(str(cif)).write_pdb(str(output))
    count += 1

print(f"Converted {count} CIF files into {dst}")