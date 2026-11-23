This repository is based on [a third-party reproduction of the EZSpecificity model](https://gitlab.com/cyp_pred_repos/ezspec); this version adds PyTorch Lightning 2.x support, custom CSV inference, and detailed instructions for each inference step.

# EZSpecificity inference


For the model and original research resources, see the [EZSpecificity paper](https://www.nature.com/articles/s41586-025-09697-2), [original code archive](https://zenodo.org/records/17981381), and [upstream reproduction](https://gitlab.com/cyp_pred_repos/ezspec).

## 1. Set up the environments

Use two separate Conda environments. Run the commands below from the repository root.

| Stage | Environment |
| --- | --- |
| Prepare CSV inputs and request NVIDIA ColabFold MSAs | `ezspec` |
| Predict protein–ligand structures with Boltz | `boltz` |
| Convert Boltz CIF files to PDB with `gemmi` | `boltz` |
| Gather complexes, generate features, and run EZSpecificity | `ezspec` |

### EZSpecificity environment

The recommended installation file is [ezspec.minimal.yaml](ezspec.minimal.yaml). 

```bash
conda env create -f ezspec.minimal.yaml
conda activate ezspec
```

This configuration targets **Linux x86_64, Python 3.10, and NVIDIA CUDA 12.8 wheels**. The PyTorch wheel requires glibc 2.28 or newer, and GPU execution requires a compatible NVIDIA driver. The main versions are:

| Package | Version |
| --- | --- |
| PyTorch | `2.8.0+cu128` |
| PyTorch Geometric | `2.8.0.post1` |
| PyG compiled extensions | `pt28cu128` builds |
| PyTorch Lightning | `2.4.0` |
| TorchMetrics | `1.2.0` |

The [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) and [PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) explain the version-specific wheel sources. The YAML requires binary wheels for PyG extensions so that missing wheels produce an installation error rather than triggering local CUDA compilation.

[ezspec.yaml](ezspec.yaml) is the full export of the existing environment, retained as a version reference. Use the minimal file for a new installation.

Check the installation:

```bash
python -m pip check
python -c "import torch, torch_geometric, pytorch_lightning; import pyg_lib, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv; print(torch.__version__, torch.version.cuda, torch_geometric.__version__, pytorch_lightning.__version__)"
```

Installing these prebuilt GPU wheels does not require a visible GPU on a login node. Run GPU availability checks and the GPU stages on a node with an allocated GPU. The minimal file is derived from the working environment.

### Boltz environment

Create the separate Python 3.12 environment using the [Boltz installation instructions](https://github.com/jwohlwend/boltz#installation):

```bash
conda create -n boltz python=3.12 -y
conda activate boltz
python -m pip install -U "boltz[cuda]"
python -c "import gemmi; from importlib.metadata import version; print('boltz', version('boltz')); print('gemmi', gemmi.__version__)"
```


## 2. Download the data and checkpoints

Set the paths for this checkout. Keep the same shell, or repeat these exports when starting a new session:

```bash
cd /path/to/ezspec
export EZSPEC_ROOT="$PWD"
export EZSPEC_DATA="$EZSPEC_ROOT/data_dir"
export EZSPEC_CHECKPOINT="$EZSPEC_ROOT/model_ckpts/best-checkpoint.ckpt"
export BOLTZ_CACHE="$EZSPEC_ROOT/.cache/boltz"

conda activate ezspec
bash ezspec_setup.sh "$EZSPEC_ROOT"
```


Feature generation also loads ESM-2 (`esm2_t33_650M_UR50D`); its weights may be downloaded on first use.

### Download training-set lists and example screening CSVs

Download [infer_data.tar.gz](https://github.com/user-attachments/files/35253084/infer_data.tar.gz) from the [v1.0 release](https://github.com/kclu-bio/ezspec_infer/releases/tag/v1.0) and extract it into `data_dir/`:

```bash
mkdir -p "$EZSPEC_DATA"
curl -fL --retry 3 \
  "https://github.com/user-attachments/files/35253084/infer_data.tar.gz" \
  -o "$EZSPEC_ROOT/infer_data.tar.gz"
tar -xzf "$EZSPEC_ROOT/infer_data.tar.gz" -C "$EZSPEC_DATA"
```

The archive already contains the `esibank/` and `screen/` directory structure. Extract it into `data_dir/`, rather than the repository root or `data_dir/esibank/`. If downloading and placing the CSVs manually, use these exact locations, relative to the repository root:

| CSV file | Destination directory | Purpose |
| --- | --- | --- |
| `ezenzymes.csv` | `data_dir/esibank/` | Training protein sequences (`sequences` column) |
| `ezreactants.csv` | `data_dir/esibank/` | Supplementary training reactant reference list |
| `ezsubstrates.csv` | `data_dir/esibank/` | Training substrate SMILES (`substrates` column) |
| `catnip_screen_binary.csv` | `data_dir/screen/catnip/` | Catnip screening input |
| `vergne_nitrilase_all_binary.csv` | `data_dir/screen/vergne_nitrilase/` | Vergne nitrilase screening input |

After extraction, the five files should be arranged as follows:

```text
data_dir/
├── esibank/
│   ├── ezenzymes.csv
│   ├── ezreactants.csv
│   └── ezsubstrates.csv
└── screen/
    ├── catnip/
    │   └── catnip_screen_binary.csv
    └── vergne_nitrilase/
        └── vergne_nitrilase_all_binary.csv
```

The screening script scans `*.csv` directly under `data_dir/esibank`, reads protein sequences from `sequences`, and reads substrate SMILES from `substrates`. Supply both sets of references for the overlap filter to work. Comparisons use exact sequence strings and the supplied substrate reference strings; the reference substrates should already use the canonical SMILES representation used for screening.

`ezspec_setup.sh` downloads the model data and checkpoints; run the commands above separately to obtain these five inference CSVs. The archive does not include the custom `all_split.csv` dataset described in section 5.

## 3. Configure the NVIDIA MSA API key

Input preparation uses NVIDIA ColabFold MSA Search:

1. Open [NVIDIA ColabFold MSA Search](https://build.nvidia.com/colabfold/msa-search?snippet_tab=Python) and sign in to or register an NVIDIA Developer account.
2. Click **Get API Key**, or open [NVIDIA Build API Keys](https://build.nvidia.com/settings/api-keys).
3. Generate a key, complete account verification if requested, and copy the key when it is shown.
4. Set it in the shell used for input preparation:

```bash
export NVIDIA_API_KEY="YOUR_API_KEY"
```

The generated Boltz YAML files reference local A3M alignments. The Boltz commands below use those alignments and do not need `--use_msa_server`.

## 4. Prepare screening datasets

### 4.1. Arrange the input CSVs

Create one subdirectory per family or screening task, with one input CSV per task directory. The v1.0 archive supplies these two screening tasks:

```text
data_dir/screen/
├── catnip/
│   └── catnip_screen_binary.csv
└── vergne_nitrilase/
    └── vergne_nitrilase_all_binary.csv
```

For another family, place its `<task>_binary.csv` in `data_dir/screen/<task>/` using the same layout.

The screening script reads only filenames ending in **`_binary.csv`**. Each file must contain:

| Column | Contents |
| --- | --- |
| `SEQ` | Protein amino-acid sequence |
| `SUBSTRATES` | Substrate SMILES |
| `Activity` | Binary label: `1` for active, `0` for inactive |

### 4.2. Filter the datasets and generate Boltz inputs — `ezspec`

```bash
conda activate ezspec
python "$EZSPEC_ROOT/src/make_screening_inputs.py" \
  -training-dir "$EZSPEC_DATA/esibank" \
  -screening-dir "$EZSPEC_DATA/screen" \
  -api-key "$NVIDIA_API_KEY"
```

This processes all task subdirectories and:

1. Filters nonspecific substrates, canonicalizes SMILES, removes atom mapping, and drops SMILES that cannot be parsed.
2. Removes a pair if **either its protein sequence or its canonical substrate SMILES occurs in the training references**. This is stricter than removing only exact training pairs.
3. Removes enzymes that have no positive examples after filtering.
4. Requests an MSA for each retained enzyme and writes local A3M files.
5. Writes the filtered CSV, `data.csv`, `Enzymes.csv`, `Substrates.csv`, and one Boltz YAML per prepared pair.

The output directories are `msa/` and `boltz_inputs/` within each task directory. A task with all three generated CSVs and existing MSA/YAML files is skipped on a subsequent run. Use `--force` to reprocess it; existing nonempty MSA files are still reused. Use a fresh task directory when changing the input dataset to avoid retaining stale generated files.

Continue with section 6 for each prepared task.

## 5. Prepare a custom CSV, such as `all_split.csv`

This workflow accepts an arbitrary CSV filename. For the `all_split` example, place it here:

```text
data_dir/esibank/all_split/
└── all_split.csv
```

Supply this custom CSV yourself; it is not included in `infer_data.tar.gz` from the v1.0 release.

`SEQ` and `SUBSTRATES` are required and must contain nonempty values. `Activity` is optional; missing labels default to `0` for the inference data format. These placeholder labels do not support meaningful evaluation metrics.

### 5.1. Generate custom inputs — `ezspec`

```bash
conda activate ezspec
export EZSPEC_TASK_DIR="$EZSPEC_DATA/esibank/all_split"
mkdir -p "$EZSPEC_TASK_DIR"
# Place all_split.csv in EZSPEC_TASK_DIR before running the next command.

python "$EZSPEC_ROOT/src/make_test_input.py" \
  --input-csv "$EZSPEC_TASK_DIR/all_split.csv" \
  --output-dir "$EZSPEC_TASK_DIR" \
  --api-key "$NVIDIA_API_KEY"
```

Note that this script **does not canonicalize SMILES, exclude training overlaps, or remove enzymes without positive examples**. It retains input pairs and duplicates when an MSA can be obtained, while assigning shared IDs to identical sequences and substrate strings. It writes the same three CSVs plus `msa/` and `boltz_inputs/`.

Matching existing A3M files are reused. Transient MSA request failures are retried; sequences that still fail, and their associated pairs, are skipped. Inspect `failed_sequences.csv` for the failures. The generated `data.csv` contains the pairs that can proceed.

For another label column, add `--label-column COLUMN`; to change the fallback label, add `--default-label VALUE`. The original CSV's extra metadata columns are not copied into the generated model input tables.

Continue with section 6, keeping `EZSPEC_TASK_DIR` set to this custom dataset directory.

## 6. Predict structures, generate features, and run inference

The following steps apply to **both** input workflows. Run them in order for one task at a time.

For a screening task, select the prepared directory:

```bash
export EZSPEC_TASK_DIR="$EZSPEC_DATA/screen/catnip"
```

For the other supplied screening task, set `EZSPEC_TASK_DIR="$EZSPEC_DATA/screen/vergne_nitrilase"` instead.

For custom `all_split` inference, use this instead:

```bash
export EZSPEC_TASK_DIR="$EZSPEC_DATA/esibank/all_split"
```

### 6.1. Predict protein–ligand complexes — `boltz`

```bash
conda activate boltz
boltz predict "$EZSPEC_TASK_DIR/boltz_inputs" \
  --out_dir "$EZSPEC_TASK_DIR/boltz_outputs" \
  --cache "$BOLTZ_CACHE"
```

The expected prediction directory is:

```text
<task>/boltz_outputs/boltz_results_boltz_inputs/predictions/
```

The first Boltz run may download model weights into the cache. Check that prediction CIFs have been produced before continuing.

### 6.2. Convert CIF predictions to PDB — `boltz`

Stay in the `boltz` environment so that `src/convert.py` uses its `gemmi`:

```bash
python "$EZSPEC_ROOT/src/convert.py" \
  --input-dir "$EZSPEC_TASK_DIR/boltz_outputs/boltz_results_boltz_inputs/predictions" \
  --output-dir "$EZSPEC_TASK_DIR/boltz_outputs/boltz_results_boltz_inputs/pdb_files"
```

The converter reads `*/*_model_0.cif` and writes the corresponding `*_model_0.pdb` files. Later stages use model 0 from each prediction.

### 6.3. Gather protein–ligand complexes — `ezspec`

```bash
conda activate ezspec
python "$EZSPEC_ROOT/src/gather_inputs_screening.py" \
  -pdb-dir "$EZSPEC_TASK_DIR/boltz_outputs/boltz_results_boltz_inputs/pdb_files" \
  -output-dir "$EZSPEC_TASK_DIR" \
  -data-df "$EZSPEC_TASK_DIR/data.csv" \
  -substrate-file "$EZSPEC_TASK_DIR/Substrates.csv"
```

This writes complexes under `<task>/Structure/complex/`. It also updates `data.csv`, dropping rows with missing prediction PDBs or conversion errors. Check the reported number of retained rows.

### 6.4. Generate EZSpecificity features — `ezspec`

```bash
python "$EZSPEC_ROOT/src/preprocess.py" \
  --prepared_data_dir "$EZSPEC_TASK_DIR" \
  --features_output_dir "$EZSPEC_TASK_DIR" \
  --src_root_dir "$EZSPEC_ROOT" \
  --data_root_dir "$EZSPEC_DATA/ezspec_data"
```

This generates ESM enzyme embeddings, substrate graph features, Morgan fingerprints, GROVER fingerprints, and protein-pocket/ligand structure features. The script runs the GROVER subprocesses itself; there is no separate manual GROVER step. Its pretrained checkpoint is resolved as `data_dir/ezspec_data/data/pretrain_model/grover_large.pt`.

### 6.5. Run EZSpecificity inference — `ezspec`

```bash
mkdir -p "$EZSPEC_TASK_DIR/logs"
python "$EZSPEC_ROOT/src/run_inference.py" \
  --src-root-dir "$EZSPEC_ROOT" \
  --logging-dir "$EZSPEC_TASK_DIR/logs" \
  --features-output-dir "$EZSPEC_TASK_DIR" \
  --prepared-data-dir "$EZSPEC_TASK_DIR" \
  --ckpt-path "$EZSPEC_CHECKPOINT"
```

The result is written inside the selected task directory:

```text
# Screening example
data_dir/screen/catnip/example_out_YYYY-MM-DD_HH-MM-SS.csv

# Custom CSV example
data_dir/esibank/all_split/example_out_YYYY-MM-DD_HH-MM-SS.csv
```

The result includes the indexed pair metadata and a `score` column containing model logits. Higher scores indicate stronger predicted activity; these raw scores are not probabilities. Resolve enzyme and substrate IDs through `Enzymes.csv` and `Substrates.csv`. Input preparation, missing structures, and feature validity checks can reduce the number of final predictions, so compare the stage logs with the original input size.

## 7. Makefile shortcuts

The Makefile provides shortcuts for predefined tasks. Before using them, set `BASE_DIR` to your checkout, replace the Boltz `--cache` paths with a writable location, and set the task directories as needed. GNU `make` runs these targets; CMake is not required for the shortcuts.

For example, run screening preparation from the repository root with:

```bash
conda activate ezspec
make preprocess BASE_DIR="$EZSPEC_ROOT" NVIDIA_API_KEY="$NVIDIA_API_KEY"
```

| Step | Environment | Vergne nitrilase | Catnip | Custom `all_split` |
| --- | --- | --- | --- | --- |
| Prepare inputs | `ezspec` | `make preprocess` | `make preprocess` | `make ALL_SPLIT_input` |
| Predict structures | `boltz` | `make VN_boltz` | `make catnip_boltz` | `make ALL_SPLIT_boltz` |
| Convert CIF to PDB | `boltz` | `make VN_cif_convert` | `make catnip_cif_convert` | `make ALL_SPLIT_cif_convert` |
| Gather complexes | `ezspec` | `make VN_gather` | `make catnip_gather` | `make ALL_SPLIT_gather` |
| Generate features | `ezspec` | `make VN_process` | `make catnip_process` | `make ALL_SPLIT_process` |
| Run inference | `ezspec` | `make VN_inference` | `make catnip_inference` | `make ALL_SPLIT_inference` |

`make preprocess` scans all screening task directories once. The `VN_*` targets point to `data_dir/screen/vergne_nitrilase`, `catnip_*` to `data_dir/screen/catnip`, and `ALL_SPLIT_*` to `data_dir/esibank/all_split`. For other task names, use the direct commands in sections 4–6 or override the corresponding Makefile task path. Pass `NVIDIA_API_KEY="$NVIDIA_API_KEY"` to `make ALL_SPLIT_input` as well, unless you have configured that Makefile variable.

## 8. Troubleshooting

- **PyTorch/PyG installation errors:** use the wheel sources in `ezspec.minimal.yaml` and keep the Torch version, CUDA suffix, and PyG extension build suffix matched. If changing CUDA variants, change the whole combination together.
- **`No module named gemmi`:** activate `boltz` before conversion and check its `gemmi` installation.
- **MSA authentication or request failures:** check the NVIDIA key and service response. For custom CSVs, inspect `failed_sequences.csv`; a failed sequence causes its pairs to be excluded from preparation.
- **No screening inputs or no retained pairs:** check the `_binary.csv` suffix, required columns, training-overlap counts, and whether each retained enzyme has a positive example. Use the custom CSV workflow when screening exclusion is not intended.
- **Missing Boltz outputs:** check that the selected task and `boltz_results_boltz_inputs/predictions` directory match. The current converter expects model-0 CIFs in per-input subdirectories.
- **Out of GPU memory:** reduce the workload for the failing stage. EZSpecificity inference currently sets its batch size to `8` in `src/run_inference.py`; changing only `configs/train_example.yml` does not override that inference value. Check `boltz predict --help` for Boltz-specific options.
- **Fewer predictions than input rows:** inspect MSA failures, the gather step's dropped-row count, feature-generation messages, and `<task>/skip_log.txt`. The feature code also skips protein sequences longer than 1,450 residues or containing unsupported amino acids.


