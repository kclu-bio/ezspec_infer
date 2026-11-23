"""Prepare an arbitrary enzyme/substrate CSV for Boltz and EZSpecificity.

Unlike ``make_screening_inputs.py``, this script intentionally does not
canonicalize SMILES, remove training-set overlaps, or filter enzymes based on
their labels. Input rows are retained unless their enzyme MSA cannot be made
within the configured retry limit.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm


RETRYABLE_MSA_STATUS_CODES = {429, 500, 502, 503, 504}
DATA_COLUMNS = ["Substrate Index", "Enzyme Index", "Label", "Dock Index"]
FAILED_SEQUENCE_COLUMNS = ["Enzyme Index", "Protein sequence", "Error"]


def _get_positive_number(name: str, default: float, number_type):
    """Read a positive numeric setting from the environment."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = number_type(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {raw_value!r}")
    return value


def reusable_a3m_path(
    sequence: str, output_dir: Path, enzyme_id: int
) -> Optional[Path]:
    """Return an existing A3M path when its query matches ``sequence``."""
    output_path = output_dir / f"{enzyme_id}.a3m"
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return None

    try:
        query_lines = []
        found_header = False
        with output_path.open() as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if found_header:
                        break
                    found_header = True
                    continue
                if found_header:
                    query_lines.append(line)
        query_sequence = "".join(query_lines).replace("-", "").upper()
    except OSError:
        return None

    if query_sequence != sequence.upper():
        return None
    return output_path.resolve()


def make_msa(sequence: str, output_dir: Path, enzyme_id: int, api_key: str) -> Path:
    """Generate (or reuse) the ColabFold A3M file for one enzyme."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{enzyme_id}.a3m"

    existing_path = reusable_a3m_path(sequence, output_dir, enzyme_id)
    if existing_path is not None:
        return existing_path
    if not api_key:
        raise ValueError(
            "An NVIDIA API key is required when a matching A3M file does not exist"
        )

    url = os.getenv(
        "URL", "https://health.api.nvidia.com/v1/biology/colabfold/msa-search/predict"
    )
    max_attempts = _get_positive_number("NVIDIA_MSA_MAX_ATTEMPTS", 8, int)
    backoff_seconds = _get_positive_number("NVIDIA_MSA_BACKOFF_SECONDS", 10.0, float)
    poll_seconds = _get_positive_number("NVIDIA_MSA_POLL_SECONDS", 600, int)
    connect_timeout = _get_positive_number(
        "NVIDIA_MSA_CONNECT_TIMEOUT", 30.0, float
    )
    read_timeout = _get_positive_number("NVIDIA_MSA_READ_TIMEOUT", 900.0, float)

    response = None
    last_error = "unknown error"
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                url=url,
                json={
                    "sequence": sequence,
                    "output_alignment_formats": ["a3m"],
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "NVCF-POLL-SECONDS": str(poll_seconds),
                },
                timeout=(connect_timeout, read_timeout),
            )
            if response.status_code == 200:
                break

            last_error = f"HTTP {response.status_code}: {response.text}"
            retryable = response.status_code in RETRYABLE_MSA_STATUS_CODES
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            retryable = True

        if not retryable or attempt == max_attempts:
            raise RuntimeError(
                f"Failed to make MSA for enzyme index {enzyme_id} after "
                f"{attempt} attempt(s): {last_error}"
            )

        delay = backoff_seconds * (2 ** (attempt - 1))
        tqdm.write(
            f"MSA request for enzyme index {enzyme_id} failed on attempt "
            f"{attempt}/{max_attempts} ({last_error}); retrying in {delay:g}s"
        )
        time.sleep(delay)

    if response is None:
        raise RuntimeError(f"Failed to make MSA for enzyme index {enzyme_id}")

    for database, formats in response.json().get("alignments", {}).items():
        for alignment in formats.values():
            if database == "colabfold" and alignment.get("format") == "a3m":
                temporary_path = output_path.with_suffix(".a3m.tmp")
                temporary_path.write_text(alignment["alignment"])
                temporary_path.replace(output_path)
                return output_path.resolve()

    raise RuntimeError(
        f"No ColabFold A3M alignment returned for enzyme index {enzyme_id}"
    )


def make_boltz_yaml(
    sequence: str,
    ligand: str,
    msa_path: Path,
    output_path: Path,
) -> None:
    """Write one protein/ligand input in the Boltz v1 YAML format."""
    output = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {sequence}\n"
        f"      msa: {msa_path.as_posix()}\n"
        "  - ligand:\n"
        "      id: B\n"
        f"      smiles: {json.dumps(ligand)}\n"
    )
    output_path.write_text(output)


def prepare_test_input(
    input_csv: Path,
    output_dir: Path,
    api_key: str,
    label_column: str = "Activity",
    default_label: float = 0.0,
) -> None:
    """Generate inputs and report sequences whose MSA generation failed."""
    df = pd.read_csv(input_csv)
    required_columns = {"SEQ", "SUBSTRATES"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing_columns)}"
        )
    if df.empty:
        raise ValueError("Input CSV contains no rows")
    if df[["SEQ", "SUBSTRATES"]].isna().any().any():
        raise ValueError("SEQ and SUBSTRATES must not contain missing values")

    # Convert to strings without changing or canonicalizing their contents.
    sequences = df["SEQ"].astype(str)
    substrates = df["SUBSTRATES"].astype(str)
    if sequences.str.strip().eq("").any() or substrates.str.strip().eq("").any():
        raise ValueError("SEQ and SUBSTRATES must not contain empty values")

    output_dir.mkdir(parents=True, exist_ok=True)
    msa_dir = output_dir / "msa"
    boltz_input_dir = output_dir / "boltz_inputs"
    msa_dir.mkdir(parents=True, exist_ok=True)
    boltz_input_dir.mkdir(parents=True, exist_ok=True)

    # Preserve first-occurrence ordering so IDs are stable and easy to inspect.
    unique_sequences = list(dict.fromkeys(sequences))
    unique_substrates = list(dict.fromkeys(substrates))
    sequence_ids = {sequence: index for index, sequence in enumerate(unique_sequences)}
    substrate_ids = {
        substrate: index for index, substrate in enumerate(unique_substrates)
    }

    reusable_msa_paths = {
        enzyme_id: path
        for enzyme_id, sequence in enumerate(unique_sequences)
        if (path := reusable_a3m_path(sequence, msa_dir, enzyme_id)) is not None
    }
    missing_msa_count = len(unique_sequences) - len(reusable_msa_paths)
    if missing_msa_count and not api_key:
        raise ValueError(
            f"Found {len(reusable_msa_paths)} reusable A3M file(s), but "
            f"{missing_msa_count} sequence(s) still need an MSA. Provide "
            "--api-key or set NVIDIA_API_KEY."
        )

    msa_paths = {}
    failed_sequences = []
    reused_msa_count = 0
    generated_msa_count = 0
    for enzyme_id, sequence in tqdm(
        enumerate(unique_sequences),
        total=len(unique_sequences),
        desc="Generating Boltz MSA",
    ):
        try:
            existing_path = reusable_msa_paths.get(enzyme_id)
            if existing_path is not None:
                msa_paths[enzyme_id] = existing_path
                reused_msa_count += 1
            else:
                msa_paths[enzyme_id] = make_msa(
                    sequence, msa_dir, enzyme_id, api_key
                )
                generated_msa_count += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            failed_sequences.append(
                {
                    "Enzyme Index": enzyme_id,
                    "Protein sequence": sequence,
                    "Error": error,
                }
            )
            tqdm.write(
                f"Skipping enzyme index {enzyme_id} after MSA generation "
                f"failed: {error}"
            )

    print(
        f"MSA checkpoint summary: reused {reused_msa_count} existing file(s), "
        f"generated {generated_msa_count} new file(s), and "
        f"failed {len(failed_sequences)} sequence(s)"
    )

    failed_sequences_path = output_dir / "failed_sequences.csv"
    pd.DataFrame(
        failed_sequences, columns=FAILED_SEQUENCE_COLUMNS
    ).to_csv(failed_sequences_path, index=False)

    if label_column in df.columns:
        labels = df[label_column]
    else:
        print(
            f"Column {label_column!r} not found; using default label "
            f"{default_label} for every row"
        )
        labels = pd.Series(default_label, index=df.index)

    data_rows = []
    skipped_pairs = 0
    for sequence, substrate, label in tqdm(
        zip(sequences, substrates, labels),
        total=len(df),
        desc="Generating Boltz YAML",
    ):
        enzyme_id = sequence_ids[sequence]
        if enzyme_id not in msa_paths:
            skipped_pairs += 1
            continue

        substrate_id = substrate_ids[substrate]
        dock_index = len(data_rows)
        make_boltz_yaml(
            sequence,
            substrate,
            msa_paths[enzyme_id],
            boltz_input_dir / f"{dock_index}.yaml",
        )
        data_rows.append(
            {
                "Substrate Index": substrate_id,
                "Enzyme Index": enzyme_id,
                "Label": label,
                "Dock Index": dock_index,
            }
        )

    pd.DataFrame(data_rows, columns=DATA_COLUMNS).to_csv(
        output_dir / "data.csv", index=False
    )
    pd.DataFrame({"Protein sequence": unique_sequences}).to_csv(
        output_dir / "Enzymes.csv", index=False
    )
    pd.DataFrame({"Substrate_SMILES": unique_substrates}).to_csv(
        output_dir / "Substrates.csv", index=False
    )

    print(
        f"Prepared {len(data_rows)} of {len(df)} pairs, "
        f"skipped {skipped_pairs} pair(s) from {len(failed_sequences)} "
        f"enzyme(s) whose MSA failed, and wrote the failed sequence report "
        f"to {failed_sequences_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a SEQ/SUBSTRATES CSV for Boltz and EZSpecificity"
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: the input CSV directory)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("NVIDIA_API_KEY"),
        help="NVIDIA API key (default: NVIDIA_API_KEY environment variable)",
    )
    parser.add_argument("--label-column", default="Activity")
    parser.add_argument("--default-label", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise ValueError(f"Input CSV does not exist: {args.input_csv}")
    output_dir = args.output_dir or args.input_csv.parent
    prepare_test_input(
        args.input_csv,
        output_dir,
        args.api_key,
        label_column=args.label_column,
        default_label=args.default_label,
    )


if __name__ == "__main__":
    main()
