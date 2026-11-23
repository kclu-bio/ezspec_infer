#!/usr/bin/env python3
"""Reconstruct UniProt/UniParc accession-to-sequence maps for screening data.

The source datasets intentionally store identifiers and protein sequences in
different files:

* CatNIP identifiers are in the raw enzyme workbook, while sequences are in
  the processed activity table.
* Vergne nitrilase identifiers are in a raw CSV, while sequences are in the
  processed activity table.

This script joins them without querying an external database.  It also handles
two source-data details:

* Two adjacent CatNIP accessions resolve to the same sequence, so their
  contiguous processed blocks appear as one sequence run.
* The Vergne processing script replaced every field in activity-class-2 rows
  with ``1``.  The remaining rows in each 25-row enzyme block still retain the
  correct sequence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile


SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RELATIONSHIP_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_RELATIONSHIP_NS = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
S = f"{{{SPREADSHEET_NS}}}"
R = f"{{{RELATIONSHIP_NS}}}"
PR = f"{{{PACKAGE_RELATIONSHIP_NS}}}"
TRUTHY = {"t", "true", "1", "yes", "y", "x"}
CATNIP_MATRIX_SHEETS = [
    "Oxidation",
    "Desaturation",
    "Demethylation",
    "Rearrangement",
]


def _column_index(cell_reference: str) -> int:
    match = re.match(r"[A-Z]+", cell_reference)
    if match is None:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference}")
    result = 0
    for character in match.group():
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _worksheet_path(target: str) -> str:
    target = target.lstrip("/")
    if target.startswith("xl/"):
        return target
    return str(PurePosixPath("xl") / target)


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[list[str]]:
    """Read cell values from one XLSX worksheet using only the standard library."""

    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        relationship_targets = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in relationships.findall(f"{PR}Relationship")
        }

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(text.text or "" for text in item.iter(f"{S}t"))
                for item in shared_root.findall(f"{S}si")
            ]

        sheets = workbook.find(f"{S}sheets")
        if sheets is None:
            raise ValueError(f"No worksheets found in {path}")
        sheet = next(
            (item for item in sheets if item.attrib.get("name") == sheet_name),
            None,
        )
        if sheet is None:
            raise ValueError(f"Worksheet {sheet_name!r} not found in {path}")

        relationship_id = sheet.attrib[f"{R}id"]
        worksheet = ET.fromstring(
            archive.read(_worksheet_path(relationship_targets[relationship_id]))
        )

        output: list[list[str]] = []
        for row in worksheet.iter(f"{S}row"):
            values: dict[int, str] = {}
            for cell in row.findall(f"{S}c"):
                value_node = cell.find(f"{S}v")
                value = "" if value_node is None else value_node.text or ""
                cell_type = cell.attrib.get("t")
                if cell_type == "s" and value:
                    value = shared_strings[int(value)]
                elif cell_type == "inlineStr":
                    value = "".join(
                        text.text or "" for text in cell.iter(f"{S}t")
                    )
                values[_column_index(cell.attrib["r"])] = value
            if values:
                output_row = [""] * (max(values) + 1)
                for index, value in values.items():
                    output_row[index] = value
                output.append(output_row)
        return output


def rows_as_dicts(rows: list[list[str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    header = rows[0]
    output = []
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        output.append(dict(zip(header, padded)))
    return output


def truthy(value: object) -> bool:
    return str(value).strip().lower() in TRUTHY


def integer_id(value: object) -> int | None:
    text = re.sub(r"\.0$", "", str(value).strip())
    return int(text) if text.isdigit() else None


def clean_accession(value: object) -> str:
    return re.sub(r"\s+", "", str(value).strip())


def clean_sequence(value: object) -> str:
    return re.sub(r"\s+", "", str(value).upper())


def is_protein_sequence(value: object) -> bool:
    sequence = clean_sequence(value)
    return len(sequence) >= 20 and bool(re.fullmatch(r"[A-Z*]+", sequence))


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def accession_type(accession: str) -> str:
    return "UniParc" if accession.upper().startswith("UPI") else "UniProtKB"


def contiguous_sequence_runs(processed_path: Path) -> list[tuple[str, int]]:
    runs: list[list[object]] = []
    with processed_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = clean_sequence(row["SEQ"])
            if not runs or sequence != runs[-1][0]:
                runs.append([sequence, 0])
            runs[-1][1] = int(runs[-1][1]) + 1
    return [(str(sequence), int(count)) for sequence, count in runs]


def catnip_expected_row_counts(
    raw_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    """Return ordered CatNIP accession candidates and their expected row counts."""

    enzyme_file = raw_dir / "SI_EnzymeTable_10012024.xlsx"
    substrate_file = raw_dir / "SI_SubstrateTable_10242024.xlsx"
    reaction_file = raw_dir / "SI_ReactionTable_10012024.xlsx"

    enzymes = rows_as_dicts(read_xlsx_sheet(enzyme_file, "Enzyme Table"))
    substrates = rows_as_dicts(read_xlsx_sheet(substrate_file, "Summary"))
    reactions = rows_as_dicts(read_xlsx_sheet(reaction_file, "Reaction Table"))

    reaction_flags: dict[tuple[int, int], list[bool]] = defaultdict(
        lambda: [False, False]
    )
    for row in reactions:
        enzyme_id = integer_id(row.get("Enzyme", ""))
        substrate_id = integer_id(row.get("Substrate", ""))
        if enzyme_id is None or substrate_id is None:
            continue
        flags = reaction_flags[enzyme_id, substrate_id]
        flags[0] = flags[0] or truthy(row.get("Observed", ""))
        flags[1] = flags[1] or truthy(row.get("Literature", ""))

    screened_enzyme_ids = {
        enzyme_id
        for (enzyme_id, _), (observed, literature) in reaction_flags.items()
        if observed and not literature
    }

    matrix_scores: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for sheet_name in CATNIP_MATRIX_SHEETS:
        rows = read_xlsx_sheet(reaction_file, sheet_name)
        header = rows[0]
        substrate_columns: list[tuple[int, int]] = []
        # This reproduces the bounds in reformat_catnip.py.
        for column_index in range(2, max(2, len(header) - 2)):
            substrate_id = integer_id(header[column_index])
            if substrate_id is not None:
                substrate_columns.append((column_index, substrate_id))
        for row in rows[1:]:
            enzyme_id = integer_id(row[0] if row else "")
            if enzyme_id is None:
                continue
            for column_index, substrate_id in substrate_columns:
                if column_index >= len(row):
                    continue
                try:
                    score = float(str(row[column_index]).strip())
                except ValueError:
                    continue
                current = matrix_scores[enzyme_id, substrate_id].get(sheet_name)
                if current is None or score > current:
                    matrix_scores[enzyme_id, substrate_id][sheet_name] = score

    screened_substrates: list[tuple[int, bool]] = []
    for row in substrates:
        substrate_id = integer_id(row.get("Substrate ID", ""))
        if (
            substrate_id is not None
            and truthy(row.get("Screened Substrate?", ""))
            and str(row.get("SMILES", "")).strip()
        ):
            screened_substrates.append(
                (substrate_id, truthy(row.get("Visible by Mz", "")))
            )

    candidates: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    for row in enzymes:
        enzyme_id = integer_id(row.get("Enzyme ID", ""))
        if enzyme_id not in screened_enzyme_ids:
            continue
        accession = clean_accession(row.get("Accession ID", ""))
        if not accession:
            unresolved.append(
                {
                    "dataset": "catnip",
                    "enzyme_id": str(enzyme_id),
                    "uniprot_id": "",
                    "sequence": "",
                    "reason": "raw enzyme table has no accession ID",
                }
            )
            continue

        expected_rows = 0
        for substrate_id, visible in screened_substrates:
            scores = matrix_scores.get((enzyme_id, substrate_id), {})
            matrix_available = bool(scores)
            matrix_positive = any(score > 0 for score in scores.values())
            observed, literature = reaction_flags[enzyme_id, substrate_id]
            positive = matrix_positive or (observed and not literature)
            negative = (
                not matrix_available
                and visible
                and truthy(row.get("Expressed?", ""))
                and not positive
            )
            expected_rows += int(positive or negative)

        candidates.append(
            {
                "dataset": "catnip",
                "enzyme_id": str(enzyme_id),
                "uniprot_id": accession,
                "enzyme_name": str(row.get("Enzyme Name", "")).strip(),
                "expected_rows": expected_rows,
            }
        )

    stats = {
        "screened_raw_enzymes": len(screened_enzyme_ids),
        "screened_substrates": len(screened_substrates),
        "accession_candidates": len(candidates),
        "missing_accession": len(unresolved),
    }
    return candidates, unresolved, stats


def align_catnip_candidates(
    candidates: list[dict[str, object]], runs: list[tuple[str, int]]
) -> list[dict[str, object]]:
    """Partition ordered candidates into processed sequence runs by row count."""

    # A state stores (number of solutions capped at 2, one representative path).
    states: dict[
        tuple[int, int], tuple[int, list[tuple[int, int, int]]]
    ] = {(0, 0): (1, [])}
    for candidate_start in range(len(candidates) + 1):
        for run_index in range(len(runs) + 1):
            state = states.get((candidate_start, run_index))
            if state is None or run_index == len(runs):
                continue
            solution_count, path = state
            target_count = runs[run_index][1]
            accumulated = 0
            for candidate_end in range(candidate_start, len(candidates)):
                accumulated += int(candidates[candidate_end]["expected_rows"])
                if accumulated > target_count:
                    break
                if accumulated != target_count:
                    continue
                key = (candidate_end + 1, run_index + 1)
                new_path = path + [
                    (candidate_start, candidate_end + 1, run_index)
                ]
                old_count, old_path = states.get(key, (0, new_path))
                states[key] = (
                    min(2, old_count + solution_count),
                    old_path if old_count else new_path,
                )

    final = states.get((len(candidates), len(runs)))
    if final is None:
        raise ValueError(
            "CatNIP raw-derived row counts cannot be aligned to processed sequence runs"
        )
    solution_count, path = final
    if solution_count != 1:
        raise ValueError(
            f"CatNIP mapping is ambiguous: found at least {solution_count} alignments"
        )

    mapping: list[dict[str, object]] = []
    for candidate_start, candidate_end, run_index in path:
        sequence = runs[run_index][0]
        if not is_protein_sequence(sequence):
            raise ValueError(
                f"Invalid CatNIP protein sequence in run {run_index}: {sequence[:20]!r}"
            )
        for candidate in candidates[candidate_start:candidate_end]:
            mapping.append(
                {
                    **candidate,
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "sequence_sha256": sequence_digest(sequence),
                    "accession_type": accession_type(
                        str(candidate["uniprot_id"])
                    ),
                    "mapping_method": (
                        "raw enzyme order + raw-derived activity-row counts "
                        "+ processed contiguous sequence runs"
                    ),
                }
            )
    return mapping


def build_catnip_mapping(
    source_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    raw_dir = source_root / "raw" / "catnip"
    processed_path = source_root / "processed" / "catnip_screen_binary.csv"
    candidates, unresolved, stats = catnip_expected_row_counts(raw_dir)
    runs = contiguous_sequence_runs(processed_path)
    mapping = align_catnip_candidates(candidates, runs)
    stats.update(
        {
            "mapped_accessions": len(mapping),
            "unique_sequences": len({row["sequence"] for row in mapping}),
            "processed_sequence_runs": len(runs),
            "processed_rows": sum(count for _, count in runs),
        }
    )
    return mapping, unresolved, stats


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_vergne_mapping(
    source_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    raw_dir = source_root / "raw" / "vergne_nitrilase"
    processed_dir = source_root / "processed"
    enzymes = read_csv_rows(raw_dir / "vergne_nitrilase_screen_enzymes.csv")
    raw_activity = read_csv_rows(
        raw_dir / "vergne_nitrilase_screen_binary.csv"
    )
    processed = read_csv_rows(
        processed_dir / "vergne_nitrilase_screen_binary.csv"
    )

    substrate_count = len(
        {row["Substrate ID"].strip() for row in raw_activity}
    )
    resolved_enzymes = [
        row for row in enzymes if clean_accession(row.get("UniProt ID", ""))
    ]
    if len(processed) != len(resolved_enzymes) * substrate_count:
        raise ValueError(
            "Vergne processed rows do not form one fixed-size substrate block "
            f"per resolved enzyme: {len(processed)} != "
            f"{len(resolved_enzymes)} * {substrate_count}"
        )

    mapping: list[dict[str, object]] = []
    corrupted_rows = 0
    for index, enzyme in enumerate(resolved_enzymes):
        block = processed[
            index * substrate_count : (index + 1) * substrate_count
        ]
        sequences = {
            clean_sequence(row.get("SEQ", ""))
            for row in block
            if is_protein_sequence(row.get("SEQ", ""))
        }
        if len(sequences) != 1:
            raise ValueError(
                f"Expected one recoverable sequence for Vergne enzyme "
                f"{enzyme.get('Nitrilase ID')}; found {len(sequences)}"
            )
        sequence = sequences.pop()
        block_corrupted = sum(
            1
            for row in block
            if row.get("SEQ") == row.get("SUBSTRATES") == row.get("Activity") == "1"
        )
        corrupted_rows += block_corrupted
        accession = clean_accession(enzyme["UniProt ID"])
        mapping.append(
            {
                "dataset": "vergne_nitrilase",
                "enzyme_id": str(enzyme["Nitrilase ID"]).strip(),
                "uniprot_id": accession,
                "enzyme_name": "",
                "expected_rows": substrate_count,
                "sequence": sequence,
                "sequence_length": len(sequence),
                "sequence_sha256": sequence_digest(sequence),
                "accession_type": accession_type(accession),
                "mapping_method": (
                    f"raw enzyme order + processed {substrate_count}-row blocks; "
                    "ignored rows corrupted to 1,1,1"
                ),
            }
        )

    unresolved: list[dict[str, object]] = []
    supplemental = read_csv_rows(processed_dir / "NIT161_nitrilase.csv")
    supplemental_by_id: dict[str, str] = {}
    for row in supplemental:
        enzyme_id = str(row.get("Nitrilase ID", "")).strip()
        sequence = clean_sequence(row.get("SEQ", ""))
        if enzyme_id and is_protein_sequence(sequence):
            supplemental_by_id[enzyme_id] = sequence
    for enzyme in enzymes:
        if clean_accession(enzyme.get("UniProt ID", "")):
            continue
        enzyme_id = str(enzyme.get("Nitrilase ID", "")).strip()
        sequence = supplemental_by_id.get(enzyme_id, "")
        unresolved.append(
            {
                "dataset": "vergne_nitrilase",
                "enzyme_id": enzyme_id,
                "uniprot_id": "",
                "sequence": sequence,
                "reason": (
                    "raw source has no UniProt ID; sequence is available "
                    "but cannot form a UniProt-to-sequence mapping"
                ),
            }
        )

    stats = {
        "raw_enzymes": len(enzymes),
        "mapped_accessions": len(mapping),
        "unique_sequences": len({row["sequence"] for row in mapping}),
        "missing_accession": len(unresolved),
        "substrates_per_enzyme": substrate_count,
        "processed_rows": len(processed),
        "corrupted_processed_rows_ignored": corrupted_rows,
    }
    return mapping, unresolved, stats


MAPPING_COLUMNS = [
    "uniprot_id",
    "sequence",
    "enzyme_id",
    "dataset",
    "accession_type",
    "sequence_length",
    "sequence_sha256",
    "enzyme_name",
    "expected_rows",
    "mapping_method",
]
UNRESOLVED_COLUMNS = [
    "dataset",
    "enzyme_id",
    "uniprot_id",
    "sequence",
    "reason",
]


def write_csv(
    path: Path, rows: Iterable[dict[str, object]], columns: list[str]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/home/kechenlu/kechenlu_link/enzymelm/data/additional_family/"
            "enzyme-datasets/data"
        ),
        help="Directory containing raw/ and processed/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_dir/uniprot_sequence_mappings"),
    )
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=Path("data_dir/screen"),
        help=(
            "Optional EZSpec screen directory containing "
            "catnip/Enzymes.csv and vergne_nitrilase/Enzymes.csv"
        ),
    )
    return parser.parse_args()


def ezspec_subset(
    mapping: list[dict[str, object]],
    unresolved: list[dict[str, object]],
    enzymes_path: Path,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Filter a full mapping to sequences present in an EZSpec Enzymes.csv."""

    if not enzymes_path.exists():
        return [], {}
    enzyme_rows = read_csv_rows(enzymes_path)
    screen_sequences = {
        clean_sequence(row.get("Protein sequence", ""))
        for row in enzyme_rows
        if is_protein_sequence(row.get("Protein sequence", ""))
    }
    subset = [
        row for row in mapping if str(row["sequence"]) in screen_sequences
    ]
    mapped_sequences = {str(row["sequence"]) for row in subset}
    unresolved_sequences = {
        str(row["sequence"])
        for row in unresolved
        if row.get("sequence") and str(row["sequence"]) in screen_sequences
    }
    unexplained_sequences = (
        screen_sequences - mapped_sequences - unresolved_sequences
    )
    if unexplained_sequences:
        raise ValueError(
            f"{enzymes_path} contains {len(unexplained_sequences)} sequences "
            "that are absent from both mapped and unresolved source records"
        )
    stats = {
        "input_sequences": len(screen_sequences),
        "mapping_rows": len(subset),
        "mapped_sequences": len(mapped_sequences),
        "unresolved_sequences": len(unresolved_sequences),
    }
    return subset, stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    catnip, catnip_unresolved, catnip_stats = build_catnip_mapping(
        args.source_root
    )
    vergne, vergne_unresolved, vergne_stats = build_vergne_mapping(
        args.source_root
    )
    combined = catnip + vergne
    catnip_subset, catnip_subset_stats = ezspec_subset(
        catnip,
        catnip_unresolved,
        args.screen_root / "catnip" / "Enzymes.csv",
    )
    vergne_subset, vergne_subset_stats = ezspec_subset(
        vergne,
        vergne_unresolved,
        args.screen_root / "vergne_nitrilase" / "Enzymes.csv",
    )

    write_csv(
        args.output_dir / "catnip_uniprot_sequence.csv",
        catnip,
        MAPPING_COLUMNS,
    )
    write_csv(
        args.output_dir / "vergne_nitrilase_uniprot_sequence.csv",
        vergne,
        MAPPING_COLUMNS,
    )
    write_csv(
        args.output_dir / "combined_uniprot_sequence.csv",
        combined,
        MAPPING_COLUMNS,
    )
    if catnip_subset_stats:
        write_csv(
            args.output_dir / "catnip_ezspec_subset_uniprot_sequence.csv",
            catnip_subset,
            MAPPING_COLUMNS,
        )
    if vergne_subset_stats:
        write_csv(
            args.output_dir
            / "vergne_nitrilase_ezspec_subset_uniprot_sequence.csv",
            vergne_subset,
            MAPPING_COLUMNS,
        )
    write_csv(
        args.output_dir / "catnip_unresolved.csv",
        catnip_unresolved,
        UNRESOLVED_COLUMNS,
    )
    write_csv(
        args.output_dir / "vergne_nitrilase_unresolved.csv",
        vergne_unresolved,
        UNRESOLVED_COLUMNS,
    )

    summary = {
        "source_root": str(args.source_root),
        "catnip": catnip_stats,
        "vergne_nitrilase": vergne_stats,
        "combined": {
            "mapped_rows": len(combined),
            "unique_accessions": len(
                {str(row["uniprot_id"]) for row in combined}
            ),
            "unique_sequences": len(
                {str(row["sequence"]) for row in combined}
            ),
        },
        "ezspec_subset": {
            "catnip": catnip_subset_stats,
            "vergne_nitrilase": vergne_subset_stats,
        },
    }
    (args.output_dir / "mapping_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
