import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import pandas as pd
import requests

from src.make_test_input import (
    DATA_COLUMNS,
    make_msa,
    prepare_test_input,
    reusable_a3m_path,
)


class MakeMsaTest(unittest.TestCase):
    @patch("src.make_test_input.requests.post")
    def test_reuses_matching_existing_alignment_without_api_request(self, post):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir, "4.a3m")
            output_path.write_text(">Query\nSEQUENCE\n>hit\nSEQUEN-E\n")

            result = make_msa("SEQUENCE", Path(tmpdir), 4, "unused")

        self.assertEqual(result, output_path.resolve())
        post.assert_not_called()

    def test_does_not_reuse_alignment_for_a_different_sequence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir, "4.a3m")
            output_path.write_text(">Query\nOTHER_SEQUENCE\n")

            self.assertIsNone(reusable_a3m_path("SEQUENCE", Path(tmpdir), 4))

    @patch.dict(
        "os.environ",
        {
            "NVIDIA_MSA_MAX_ATTEMPTS": "3",
            "NVIDIA_MSA_BACKOFF_SECONDS": "0.25",
        },
    )
    @patch("src.make_test_input.time.sleep")
    @patch("src.make_test_input.requests.post")
    def test_retries_504_and_writes_alignment_atomically(self, post, sleep):
        unavailable = Mock(status_code=504, text="gateway timeout")
        success = Mock(status_code=200)
        success.json.return_value = {
            "alignments": {
                "colabfold": {
                    "a3m": {"format": "a3m", "alignment": ">query\nSEQUENCE\n"}
                }
            }
        }
        post.side_effect = [unavailable, success]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = make_msa("SEQUENCE", Path(tmpdir), 51, "unused")

            self.assertEqual(output_path.read_text(), ">query\nSEQUENCE\n")
            self.assertFalse(Path(tmpdir, "51.a3m.tmp").exists())

        self.assertEqual(post.call_count, 2)
        self.assertEqual(
            post.call_args.kwargs["headers"]["NVCF-POLL-SECONDS"], "600"
        )
        self.assertEqual(post.call_args.kwargs["timeout"], (30.0, 900.0))
        sleep.assert_called_once_with(0.25)

    @patch.dict("os.environ", {"NVIDIA_MSA_MAX_ATTEMPTS": "3"})
    @patch("src.make_test_input.time.sleep")
    @patch("src.make_test_input.requests.post")
    def test_does_not_retry_non_transient_http_error(self, post, sleep):
        post.return_value = Mock(status_code=401, text="unauthorized")

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "after 1 attempt"):
                make_msa("SEQUENCE", Path(tmpdir), 7, "unused")

        post.assert_called_once()
        sleep.assert_not_called()

    @patch.dict(
        "os.environ",
        {
            "NVIDIA_MSA_MAX_ATTEMPTS": "3",
            "NVIDIA_MSA_BACKOFF_SECONDS": "2",
        },
    )
    @patch("src.make_test_input.time.sleep")
    @patch("src.make_test_input.requests.post")
    def test_retries_connection_errors_with_exponential_backoff(self, post, sleep):
        post.side_effect = [
            requests.ConnectionError("network down"),
            requests.Timeout("read timeout"),
            Mock(status_code=503, text="unavailable"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempt"):
                make_msa("SEQUENCE", Path(tmpdir), 9, "unused")

        self.assertEqual(sleep.call_args_list, [call(2.0), call(4.0)])


class PrepareTestInputTest(unittest.TestCase):
    def test_missing_alignment_without_api_key_fails_before_processing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_csv = root / "input.csv"
            pd.DataFrame({"SEQ": ["SEQ"], "SUBSTRATES": ["CC"]}).to_csv(
                input_csv, index=False
            )

            with self.assertRaisesRegex(ValueError, "1 sequence.*still need an MSA"):
                prepare_test_input(input_csv, root / "output", "")

    def test_preserves_every_pair_and_raw_substrate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_csv = root / "input.csv"
            output_dir = root / "output"
            pd.DataFrame(
                {
                    "SEQ": ["SEQBBB", "SEQAAA", "SEQBBB"],
                    "SUBSTRATES": ["[Na+].CC", "C(C)O", "[Na+].CC"],
                    "Activity": [0, 0, 0],
                }
            ).to_csv(input_csv, index=False)

            msa_dir = output_dir / "msa"
            msa_dir.mkdir(parents=True)
            (msa_dir / "0.a3m").write_text(">query\nSEQBBB\n")
            (msa_dir / "1.a3m").write_text(">query\nSEQAAA\n")

            prepare_test_input(input_csv, output_dir, "unused")

            data = pd.read_csv(output_dir / "data.csv")
            self.assertEqual(len(data), 3)
            self.assertEqual(data["Label"].tolist(), [0, 0, 0])
            self.assertEqual(data["Dock Index"].tolist(), [0, 1, 2])
            self.assertEqual(
                pd.read_csv(output_dir / "Enzymes.csv")["Protein sequence"].tolist(),
                ["SEQBBB", "SEQAAA"],
            )
            self.assertEqual(
                pd.read_csv(output_dir / "Substrates.csv")[
                    "Substrate_SMILES"
                ].tolist(),
                ["[Na+].CC", "C(C)O"],
            )
            self.assertIn(
                'smiles: "C(C)O"',
                (output_dir / "boltz_inputs" / "1.yaml").read_text(),
            )
            self.assertTrue(pd.read_csv(output_dir / "failed_sequences.csv").empty)

    def test_uses_default_label_when_activity_is_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_csv = root / "input.csv"
            output_dir = root / "output"
            pd.DataFrame({"SEQ": ["SEQ"], "SUBSTRATES": ["CC"]}).to_csv(
                input_csv, index=False
            )

            msa_dir = output_dir / "msa"
            msa_dir.mkdir(parents=True)
            (msa_dir / "0.a3m").write_text(">query\nSEQ\n")

            prepare_test_input(input_csv, output_dir, "unused", default_label=-1.0)

            data = pd.read_csv(output_dir / "data.csv")
            self.assertEqual(data["Label"].tolist(), [-1.0])

    @patch("src.make_test_input.make_msa")
    def test_skips_failed_enzymes_and_collects_their_sequences(self, make_msa_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_csv = root / "input.csv"
            output_dir = root / "output"
            pd.DataFrame(
                {
                    "SEQ": ["FAILED", "WORKS", "FAILED", "ALSO_WORKS"],
                    "SUBSTRATES": ["C", "CC", "CCC", "CCCC"],
                    "Activity": [0, 1, 0, 1],
                }
            ).to_csv(input_csv, index=False)

            def make_msa_side_effect(sequence, output_dir, enzyme_id, api_key):
                if sequence == "FAILED":
                    raise RuntimeError(
                        "Failed to make MSA for enzyme index 0 after 8 attempt(s)"
                    )
                output_path = output_dir / f"{enzyme_id}.a3m"
                output_path.write_text(f">query\n{sequence}\n")
                return output_path.resolve()

            make_msa_mock.side_effect = make_msa_side_effect

            prepare_test_input(input_csv, output_dir, "unused")

            data = pd.read_csv(output_dir / "data.csv")
            self.assertEqual(data["Enzyme Index"].tolist(), [1, 2])
            self.assertEqual(data["Dock Index"].tolist(), [0, 1])
            self.assertEqual(data["Label"].tolist(), [1, 1])
            self.assertEqual(
                sorted(path.name for path in (output_dir / "boltz_inputs").iterdir()),
                ["0.yaml", "1.yaml"],
            )

            failures = pd.read_csv(output_dir / "failed_sequences.csv")
            self.assertEqual(failures["Enzyme Index"].tolist(), [0])
            self.assertEqual(failures["Protein sequence"].tolist(), ["FAILED"])
            self.assertIn("after 8 attempt(s)", failures.loc[0, "Error"])

    @patch("src.make_test_input.make_msa")
    def test_writes_readable_empty_data_csv_when_every_msa_fails(self, make_msa_mock):
        make_msa_mock.side_effect = RuntimeError("all attempts failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_csv = root / "input.csv"
            output_dir = root / "output"
            pd.DataFrame(
                {
                    "SEQ": ["FAILED"],
                    "SUBSTRATES": ["C"],
                    "Activity": [0],
                }
            ).to_csv(input_csv, index=False)

            prepare_test_input(input_csv, output_dir, "unused")

            data = pd.read_csv(output_dir / "data.csv")
            self.assertTrue(data.empty)
            self.assertEqual(data.columns.tolist(), DATA_COLUMNS)
            self.assertEqual(len(pd.read_csv(output_dir / "failed_sequences.csv")), 1)


if __name__ == "__main__":
    unittest.main()
