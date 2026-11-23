import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import make_screening_inputs


class ProcessedOutputDetectionTest(unittest.TestCase):
    def make_screening_set(self, root: Path, complete: bool = True) -> Path:
        screening_set = root / "example"
        screening_set.mkdir()
        (screening_set / "example_binary.csv").write_text(
            "SEQ,SUBSTRATES,Activity\nACDE,CC,1\n"
        )

        if complete:
            for filename in make_screening_inputs.PROCESSED_OUTPUT_FILES:
                (screening_set / filename).write_text("output\n")
            msa_dir = screening_set / "msa"
            msa_dir.mkdir()
            (msa_dir / "0.a3m").write_text(">query\nACDE\n")
            boltz_dir = screening_set / "boltz_inputs"
            boltz_dir.mkdir()
            (boltz_dir / "0.yaml").write_text("version: 1\n")

        return screening_set

    def test_complete_outputs_are_skipped_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            screening_set = self.make_screening_set(Path(tmpdir))

            self.assertTrue(
                make_screening_inputs.has_processed_outputs(str(screening_set))
            )
            with mock.patch.object(
                make_screening_inputs, "process_screening_file"
            ) as process_file:
                make_screening_inputs.process_screening_sets(
                    tmpdir, set(), set(), "unused"
                )

            process_file.assert_not_called()

    def test_force_reprocesses_complete_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.make_screening_set(Path(tmpdir))

            with mock.patch.object(
                make_screening_inputs,
                "process_screening_file",
                return_value=None,
            ) as process_file:
                make_screening_inputs.process_screening_sets(
                    tmpdir, set(), set(), "unused", force=True
                )

            process_file.assert_called_once()

    def test_partial_outputs_are_not_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            screening_set = self.make_screening_set(Path(tmpdir), complete=False)
            (screening_set / "data.csv").write_text("partial\n")

            self.assertFalse(
                make_screening_inputs.has_processed_outputs(str(screening_set))
            )
            with mock.patch.object(
                make_screening_inputs,
                "process_screening_file",
                return_value=None,
            ) as process_file:
                make_screening_inputs.process_screening_sets(
                    tmpdir, set(), set(), "unused"
                )

            process_file.assert_called_once()


class MakeMsaCheckpointTest(unittest.TestCase):
    @mock.patch.object(make_screening_inputs.requests, "post")
    def test_existing_nonempty_a3m_skips_api_request(self, post):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "3.a3m"
            output_path.write_text(">query\nACDE\n")

            result = make_screening_inputs.make_msa(
                "ACDE", tmpdir, 3, "unused"
            )

        self.assertEqual(Path(result), output_path.resolve())
        post.assert_not_called()

if __name__ == "__main__":
    unittest.main()
