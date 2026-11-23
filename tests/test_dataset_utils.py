import unittest

from rdkit import Chem

from Datasets.utils import _get_smiles_atom_output_order, parse_smile


class SmilesAtomOutputOrderTest(unittest.TestCase):
    def test_accepts_current_rdkit_format(self):
        mol = Chem.MolFromSmiles("CCC")
        mol.SetProp("_smilesAtomOutputOrder", "[0,1,2]")

        self.assertEqual(_get_smiles_atom_output_order(mol), [0, 1, 2])

    def test_accepts_legacy_trailing_comma_format(self):
        mol = Chem.MolFromSmiles("CCC")
        mol.SetProp("_smilesAtomOutputOrder", "[0,1,2,]")

        self.assertEqual(_get_smiles_atom_output_order(mol), [0, 1, 2])

    def test_preserves_multi_digit_atom_indices(self):
        mol = Chem.MolFromSmiles("CCCCCCCCCCC")
        mol.SetProp(
            "_smilesAtomOutputOrder",
            "[0,1,2,3,4,5,6,7,8,9,10]",
        )

        self.assertEqual(
            _get_smiles_atom_output_order(mol),
            list(range(11)),
        )

    def test_rejects_incomplete_atom_order(self):
        mol = Chem.MolFromSmiles("CCC")
        mol.SetProp("_smilesAtomOutputOrder", "[0,1]")

        with self.assertRaisesRegex(ValueError, "complete atom permutation"):
            _get_smiles_atom_output_order(mol)

    def test_parse_smile_handles_reported_substrate(self):
        data, _, _, mol = parse_smile("CC/C=C/C#N")

        self.assertEqual(data["element"].shape[0], mol.GetNumAtoms())
        self.assertEqual(mol.GetNumAtoms(), 6)


if __name__ == "__main__":
    unittest.main()
