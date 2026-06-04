import copy
from itertools import chain
import pickle
import unittest

import pytest

from parmed import Atom, Bond, Structure, load_rosetta, read_PDB, save_rosetta
from parmed.rosetta import pose as rosetta_pose_module
from utils import get_fn
from tests.fake_rosetta_bridge import (
    FakeAtomType,
    FakePDBInfo,
    FakePose,
    FakeResidue,
    FakeResidueType,
    build_fake_import,
)

try:
    from openmm.app import PDBFile
except:
    PDBFile = None
try:
    from rosetta import init, pose_from_sequence
except ImportError:
    init = pose_from_sequence = None

@unittest.skipIf(init is None, "Cannot test load_rosetta module without PyRosetta.")
class TestRosetta(unittest.TestCase):
    """ Tests loading of a Rosetta pose object """

    def test_loaded_positions(self):
        """ Test that positions were properly loaded"""

        init()
        seq = 3*'A'
        pose = pose_from_sequence(seq)

        struct = load_rosetta(pose)

        posexyz = list(
            chain(*[[tuple(atom.xyz()) for atom in res.atoms()]
                    for res in [pose.residue(idx) for idx in range(1, len(seq)+1)]]))

        structxyz = [(atom.xx, atom.xy, atom.xz) for atom in struct.atoms]

        self.assertEqual(posexyz, structxyz)

    def test_load_struct(self):
        """ Test load_rosetta against read_PDB"""

        init()
        pose = pose_from_sequence(3*'A')

        struct = load_rosetta(pose)
        pdb = read_PDB(get_fn('ala_ala_ala.pdb'))

        self.assertEqual(len(struct.atoms), len(pdb.atoms))
        self.assertEqual(len(struct.bonds), len(pdb.bonds))
        self.assertEqual(len(struct.residues), len(pdb.residues))

    @unittest.skipIf(PDBFile is None, "Cannot compare topologies without OpenMM.")
    def test_loaded_topology(self):
        """ Test load_rosetta against OpenMM topology"""

        init()
        pose = pose_from_sequence(3*'A')

        struct = load_rosetta(pose)
        pdb = PDBFile(get_fn('ala_ala_ala.pdb'))

        self.assertEqual(len(list(struct.topology.atoms())),
                         len(list(pdb.topology.atoms())))

        self.assertEqual(len(list(struct.topology.bonds())),
                         len(list(pdb.topology.bonds())))

        self.assertEqual(len(list(struct.topology.residues())),
                         len(list(pdb.topology.residues())))


def _build_fake_residue_types():
    return [
        FakeResidueType(
            unique_name="ALA:NtermProteinFull",
            name3_value="ALA",
            name1_value="A",
            atom_order=("N", "CA", "C", "V1"),
            atom_types={
                "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
                "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
                "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
                "V1": FakeAtomType(element="EP", atom_type_name="VIRT", lj_radius=0.0, lj_wdepth=0.0, is_virtual=True),
            },
            charges={"N": -0.3, "CA": 0.1, "C": 0.5, "V1": 0.0},
        ),
        FakeResidueType(
            unique_name="ALA",
            name3_value="ALA",
            name1_value="A",
            atom_order=("N", "CA", "C", "V1"),
            atom_types={
                "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
                "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
                "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
                "V1": FakeAtomType(element="EP", atom_type_name="VIRT", lj_radius=0.0, lj_wdepth=0.0, is_virtual=True),
            },
            charges={"N": -0.3, "CA": 0.1, "C": 0.5, "V1": 0.0},
        ),
        FakeResidueType(
            unique_name="GLY:CtermProteinFull",
            name3_value="GLY",
            name1_value="G",
            atom_order=("N", "CA", "C", "1HA", "2HA"),
            atom_types={
                "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
                "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
                "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
                "1HA": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
                "2HA": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            },
            charges={"N": -0.25, "CA": 0.05, "C": 0.45},
            canonical_alias_map={"1HA": "HA2", "2HA": "HA3"},
        ),
        FakeResidueType(
            unique_name="ACE",
            name3_value="ACE",
            name1_value="X",
            atom_order=("CH3", "C"),
            atom_types={
                "CH3": FakeAtomType(element="C", atom_type_name="CH3", lj_radius=1.9, lj_wdepth=0.08),
                "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
            },
            polymer=False,
        ),
        FakeResidueType(
            unique_name="NME",
            name3_value="NME",
            name1_value="X",
            atom_order=("N", "CH3"),
            atom_types={
                "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
                "CH3": FakeAtomType(element="C", atom_type_name="CH3", lj_radius=1.9, lj_wdepth=0.08),
            },
            polymer=False,
        ),
        FakeResidueType(
            unique_name="CYS:NtermProteinFull",
            name3_value="CYS",
            name1_value="C",
            atom_order=("N", "CA", "C", "SG"),
            atom_types={
                "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
                "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
                "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
                "SG": FakeAtomType(element="S", atom_type_name="Sthio", lj_radius=2.0, lj_wdepth=0.25),
            },
        ),
        FakeResidueType(
            unique_name="CYS",
            name3_value="CYS",
            name1_value="C",
            atom_order=("N", "CA", "C", "SG"),
            atom_types={
                "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
                "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
                "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
                "SG": FakeAtomType(element="S", atom_type_name="Sthio", lj_radius=2.0, lj_wdepth=0.25),
            },
        ),
        FakeResidueType(
            unique_name="CYS:CtermProteinFull",
            name3_value="CYS",
            name1_value="C",
            atom_order=("N", "CA", "C", "SG"),
            atom_types={
                "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
                "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
                "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
                "SG": FakeAtomType(element="S", atom_type_name="Sthio", lj_radius=2.0, lj_wdepth=0.25),
            },
        ),
        FakeResidueType(
            unique_name="SEP",
            name3_value="SEP",
            name1_value="S",
            atom_order=("N", "CA", "C", "P"),
            atom_types={
                "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
                "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
                "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
                "P": FakeAtomType(element="P", atom_type_name="Pha", lj_radius=2.1, lj_wdepth=0.2),
            },
        ),
    ]


def _install_fake_pyrosetta(monkeypatch):
    fake_import = build_fake_import(_build_fake_residue_types())
    monkeypatch.setattr(rosetta_pose_module, "_import_pyrosetta", lambda: fake_import)
    return fake_import


def _make_atom(name, atomic_number, atom_type, xyz):
    atom = Atom(atomic_number=atomic_number, name=name, type=atom_type, charge=0.0)
    atom.xx, atom.xy, atom.xz = xyz
    return atom


def _build_structure(residue_specs, bonds, ter_indices=(), zero_hydrogen_atomic_numbers=False):
    structure = Structure()
    atom_index = 1
    atoms_by_key = {}
    atomic_numbers = {"N": 7, "C": 6, "O": 8, "S": 16, "P": 15, "H": 1, "V": 0}
    for residue_number, chain_id, residue_name, atoms in residue_specs:
        for atom_name, atom_type, xyz in atoms:
            element_key = next((char for char in atom_name if char.isalpha()), atom_name[0])
            atomic_number = atomic_numbers[element_key]
            if zero_hydrogen_atomic_numbers and atomic_number == 1:
                atomic_number = 0
            atom = _make_atom(atom_name, atomic_number, atom_type, xyz)
            atom.number = atom_index
            atom_index += 1
            structure.add_atom(atom, residue_name, residue_number, chain_id, "")
            atoms_by_key[(residue_number, chain_id, atom_name)] = atom
    for atom_key_a, atom_key_b in bonds:
        structure.bonds.append(Bond(atoms_by_key[atom_key_a], atoms_by_key[atom_key_b]))
    for residue_index in ter_indices:
        structure.residues[residue_index - 1].ter = True
    structure.unchange()
    return structure


def test_fake_pose_to_structure_preserves_metadata(monkeypatch):
    _install_fake_pyrosetta(monkeypatch)
    residue_types = _build_fake_residue_types()
    residue1 = FakeResidue(
        residue_types[0],
        coordinates={"N": (0.0, 0.0, 0.0), "CA": (1.0, 0.0, 0.0), "C": (2.0, 0.0, 0.0), "V1": (0.5, 0.5, 0.5)},
    )
    residue2 = FakeResidue(
        residue_types[2],
        coordinates={"N": (3.0, 0.0, 0.0), "CA": (4.0, 0.0, 0.0), "C": (5.0, 0.0, 0.0), "1HA": (4.0, 1.0, 0.0), "2HA": (4.0, -1.0, 0.0)},
    )
    pdb_info = FakePDBInfo(2)
    pdb_info.chain(1, "A")
    pdb_info.chain(2, "A")
    pdb_info.number(1, 10)
    pdb_info.number(2, 11)
    pdb_info.icode(2, "B")
    pdb_info.alt_loc(1, 1, "A")
    pdb_info.occupancy(1, 1, 0.75)
    pdb_info.bfactor(1, 1, 12.5)
    pose = FakePose(
        [residue1, residue2],
        chain_indices=[1, 1],
        bonds=[((1, 3), (2, 1))],
        pdb_info=pdb_info,
    )

    structure = load_rosetta(pose)

    assert len(structure.residues) == 2
    assert structure.residues[0].number == 10
    assert structure.residues[1].insertion_code == "B"
    assert structure.residues[1].ter is True
    assert structure.atoms[0].type == "Nbb"
    assert structure.atoms[0].charge == pytest.approx(-0.3)
    assert structure.atoms[0].occupancy == pytest.approx(0.75)
    assert structure.atoms[0].altloc == "A"
    assert structure.atoms[3].atomic_number == 0
    metadata = getattr(structure, rosetta_pose_module.ROSETTA_BRIDGE_METADATA_ATTR)
    assert metadata["residues"][0]["type_name"] == "ALA:NtermProteinFull"
    assert metadata["residues"][1]["type_name"] == "GLY:CtermProteinFull"


def test_fake_structure_to_pose_handles_caps_and_chain_breaks(monkeypatch):
    _install_fake_pyrosetta(monkeypatch)
    structure = _build_structure(
        [
            (1, "A", "ACE", [("CH3", "CH3", (0.0, 0.0, 0.0)), ("C", "CObb", (1.0, 0.0, 0.0))]),
            (2, "A", "ALA", [("N", "Nbb", (2.0, 0.0, 0.0)), ("CA", "CAbb", (3.0, 0.0, 0.0)), ("C", "CObb", (4.0, 0.0, 0.0)), ("V1", "VIRT", (3.0, 1.0, 0.0))]),
            (3, "A", "NME", [("N", "Nbb", (5.0, 0.0, 0.0)), ("CH3", "CH3", (6.0, 0.0, 0.0))]),
            (4, "B", "ALA", [("N", "Nbb", (7.0, 0.0, 0.0)), ("CA", "CAbb", (8.0, 0.0, 0.0)), ("C", "CObb", (9.0, 0.0, 0.0)), ("V1", "VIRT", (8.0, 1.0, 0.0))]),
        ],
        bonds=[
            ((1, "A", "C"), (2, "A", "N")),
            ((2, "A", "C"), (3, "A", "N")),
        ],
        ter_indices=(3, 4),
    )

    pose = save_rosetta(structure)
    round_tripped = load_rosetta(pose)

    assert pose.append_operations == [
        ("bond", False),
        ("bond", False),
        ("bond", False),
        ("jump", True),
    ]
    assert pose.chain(4) == 2
    assert len(round_tripped.residues) == 4
    assert [res.chain for res in round_tripped.residues] == ["A", "A", "A", "B"]
    assert round_tripped.residues[2].ter is True
    assert round_tripped.residues[3].ter is True


@pytest.mark.parametrize(
    "hydrogen_names",
    [
        ("H", "H2", "H3"),
        ("H1", "H2", "H3"),
        ("H", "H1", "H2"),
        ("H1", "HT2", "HT3"),
    ],
)
def test_fake_structure_to_pose_handles_nterminal_hydrogen_aliases(monkeypatch, hydrogen_names):
    nterm_gly = FakeResidueType(
        unique_name="GLY:NtermProteinFull",
        name3_value="GLY",
        name1_value="G",
        atom_order=("N", "CA", "C", "1H", "2H", "3H"),
        atom_types={
            "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
            "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
            "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
            "1H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "2H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "3H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
        },
    )
    fake_import = build_fake_import([nterm_gly])
    monkeypatch.setattr(rosetta_pose_module, "_import_pyrosetta", lambda: fake_import)
    structure = _build_structure(
        [
            (
                0,
                "A",
                "GLY",
                [
                    ("N", "Nbb", (0.0, 0.0, 0.0)),
                    ("CA", "CAbb", (1.0, 0.0, 0.0)),
                    ("C", "CObb", (2.0, 0.0, 0.0)),
                    (hydrogen_names[0], "Hpol", (-0.5, 0.0, 0.0)),
                    (hydrogen_names[1], "Hpol", (-0.5, 0.5, 0.0)),
                    (hydrogen_names[2], "Hpol", (-0.5, -0.5, 0.0)),
                ],
            ),
        ],
        bonds=[],
        ter_indices=(1,),
        zero_hydrogen_atomic_numbers=True,
    )

    pose = save_rosetta(structure)

    assert pose.total_residue() == 1


@pytest.mark.parametrize(
    "alpha_hydrogen_names",
    [
        ("HA2", "HA3"),
        ("HA1", "HA2"),
        ("1HA", "2HA"),
        ("2HA", "3HA"),
    ],
)
def test_fake_structure_to_pose_handles_leading_digit_hydrogen_stem_aliases(
    monkeypatch,
    alpha_hydrogen_names,
):
    nterm_gly = FakeResidueType(
        unique_name="GLY:NtermProteinFull",
        name3_value="GLY",
        name1_value="G",
        atom_order=("N", "CA", "C", "1H", "2H", "3H", "1HA", "2HA"),
        atom_types={
            "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
            "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
            "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
            "1H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "2H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "3H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "1HA": FakeAtomType(element="H", atom_type_name="Hapo", lj_radius=1.0, lj_wdepth=0.02),
            "2HA": FakeAtomType(element="H", atom_type_name="Hapo", lj_radius=1.0, lj_wdepth=0.02),
        },
    )
    fake_import = build_fake_import([nterm_gly])
    monkeypatch.setattr(rosetta_pose_module, "_import_pyrosetta", lambda: fake_import)
    structure = _build_structure(
        [
            (
                0,
                "A",
                "GLY",
                [
                    ("N", "Nbb", (0.0, 0.0, 0.0)),
                    ("CA", "CAbb", (1.0, 0.0, 0.0)),
                    ("C", "CObb", (2.0, 0.0, 0.0)),
                    ("H1", "Hpol", (-0.5, 0.0, 0.0)),
                    ("H2", "Hpol", (-0.5, 0.5, 0.0)),
                    ("H3", "Hpol", (-0.5, -0.5, 0.0)),
                    (alpha_hydrogen_names[0], "Hapo", (1.0, 0.5, 0.5)),
                    (alpha_hydrogen_names[1], "Hapo", (1.0, -0.5, -0.5)),
                ],
            ),
        ],
        bonds=[],
        ter_indices=(1,),
        zero_hydrogen_atomic_numbers=True,
    )

    pose = save_rosetta(structure)

    assert pose.total_residue() == 1


def test_fake_structure_to_pose_does_not_force_nterminus_variant_without_nh3(monkeypatch):
    internal_gly = FakeResidueType(
        unique_name="GLY",
        name3_value="GLY",
        name1_value="G",
        atom_order=("N", "H", "CA", "1HA", "2HA", "C", "O"),
        atom_types={
            "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
            "H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
            "1HA": FakeAtomType(element="H", atom_type_name="Hapo", lj_radius=1.0, lj_wdepth=0.02),
            "2HA": FakeAtomType(element="H", atom_type_name="Hapo", lj_radius=1.0, lj_wdepth=0.02),
            "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
            "O": FakeAtomType(element="O", atom_type_name="OCbb", lj_radius=1.5, lj_wdepth=0.2),
        },
    )
    nterm_gly = FakeResidueType(
        unique_name="GLY:NtermProteinFull",
        name3_value="GLY",
        name1_value="G",
        atom_order=("N", "1H", "2H", "3H", "CA", "1HA", "2HA", "C", "O"),
        atom_types={
            "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
            "1H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "2H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "3H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
            "1HA": FakeAtomType(element="H", atom_type_name="Hapo", lj_radius=1.0, lj_wdepth=0.02),
            "2HA": FakeAtomType(element="H", atom_type_name="Hapo", lj_radius=1.0, lj_wdepth=0.02),
            "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
            "O": FakeAtomType(element="O", atom_type_name="OCbb", lj_radius=1.5, lj_wdepth=0.2),
        },
    )
    internal_ala = FakeResidueType(
        unique_name="ALA",
        name3_value="ALA",
        name1_value="A",
        atom_order=("N", "H", "CA", "C", "O"),
        atom_types={
            "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
            "H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
            "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
            "O": FakeAtomType(element="O", atom_type_name="OCbb", lj_radius=1.5, lj_wdepth=0.2),
        },
    )
    fake_import = build_fake_import([nterm_gly, internal_gly, internal_ala])
    monkeypatch.setattr(rosetta_pose_module, "_import_pyrosetta", lambda: fake_import)
    structure = _build_structure(
        [
            (
                0,
                "A",
                "GLY",
                [
                    ("N", "Nbb", (0.0, 0.0, 0.0)),
                    ("H", "Hpol", (-0.5, 0.0, 0.0)),
                    ("CA", "CAbb", (1.0, 0.0, 0.0)),
                    ("HA2", "Hapo", (1.0, 0.5, 0.5)),
                    ("HA3", "Hapo", (1.0, -0.5, -0.5)),
                    ("C", "CObb", (2.0, 0.0, 0.0)),
                    ("O", "OCbb", (2.5, 0.0, 0.0)),
                ],
            ),
            (
                1,
                "A",
                "ALA",
                [
                    ("N", "Nbb", (3.0, 0.0, 0.0)),
                    ("H", "Hpol", (3.0, 0.5, 0.0)),
                    ("CA", "CAbb", (4.0, 0.0, 0.0)),
                    ("C", "CObb", (5.0, 0.0, 0.0)),
                    ("O", "OCbb", (5.5, 0.0, 0.0)),
                ],
            ),
        ],
        bonds=[((0, "A", "C"), (1, "A", "N"))],
    )

    pose = save_rosetta(structure)

    assert pose.total_residue() == 2
    assert pose.residue(1).type().name() == "GLY"


def test_fake_structure_to_pose_allows_unmapped_virtual_atoms(monkeypatch):
    pro_type = FakeResidueType(
        unique_name="PRO",
        name3_value="PRO",
        name1_value="P",
        atom_order=("N", "NV", "CD", "CA", "C", "O"),
        atom_types={
            "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
            "NV": FakeAtomType(
                element="EP",
                atom_type_name="VIRT",
                lj_radius=0.0,
                lj_wdepth=0.0,
                is_virtual=True,
            ),
            "CD": FakeAtomType(element="C", atom_type_name="CH2", lj_radius=1.8, lj_wdepth=0.1),
            "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
            "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
            "O": FakeAtomType(element="O", atom_type_name="OCbb", lj_radius=1.5, lj_wdepth=0.2),
        },
    )
    fake_import = build_fake_import([pro_type])
    monkeypatch.setattr(rosetta_pose_module, "_import_pyrosetta", lambda: fake_import)
    structure = _build_structure(
        [
            (
                3,
                "A",
                "PRO",
                [
                    ("N", "Nbb", (0.0, 0.0, 0.0)),
                    ("CD", "CH2", (1.0, 0.0, 0.0)),
                    ("CA", "CAbb", (2.0, 0.0, 0.0)),
                    ("C", "CObb", (3.0, 0.0, 0.0)),
                    ("O", "OCbb", (4.0, 0.0, 0.0)),
                ],
            ),
        ],
        bonds=[],
    )

    pose = save_rosetta(structure)

    assert pose.total_residue() == 1
    assert pose.residue(1).type().name() == "PRO"


def test_fake_structure_to_pose_uses_cterminus_variant_only_with_terminal_oxygen(monkeypatch):
    ala_type = FakeResidueType(
        unique_name="ALA",
        name3_value="ALA",
        name1_value="A",
        atom_order=("N", "H", "CA", "C", "O"),
        atom_types={
            "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
            "H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
            "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
            "O": FakeAtomType(element="O", atom_type_name="OCbb", lj_radius=1.5, lj_wdepth=0.2),
        },
    )
    cterm_ala_type = FakeResidueType(
        unique_name="ALA:CtermProteinFull",
        name3_value="ALA",
        name1_value="A",
        atom_order=("N", "H", "CA", "C", "O", "OXT"),
        atom_types={
            "N": FakeAtomType(element="N", atom_type_name="Nbb", lj_radius=1.4, lj_wdepth=0.2),
            "H": FakeAtomType(element="H", atom_type_name="Hpol", lj_radius=1.0, lj_wdepth=0.02),
            "CA": FakeAtomType(element="C", atom_type_name="CAbb", lj_radius=1.8, lj_wdepth=0.1),
            "C": FakeAtomType(element="C", atom_type_name="CObb", lj_radius=1.7, lj_wdepth=0.12),
            "O": FakeAtomType(element="O", atom_type_name="OCbb", lj_radius=1.5, lj_wdepth=0.2),
            "OXT": FakeAtomType(element="O", atom_type_name="OCbb", lj_radius=1.5, lj_wdepth=0.2),
        },
    )
    fake_import = build_fake_import([cterm_ala_type, ala_type])
    monkeypatch.setattr(rosetta_pose_module, "_import_pyrosetta", lambda: fake_import)

    no_oxt = _build_structure(
        [(1, "A", "ALA", [("N", "Nbb", (0.0, 0.0, 0.0)), ("H", "Hpol", (0.0, 0.5, 0.0)), ("CA", "CAbb", (1.0, 0.0, 0.0)), ("C", "CObb", (2.0, 0.0, 0.0)), ("O", "OCbb", (3.0, 0.0, 0.0))])],
        bonds=[],
        ter_indices=(1,),
    )
    with_oxt = _build_structure(
        [(1, "A", "ALA", [("N", "Nbb", (0.0, 0.0, 0.0)), ("H", "Hpol", (0.0, 0.5, 0.0)), ("CA", "CAbb", (1.0, 0.0, 0.0)), ("C", "CObb", (2.0, 0.0, 0.0)), ("O", "OCbb", (3.0, 0.0, 0.0)), ("OXT", "OCbb", (2.0, 1.0, 0.0))])],
        bonds=[],
        ter_indices=(1,),
    )

    assert save_rosetta(no_oxt).residue(1).type().name() == "ALA"
    assert save_rosetta(with_oxt).residue(1).type().name() == "ALA:CtermProteinFull"


def test_fake_structure_to_pose_handles_disulfides_ptms_and_metadata_copy(monkeypatch):
    _install_fake_pyrosetta(monkeypatch)
    structure = _build_structure(
        [
            (1, "A", "SEP", [("N", "Nbb", (0.0, 0.0, 0.0)), ("CA", "CAbb", (1.0, 0.0, 0.0)), ("C", "CObb", (2.0, 0.0, 0.0)), ("P", "Pha", (1.0, 1.0, 0.0))]),
            (2, "B", "CYS", [("N", "Nbb", (3.0, 0.0, 0.0)), ("CA", "CAbb", (4.0, 0.0, 0.0)), ("C", "CObb", (5.0, 0.0, 0.0)), ("SG", "Sthio", (4.0, 1.0, 0.0))]),
            (3, "C", "CYS", [("N", "Nbb", (6.0, 0.0, 0.0)), ("CA", "CAbb", (7.0, 0.0, 0.0)), ("C", "CObb", (8.0, 0.0, 0.0)), ("SG", "Sthio", (7.0, 1.0, 0.0))]),
        ],
        bonds=[((2, "B", "SG"), (3, "C", "SG"))],
        ter_indices=(1, 2, 3),
    )

    pose = save_rosetta(structure)
    round_tripped = load_rosetta(pose)
    copied = copy.copy(round_tripped)
    serialized = pickle.loads(pickle.dumps(round_tripped))

    assert pose.disulfides == [(2, 3)]
    assert pose.append_operations == [
        ("bond", False),
        ("jump", True),
        ("jump", True),
    ]
    assert [res.name for res in round_tripped.residues] == ["SEP", "CYS", "CYS"]
    assert any({bond.atom1.residue.idx + 1, bond.atom2.residue.idx + 1} == {2, 3} for bond in round_tripped.bonds)
    assert hasattr(copied, rosetta_pose_module.ROSETTA_BRIDGE_METADATA_ATTR)
    assert hasattr(serialized, rosetta_pose_module.ROSETTA_BRIDGE_METADATA_ATTR)
