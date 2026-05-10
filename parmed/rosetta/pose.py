"""
In-memory bridge between PyRosetta ``Pose`` and ParmEd ``Structure`` objects.
"""

from __future__ import annotations

import re
from typing import Any

from ..exceptions import RosettaError
from ..periodic_table import AtomicNum, Mass
from ..structure import Structure
from ..topologyobjects import Atom, Bond, ExtraPoint

ROSETTA_BRIDGE_METADATA_ATTR = "_rosetta_bridge_metadata"
DEFAULT_ROSETTA_TYPE_SET = "fa_standard"

POSE_TO_PARMED_ATOM_NAME_ALIASES = {
    "1H": ("H", "H1", "HN", "HN1"),
    "2H": ("H1", "H2", "HN2"),
    "3H": ("H2", "H3", "HN3"),
    "1HH3": ("H1",),
    "2HH3": ("H2",),
    "3HH3": ("H3",),
    "1HH2": ("H1",),
    "2HH2": ("H2",),
    "3HH2": ("H3",),
    "1HA": ("HA2",),
    "2HA": ("HA3",),
    "HN2": ("H",),
}

PARMED_TO_ROSETTA_RESIDUE_ALIASES = {
    "HID": ("HID", "HIS_D", "HIS"),
    "HIE": ("HIE", "HIS", "HIS_E"),
    "HIP": ("HIP", "HIS_P", "HIS"),
    "CYX": ("CYX", "CYD", "CYS"),
    "CYM": ("CYM", "CYZ", "CYS"),
    # Protonated Asp/Glu are intentionally not aliased here. In Rosetta they are
    # commonly exposed through pH-mode/custom params rather than a stable
    # always-loaded fa_standard three-letter code, and silently collapsing them
    # onto ASP/GLU would defeat protonation-state studies.
    "SEP": ("SEP",),
    "TPO": ("TPO",),
    "PTR": ("PTR",),
    "ACE": ("ACE",),
    "NME": ("NME",),
}

_LEADING_DIGIT_ATOM_NAME = re.compile(r"^([123])([A-Za-z0-9]+)$")


def _import_pyrosetta():
    try:
        import pyrosetta  # pyright: ignore[reportMissingImports]
        from pyrosetta import Pose  # pyright: ignore[reportMissingImports]
        from pyrosetta.rosetta.core.id import AtomID  # pyright: ignore[reportMissingImports]
        from pyrosetta.rosetta.core.conformation import ResidueFactory  # pyright: ignore[reportMissingImports]
        from pyrosetta.rosetta.core.chemical import ChemicalManager  # pyright: ignore[reportMissingImports]
        from pyrosetta.rosetta.numeric import xyzVector_double_t  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError("Could not load the PyRosetta module.") from exc
    return pyrosetta, Pose, AtomID, ResidueFactory, ChemicalManager, xyzVector_double_t


def _n_prior(pose, nbr):
    prior = -1
    for i in range(1, nbr.rsd()):
        prior += pose.residue(i).natoms()
    return prior + nbr.atomno()


def _safe_call(obj, attr, *args, default=None):
    if obj is None or not hasattr(obj, attr):
        return default
    value = getattr(obj, attr)
    try:
        return value(*args) if callable(value) else value
    except Exception:
        return default


def _strip(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalize_pdb_char(value: Any) -> str:
    text = _strip(value)
    return "" if text in {"", "^"} else text


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _default_chain_id(chain_index: int) -> str:
    if 1 <= chain_index <= 26:
        return chr(ord("A") + chain_index - 1)
    return str(chain_index)


def _unique_strings(values):
    seen = set()
    ordered = []
    for value in values:
        text = _strip(value)
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _vector_to_xyz(vector) -> tuple[float, float, float]:
    try:
        return float(vector[0]), float(vector[1]), float(vector[2])
    except Exception:
        return float(vector.x), float(vector.y), float(vector.z)


def _total_residue(pose) -> int:
    return _coerce_int(
        _safe_call(pose, "total_residue", default=_safe_call(pose, "size", default=0)),
        default=0,
    )


def _pose_chain_end(pose, chain_index: int) -> int:
    total = _total_residue(pose)
    chain_end = _coerce_int(_safe_call(pose, "chain_end", chain_index, default=total), default=total)
    return chain_end or total


def _structure_bridge_metadata(structure):
    metadata = getattr(structure, ROSETTA_BRIDGE_METADATA_ATTR, None)
    return metadata if isinstance(metadata, dict) else {}


def _structure_residue_metadata(structure):
    residues = _structure_bridge_metadata(structure).get("residues")
    if isinstance(residues, list) and len(residues) == len(structure.residues):
        return residues
    return [None] * len(structure.residues)


def _has_name(residue_type_set, name: str) -> bool:
    if hasattr(residue_type_set, "has_name"):
        return bool(residue_type_set.has_name(name))
    try:
        residue_type_set.name_map(name)
    except Exception:
        return False
    return True


def _has_name3(residue_type_set, name3: str) -> bool:
    if hasattr(residue_type_set, "has_name3"):
        return bool(residue_type_set.has_name3(name3))
    try:
        residue_type_set.get_representative_type_name3(name3)
    except Exception:
        return False
    return True


def _collect_residue_metadata(pose, resid, residue, pdb_info):
    residue_type = _safe_call(residue, "type")
    chain_index = _coerce_int(
        _safe_call(pose, "chain", resid, default=_safe_call(residue, "chain", default=1)),
        default=1,
    )
    ter = resid == _pose_chain_end(pose, chain_index)
    chain_id = _default_chain_id(chain_index)
    residue_number = resid
    insertion_code = ""
    if pdb_info is not None:
        residue_number = _coerce_int(_safe_call(pdb_info, "number", resid, default=resid), default=resid)
        chain_id = _normalize_pdb_char(_safe_call(pdb_info, "chain", resid, default=chain_id)) or chain_id
        insertion_code = _normalize_pdb_char(_safe_call(pdb_info, "icode", resid, default=""))
    return {
        "seqpos": resid,
        "type_name": _strip(_safe_call(residue_type, "name", default="")),
        "base_name": _strip(_safe_call(residue_type, "base_name", default=_safe_call(residue, "name3", default=""))),
        "name1": _strip(_safe_call(residue_type, "name1", default="")),
        "name3": _strip(_safe_call(residue_type, "name3", default=_safe_call(residue, "name3", default=""))),
        "chain_index": chain_index,
        "chain_id": chain_id,
        "residue_number": residue_number,
        "insertion_code": insertion_code,
        "is_polymer": bool(_safe_call(residue, "is_polymer", default=False)),
        "is_lower_terminus": bool(_safe_call(residue, "is_lower_terminus", default=False)),
        "is_upper_terminus": bool(_safe_call(residue, "is_upper_terminus", default=False)),
        "is_virtual_residue": bool(_safe_call(residue, "is_virtual_residue", default=False)),
        "connected_lower_residue": _coerce_int(
            _safe_call(residue, "connected_residue_at_lower", default=0),
            default=0,
        ),
        "connected_upper_residue": _coerce_int(
            _safe_call(residue, "connected_residue_at_upper", default=0),
            default=0,
        ),
        "ter": bool(ter),
        "reslabels": [str(label) for label in (_safe_call(pdb_info, "get_reslabels", resid, default=[]) or [])],
        "atoms": [],
    }


def _collect_atom_metadata(atom, residue, resid, atno, pdb_info):
    atom_type = _safe_call(residue, "atom_type", atno)
    if atom_type is None:
        raise RosettaError(f"Could not query atom type for residue {resid} atom {atno}.")
    atom_name = _strip(_safe_call(residue, "atom_name", atno, default=""))
    is_virtual = bool(_safe_call(atom_type, "is_virtual", default=False))
    element = "EP" if is_virtual else _strip(_safe_call(atom_type, "element", default=""))
    try:
        atomic_number = AtomicNum[element]
        mass = Mass[element]
    except KeyError as err:
        raise RosettaError(f"Could not recognize element: {element}") from err
    coordinates = _vector_to_xyz(_safe_call(atom, "xyz", default=(0.0, 0.0, 0.0)))
    return {
        "pose_name": atom_name,
        "structure_name": atom_name,
        "element": element,
        "atomic_number": atomic_number,
        "mass": mass,
        "atom_type_name": _strip(_safe_call(atom_type, "atom_type_name", default=_safe_call(atom_type, "name", default=""))),
        "mm_name": _strip(_safe_call(_safe_call(residue, "type"), "mm_name", atno, default="")),
        "charge": _coerce_float(_safe_call(residue, "atomic_charge", atno, default=0.0), default=0.0),
        "lj_radius": _coerce_float(_safe_call(atom_type, "lj_radius", default=0.0), default=0.0),
        "lj_wdepth": _coerce_float(_safe_call(atom_type, "lj_wdepth", default=0.0), default=0.0),
        "occupancy": _coerce_float(_safe_call(pdb_info, "occupancy", resid, atno, default=0.0), default=0.0),
        "bfactor": _coerce_float(_safe_call(pdb_info, "bfactor", resid, atno, default=0.0), default=0.0),
        "altloc": _normalize_pdb_char(_safe_call(pdb_info, "alt_loc", resid, atno, default="")),
        "is_het": _safe_call(pdb_info, "is_het", resid, atno, default=None),
        "is_virtual": is_virtual,
        "coordinates": coordinates,
    }


def _attach_structure_metadata(structure, metadata):
    setattr(structure, ROSETTA_BRIDGE_METADATA_ATTR, metadata)
    for residue, residue_metadata in zip(structure.residues, metadata.get("residues", ())):
        residue.ter = bool(residue_metadata.get("ter", False))


def _candidate_unique_residue_names(base_name: str, lower_terminus: bool, upper_terminus: bool):
    if not base_name:
        return []
    if ":" in base_name:
        return [base_name]
    names = []
    if lower_terminus and upper_terminus:
        names.extend(
            (
                f"{base_name}:NtermProteinFull:CtermProteinFull",
                f"{base_name}:CtermProteinFull:NtermProteinFull",
            )
        )
    if lower_terminus:
        names.append(f"{base_name}:NtermProteinFull")
    if upper_terminus:
        names.append(f"{base_name}:CtermProteinFull")
    names.append(base_name)
    return names


def _residues_are_polymerically_bonded(left_residue, right_residue) -> bool:
    for atom in left_residue.atoms:
        for bond in atom.bonds:
            partner = bond.atom2 if bond.atom1 is atom else bond.atom1
            if partner.residue is right_residue and {_strip(atom.name).upper(), _strip(partner.name).upper()} == {"C", "N"}:
                return True
    return False


def _is_polymeric_successor(previous_residue, current_residue, previous_metadata, current_metadata, current_index: int) -> bool:
    if previous_residue is None or previous_residue.ter:
        return False
    if current_metadata and _coerce_int(current_metadata.get("connected_lower_residue"), default=0) == current_index - 1:
        return True
    if previous_metadata and _coerce_int(previous_metadata.get("connected_upper_residue"), default=0) == current_index:
        return True
    return _residues_are_polymerically_bonded(previous_residue, current_residue)


def _infer_lower_terminus(structure, residue_index: int, residue_metadata):
    metadata = residue_metadata[residue_index - 1]
    if metadata is not None and "is_lower_terminus" in metadata:
        return bool(metadata["is_lower_terminus"])
    if residue_index == 1:
        return True
    return not _is_polymeric_successor(
        structure.residues[residue_index - 2],
        structure.residues[residue_index - 1],
        residue_metadata[residue_index - 2],
        metadata,
        residue_index,
    )


def _infer_upper_terminus(structure, residue_index: int, residue_metadata):
    metadata = residue_metadata[residue_index - 1]
    if metadata is not None and "is_upper_terminus" in metadata:
        return bool(metadata["is_upper_terminus"])
    residue = structure.residues[residue_index - 1]
    if residue.ter or residue_index == len(structure.residues):
        return True
    return not _is_polymeric_successor(
        residue,
        structure.residues[residue_index],
        metadata,
        residue_metadata[residue_index],
        residue_index + 1,
    )


def _resolve_residue_type(residue_type_set, residue, residue_meta, lower_terminus: bool, upper_terminus: bool):
    residue_name = _strip(residue.name).upper()
    metadata = residue_meta or {}
    candidate_names = []
    exact_type_name = _strip(metadata.get("type_name")).upper()
    if exact_type_name:
        candidate_names.append(exact_type_name)
    base_candidates = [
        residue_name,
        _strip(metadata.get("name3")).upper(),
        _strip(metadata.get("base_name")).upper(),
    ]
    base_candidates.extend(PARMED_TO_ROSETTA_RESIDUE_ALIASES.get(residue_name, ()))
    for base_name in _unique_strings(base_candidates):
        candidate_names.extend(_candidate_unique_residue_names(base_name, lower_terminus, upper_terminus))
    for candidate in _unique_strings(candidate_names):
        if _has_name(residue_type_set, candidate):
            return residue_type_set.name_map(candidate)
    for candidate in _unique_strings(base_candidates):
        if _has_name3(residue_type_set, candidate):
            return residue_type_set.get_representative_type_name3(candidate)
        if hasattr(residue_type_set, "get_representative_type_base_name"):
            try:
                return residue_type_set.get_representative_type_base_name(candidate)
            except Exception:
                continue
    raise RosettaError(f"Unsupported residue for Pose conversion: {residue.name}")


def _residue_atom_metadata(residue_meta, pose_atom_name: str):
    if not residue_meta:
        return None
    for atom_meta in residue_meta.get("atoms", ()) or ():
        if _strip(atom_meta.get("pose_name")).upper() == pose_atom_name.upper():
            return atom_meta
    return None


def _canonical_atom_candidates(residue_type, pose_atom_name: str):
    candidates = [pose_atom_name]
    canonical_alias = _strip(_safe_call(residue_type, "canonical_atom_alias", pose_atom_name, default=""))
    if canonical_alias:
        candidates.append(canonical_alias)
    canonical_aliases = _safe_call(residue_type, "canonical_atom_aliases", default={}) or {}
    if hasattr(canonical_aliases, "get"):
        candidates.append(canonical_aliases.get(pose_atom_name))
    atom_aliases = _safe_call(residue_type, "atom_aliases", default={}) or {}
    items = atom_aliases.items() if hasattr(atom_aliases, "items") else ()
    for alias_name, canonical_name in items:
        if _strip(canonical_name).upper() == pose_atom_name.upper():
            candidates.append(alias_name)
    candidates.extend(POSE_TO_PARMED_ATOM_NAME_ALIASES.get(pose_atom_name, ()))
    match = _LEADING_DIGIT_ATOM_NAME.match(pose_atom_name)
    if match is not None:
        digit, stem = match.groups()
        candidates.append(f"{stem}{digit}")
        candidates.append(f"{stem}{int(digit) + 1}")
    return _unique_strings(candidates)


def _is_terminal_hydrogen_name(atom_name: str) -> bool:
    return _strip(atom_name).upper() in {"1H", "2H", "3H"}


def _hydrogen_order(atom_name: str) -> int:
    match = _LEADING_DIGIT_ATOM_NAME.match(_strip(atom_name).upper())
    if match is None:
        return 0
    return int(match.group(1)) - 1


def _leading_digit_hydrogen_stem(atom_name: str) -> str:
    match = _LEADING_DIGIT_ATOM_NAME.match(_strip(atom_name).upper())
    if match is None:
        return ""
    stem = match.group(2)
    return stem if stem.startswith("H") else ""


def _hydrogen_stem(atom_name: str) -> str:
    text = _strip(atom_name).upper()
    leading_stem = _leading_digit_hydrogen_stem(text)
    if leading_stem:
        return leading_stem
    if not text.startswith("H"):
        return ""
    if len(text) > 1 and text[-1].isdigit():
        return text[:-1]
    return text


def _is_hydrogen_atom(atom) -> bool:
    try:
        return int(getattr(atom, "atomic_number", 0) or 0) == 1
    except Exception:
        return _strip(getattr(atom, "name", "")).upper().startswith("H")


def _match_terminal_hydrogen_by_order(residue, pose_atom_name: str, used_atom_ids=None):
    if not _is_terminal_hydrogen_name(pose_atom_name):
        return None
    used_atom_ids = used_atom_ids or set()
    all_hydrogens = [atom for atom in residue.atoms if _is_hydrogen_atom(atom)]
    hydrogens = [
        atom
        for atom in all_hydrogens
        if id(atom) not in used_atom_ids
    ]
    if not hydrogens:
        return None
    if used_atom_ids:
        return hydrogens[0]
    order = _hydrogen_order(pose_atom_name)
    if order < len(all_hydrogens):
        return all_hydrogens[order]
    return hydrogens[0]


def _match_leading_digit_hydrogen_by_stem(residue, pose_atom_name: str, used_atom_ids=None):
    pose_stem = _leading_digit_hydrogen_stem(pose_atom_name)
    if not pose_stem:
        return None
    used_atom_ids = used_atom_ids or set()
    for atom in residue.atoms:
        if id(atom) in used_atom_ids or not _is_hydrogen_atom(atom):
            continue
        if _hydrogen_stem(getattr(atom, "name", "")) == pose_stem:
            return atom
    return None


def _match_structure_atom(residue, residue_type, pose_atom_name: str, residue_meta, used_atom_ids=None):
    used_atom_ids = used_atom_ids or set()
    atoms_by_name = {_strip(atom.name).upper(): atom for atom in residue.atoms}
    atom_meta = _residue_atom_metadata(residue_meta, pose_atom_name)
    if atom_meta is not None:
        structure_name = _strip(atom_meta.get("structure_name")).upper()
        if structure_name in atoms_by_name and id(atoms_by_name[structure_name]) not in used_atom_ids:
            return atoms_by_name[structure_name]
    for candidate in _canonical_atom_candidates(residue_type, pose_atom_name):
        atom = atoms_by_name.get(candidate.upper())
        if atom is not None and id(atom) not in used_atom_ids:
            return atom
    fallback_atom = _match_terminal_hydrogen_by_order(residue, pose_atom_name, used_atom_ids)
    if fallback_atom is not None:
        return fallback_atom
    fallback_atom = _match_leading_digit_hydrogen_by_stem(residue, pose_atom_name, used_atom_ids)
    if fallback_atom is not None:
        return fallback_atom
    return None


def _apply_pdb_info(pyrosetta, pose, structure, residue_metadata):
    pdb_info = pyrosetta.rosetta.core.pose.PDBInfo(pose)
    for residue_index, residue in enumerate(structure.residues, start=1):
        residue_meta = residue_metadata[residue_index - 1] or {}
        chain_id = _normalize_pdb_char(residue_meta.get("chain_id") or residue.chain)
        if not chain_id:
            chain_id = _default_chain_id(
                _coerce_int(_safe_call(pose, "chain", residue_index, default=1), default=1)
            )
        pdb_info.chain(residue_index, chain_id)
        residue_number = residue.number if residue.number != -1 else residue_index
        pdb_info.number(
            residue_index,
            _coerce_int(residue_meta.get("residue_number"), default=residue_number),
        )
        insertion_code = _normalize_pdb_char(residue_meta.get("insertion_code") or residue.insertion_code)
        if insertion_code:
            pdb_info.icode(residue_index, insertion_code)
        for label in residue_meta.get("reslabels", ()) or ():
            try:
                pdb_info.add_reslabel(residue_index, str(label))
            except Exception:
                continue
        pose_residue = pose.residue(residue_index)
        pose_residue_type = _safe_call(pose_residue, "type")
        for atom_index in range(1, pose_residue.natoms() + 1):
            pose_atom_name = _strip(_safe_call(pose_residue, "atom_name", atom_index, default=""))
            source_atom = _match_structure_atom(residue, pose_residue_type, pose_atom_name, residue_meta)
            if source_atom is None:
                continue
            atom_meta = _residue_atom_metadata(residue_meta, pose_atom_name) or {}
            altloc = _normalize_pdb_char(atom_meta.get("altloc") or source_atom.altloc)
            if altloc:
                pdb_info.alt_loc(residue_index, atom_index, altloc)
            pdb_info.occupancy(
                residue_index,
                atom_index,
                _coerce_float(atom_meta.get("occupancy"), default=source_atom.occupancy),
            )
            pdb_info.bfactor(
                residue_index,
                atom_index,
                _coerce_float(atom_meta.get("bfactor"), default=source_atom.bfactor),
            )
            if atom_meta.get("is_het") is not None:
                try:
                    pdb_info.is_het(residue_index, atom_index, bool(atom_meta["is_het"]))
                except Exception:
                    continue
    pose.pdb_info(pdb_info)


def _find_disulfide_pairs(structure):
    pairs = set()
    for bond in structure.bonds:
        atom1 = bond.atom1
        atom2 = bond.atom2
        if atom1.residue is atom2.residue:
            continue
        if atom1.atomic_number != 16 or atom2.atomic_number != 16:
            continue
        if {_strip(atom1.name).upper(), _strip(atom2.name).upper()} != {"SG"}:
            continue
        pairs.add(tuple(sorted((atom1.residue.idx + 1, atom2.residue.idx + 1))))
    return sorted(pairs)


class RosettaPose:
    """Bridge between PyRosetta ``Pose`` and ParmEd ``Structure`` objects."""

    @staticmethod
    def load(pose):
        """
        Load a PyRosetta ``Pose`` object and return a populated ParmEd ``Structure``.

        The converted structure preserves coordinates, bond topology, chain and PDB
        numbering when available, and stores Rosetta-specific provenance in
        ``Structure._rosetta_bridge_metadata`` for later round-trips back into a Pose.
        """
        _, Pose, AtomID, _, _, _ = _import_pyrosetta()
        if not isinstance(pose, Pose):
            raise TypeError("Object is not a PyRosetta Pose object.")

        structure = Structure()
        metadata = {
            "version": 1,
            "type_set": DEFAULT_ROSETTA_TYPE_SET,
            "residues": [],
        }

        atom_number = 1
        conformation = pose.conformation()
        pdb_info = _safe_call(pose, "pdb_info")
        for resid in range(1, _total_residue(pose) + 1):
            residue = pose.residue(resid)
            residue_metadata = _collect_residue_metadata(pose, resid, residue, pdb_info)
            residue_name = residue_metadata["name3"] or _strip(_safe_call(residue, "name3", default=""))
            residue_number = residue_metadata["residue_number"] or resid
            chain_id = residue_metadata["chain_id"]
            insertion_code = residue_metadata["insertion_code"]
            atoms = list(_safe_call(residue, "atoms", default=[]))
            for atno, atom in enumerate(atoms, start=1):
                atom_metadata = _collect_atom_metadata(atom, residue, resid, atno, pdb_info)
                params = dict(
                    atomic_number=atom_metadata["atomic_number"],
                    name=atom_metadata["structure_name"],
                    type=atom_metadata["atom_type_name"],
                    charge=atom_metadata["charge"],
                    mass=atom_metadata["mass"],
                    occupancy=atom_metadata["occupancy"],
                    bfactor=atom_metadata["bfactor"],
                    altloc=atom_metadata["altloc"],
                    number=atom_number,
                    rmin=atom_metadata["lj_radius"],
                    epsilon=atom_metadata["lj_wdepth"],
                )
                if atom_metadata["is_virtual"]:
                    parmed_atom = ExtraPoint(**params)
                else:
                    parmed_atom = Atom(**params)
                parmed_atom.xx, parmed_atom.xy, parmed_atom.xz = atom_metadata["coordinates"]
                structure.add_atom(parmed_atom, residue_name, residue_number, chain_id, insertion_code)
                residue_metadata["atoms"].append(atom_metadata)
                atom_number += 1
                try:
                    for neighbor in conformation.bonded_neighbor_all_res(AtomID(atno, resid)):
                        if neighbor.rsd() < resid or (
                            neighbor.rsd() == resid and neighbor.atomno() < atno
                        ):
                            structure.bonds.append(Bond(structure.atoms[_n_prior(pose, neighbor)], parmed_atom))
                except Exception as err:
                    raise RosettaError("Could not add bonds.") from err
            metadata["residues"].append(residue_metadata)

        _attach_structure_metadata(structure, metadata)
        structure.unchange()
        return structure

    @staticmethod
    def dump(structure, residue_type_set: str = DEFAULT_ROSETTA_TYPE_SET):
        """
        Build a PyRosetta ``Pose`` from a ParmEd ``Structure`` entirely in memory.

        When the structure originated from ``RosettaPose.load``, the exact Rosetta
        residue type name and atom provenance captured in the structure metadata are
        used to recreate the original residue identities and chain segmentation. For
        structures loaded from other sources, the bridge falls back to a strict,
        protein-focused residue-type lookup based on residue names and topology.
        """
        pyrosetta, Pose, AtomID, ResidueFactory, ChemicalManager, xyz_vector = _import_pyrosetta()
        if not isinstance(structure, Structure):
            raise TypeError("Object is not a ParmEd Structure.")
        if not structure.residues:
            raise RosettaError("Cannot convert an empty Structure to a Pose.")

        pose = Pose()
        residue_metadata = _structure_residue_metadata(structure)
        residue_type_set_object = ChemicalManager.get_instance().residue_type_set(residue_type_set)

        for residue_index, residue in enumerate(structure.residues, start=1):
            residue_meta = residue_metadata[residue_index - 1]
            lower_terminus = _infer_lower_terminus(structure, residue_index, residue_metadata)
            upper_terminus = _infer_upper_terminus(structure, residue_index, residue_metadata)
            residue_type = _resolve_residue_type(
                residue_type_set_object,
                residue,
                residue_meta,
                lower_terminus,
                upper_terminus,
            )
            pose_residue = ResidueFactory.create_residue(residue_type)
            if residue_index == 1:
                try:
                    pose.append_residue_by_bond(pose_residue, False)
                except Exception as err:
                    raise RosettaError("Could not append the first residue to an empty Pose.") from err
            else:
                previous_residue = structure.residues[residue_index - 2]
                previous_meta = residue_metadata[residue_index - 2]
                if _is_polymeric_successor(previous_residue, residue, previous_meta, residue_meta, residue_index):
                    pose.append_residue_by_bond(pose_residue, False)
                else:
                    pose.append_residue_by_jump(pose_residue, residue_index - 1, "", "", True)

            pose_residue = pose.residue(residue_index)
            pose_residue_type = _safe_call(pose_residue, "type")
            used_atom_ids = set()
            for atom_index in range(1, pose_residue.natoms() + 1):
                pose_atom_name = _strip(_safe_call(pose_residue, "atom_name", atom_index, default=""))
                source_atom = _match_structure_atom(
                    residue,
                    pose_residue_type,
                    pose_atom_name,
                    residue_meta,
                    used_atom_ids,
                )
                if source_atom is None:
                    raise RosettaError(
                        f"Could not map atom {pose_atom_name} in residue {residue.name} {residue.number}."
                    )
                used_atom_ids.add(id(source_atom))
                try:
                    coordinates = xyz_vector(float(source_atom.xx), float(source_atom.xy), float(source_atom.xz))
                except AttributeError as err:
                    raise RosettaError(
                        f"Missing coordinates for atom {source_atom.name} in residue {residue.name} {residue.number}."
                    ) from err
                pose.set_xyz(AtomID(atom_index, residue_index), coordinates)

        _apply_pdb_info(pyrosetta, pose, structure, residue_metadata)
        disulfide_pairs = _find_disulfide_pairs(structure)
        if disulfide_pairs:
            form_disulfide = pyrosetta.rosetta.core.conformation.form_disulfide
            for lower_residue, upper_residue in disulfide_pairs:
                form_disulfide(pose.conformation(), lower_residue, upper_residue)
        else:
            try:
                pose.conformation().detect_disulfides()
            except Exception:
                pass
        return pose

    to_pose = dump
