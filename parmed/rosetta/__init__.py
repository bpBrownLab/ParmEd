"""PyRosetta bridge helpers exposed by vendored ParmEd."""

__author__ = "Carlos Xavier Hernandez <cxh@stanford.edu>"

from .pose import (
    PARMED_TO_ROSETTA_RESIDUE_ALIASES,
    POSE_TO_PARMED_ATOM_NAME_ALIASES,
    ROSETTA_BRIDGE_METADATA_ATTR,
    RosettaPose,
)

load_rosetta = RosettaPose.load
save_rosetta = RosettaPose.dump

__all__ = [
    "PARMED_TO_ROSETTA_RESIDUE_ALIASES",
    "POSE_TO_PARMED_ATOM_NAME_ALIASES",
    "ROSETTA_BRIDGE_METADATA_ATTR",
    "RosettaPose",
    "load_rosetta",
    "save_rosetta",
]
