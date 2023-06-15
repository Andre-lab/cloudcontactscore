#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ModifyRepresentation class: Modifies the amino acid representation of a pose.
@Author: Mads Jeppesen
@Date: 4/7/21
"""
from pyrosettawrapper.taskfactorywrapper import TaskFactoryWrapper
from cloudcontactscore.util import build_new_aa_repr
from pyrosetta import SwitchResidueTypeSetMover
from typing import Optional, Sequence
class ModifyRepresentation:
    """Modifies the amino acid representation in the pose."""

    aa_1_letter = ['C', 'D', 'S', 'Q', 'K', 'I', 'P', 'T', 'F', 'N', 'G', 'H', 'L', 'R', 'W', 'A', 'V', 'E', 'Y', 'M']
    def __init__(self, mutant: str = "A", centroid: bool = False, affect_only: Optional[Sequence[str]] = None,
                 unaffected: Optional[Sequence[str]] = None, residues=None, interface_only: bool = False, cb_dist_cutoff: float = 10,
                 nearby_atom_cutoff: float = 0):
        """Initialize the mover.

        :param mutant: The residue to mutate to (1-letter).
        :param centroid: To change to centroid mode or not.
        :param affect_only: Only affect the residue types (1-letter) in the given sequence.
        :param unaffected: Do not affect the residue types (1-letter) in the given sequence.
        :param residues: Mutate only these specific residues (residue pose positions: int) in the given sequence.
        :param interface_only: To only mutate residue interface residues or not.
        :param cb_dist_cutoff: If using interface_only, in conjunction to nearby_atom, selects the interface based on
        the distance cutoff to any nearby cb atoms (Angstrom). (See RestrictToInterfaceVector for more).
        :param nearby_atom: If using interface_only, in conjunction to cb_dist_cutoff, selects the interface based on
        the distance to any nearby atom. (See RestrictToInterfaceVector for more).
        """
        self.mutant = mutant
        self.centroid = centroid
        self.affect_only = affect_only
        self.unaffected = unaffected
        self.interface_only = interface_only
        self.cb_dist_cutoff = cb_dist_cutoff
        self.nearby_atom_cutoff = nearby_atom_cutoff
        if not residues is None:
            if not isinstance(residues, list):
                residues = list(residues)
        self.residues = residues

    def switch_to_centroid(self, pose):
        """Switches pose to centroid mode."""
        switch = SwitchResidueTypeSetMover("centroid")
        switch.apply(pose)

    # FIXME: Can some of this be replaced by the SequenceThreader class?
    def apply(self, pose):
        """Changes the representation of a pose."""
        # change to pose to centroid if set
        if self.centroid:
            self.switch_to_centroid(pose)
        if self.interface_only:
            print("modifying the interface AA representation.")
            # FIXME: There must be a smarter way to do this
            # we dont want to limit the taskfactory at this point, so
            # we have to create a temporary pose to do this.
            temp_pose = pose.clone()
            task = TaskFactoryWrapper()
            task.RestrictToInterfaceVector(None, cb_dist_cutoff=self.cb_dist_cutoff, nearby_atom_cutoff=self.nearby_atom_cutoff)
            task.get_designable_residues(temp_pose)
            residues = task.get_designable_residues(temp_pose)
            # extra option: if affect only is also set, it will take away those residues that are not specified
            # with 'affect only'
            if self.affect_only:
                temp_residues = []
                for resi in residues:
                    if pose.residue(resi).name1() in self.affect_only:
                        temp_residues.append(resi)
                residues = temp_residues
        # limit mutations to certain residue types
        elif self.affect_only:
            residues = None
            print(f"modifying the following residues: {' '.join(self.affect_only)}")
            if self.unaffected == None:
                self.unaffected = [r for r in self.aa_1_letter if r not in self.affect_only]
            else:
               self.unaffected += [r for r in self.aa_1_letter if (r not in self.affect_only and r not in self.unaffected)]
        else:
            residues = None
        if not self.residues is None:
            # if residues have not been selected previously, just use the specified residues
            if not residues:
                residues = self.residues
            # else remove the ones that are not specifed in the residues
            else:
                residues = [resi for resi in residues if resi in self.residues]
        # build the representation
        build_new_aa_repr(pose, mutant_aa=self.mutant, residues=residues, types_not_to_mutate=self.unaffected)