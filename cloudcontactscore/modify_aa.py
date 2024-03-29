#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen
@Date: 6/15/23
"""
import pyrosetta
from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

def build_new_aa_repr(pose, mutant_aa, residues=None, types_not_to_mutate=None):
    """Mutates positions in the pose to the mutant_aa. Default = mutate all residues.

    NOTE The packer will complain about not doing anything if nothing is mutated!

    # THIS DOES PACKING ON THE MUTANT.
    # I have not found a way to turn off packing, without also
    # turning off shapedesign. I have looked both in the PackerTask_ and in ResidueLevelTask_.
    # It is okay becuase since there's only 1 AA packing is fast! (there must be at least one that packs for the packer not to throw an error!)
    # I have made the scorefunction to be empty to make it even faster.
    # Alternatively I have found this function in Pose.cc: replace_residue. You can see how it is
    # used in SaveAndRetrieveSidechains.cc. Dont use it directly. It will not superimpose the residue onto the BB so the residue will just
    # hang in space....

    :param pose: pose to mutate
    :param mutant_aa: the ONE LETTER amino acid (aa) to mutate to.
    :param residues: If specified, only mutate those particular residues. If None: mutate all the residues.
    :param types_not_to_mutate: The 1-letter aminoacids NOT to mutate.
    """


    # create the packer scorefunction
    pack_scorefxn = pyrosetta.rosetta.core.scoring.ScoreFunctionFactory.create_score_function("empty")#pyrosetta.get_score_function("empty")

    task = pyrosetta.rosetta.core.pack.task.TaskFactory().create_task_and_apply_taskoperations(pose)

    # mutation is performed by using a PackerTask with only the mutant amino acid available during design
    if residues == None:
        residues = list(range(1, pose.size() + 1))
    else:
        assert all([resi >= 1 and resi <= pose.size() for resi in residues]), "residues are not within the pose"

    # build which residues are allowed for the pose at particular positions
    for r in range(1, pose.size() + 1):
        one_letter_aa = pose.residue(r).name1()
        # if r not in 'residues' or if r is a type not to mutate keep the original residue type
        if (residues != None and not r in residues) or \
                (types_not_to_mutate != None and one_letter_aa in types_not_to_mutate):
            # original_aa = int(aa_from_oneletter_code(one_letter_aa))
            # aa_bool = pyrosetta.Vector1([aa == original_aa for aa in range(1, 21)])
            # # since the residues stays the same also keep the roatmer
            residuelvltask = task.nonconst_residue_task(r)
            residuelvltask.prevent_repacking()

        # else change it to mutant_aa
        else:
            mutant = int(aa_from_oneletter_code(mutant_aa))
            aa_bool = pyrosetta.Vector1([aa == mutant for aa in range(1, 21)])
            task.nonconst_residue_task(r).restrict_absent_canonical_aas(aa_bool)
        # first gets a residueleveltask and then call restrict.. this restrict the AA
        # todo maybe this can be move under else, and we can move the aa stuff from if?
        #task.nonconst_residue_task(r).restrict_absent_canonical_aas(aa_bool)

    # call the packer to change the aminoacids
    packer = PackRotamersMover(pack_scorefxn, task)
    packer.apply(pose)