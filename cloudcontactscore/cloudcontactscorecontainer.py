#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen
@Date: 1/26/23
"""
from cubicsym.cubicmontecarlo import CubicMonteCarlo
from pyrosetta.rosetta.core.pose.datacache import CacheableDataType
from cloudcontactscore.cloudcontactscore import CloudContactScore
from cloudcontactscore.util import pickle_dump_cloudcontactscore, pickle_load_cloudcontactscore
from cubicsym.dofspec import DofSpec
from cubicsym.cubicsetup import CubicSetup
from cubicsym.actors.symdefswapper import SymDefSwapper
import tempfile
import uuid

class CloudContactScoreContainer:
    """A container of CloudContactScore objects that can function as a score function for 1 or multiple different poses with different
    backbones.

    Details
    ------------------------
    It stores a specific CloudContactScore object inside a specific CubicMonteCarlo object that points to a specific pose. If using this
    class for multiple backbones the pointer is based on an CacheableStringMap with they key 'id' that must stored in the pose datacache.
    If using this class for multiple backbones you must mark the specific poses with specific ids. If you are not using this for multiple
    backbones there's no need for marking the pose. It will treat a pose as the same even though it has changed symmetry with between
    HF, 3F and 2F based symmetries.

    Using it in practice:
    ------------------------
    # If using it for multiple backbones use cubicsym.utilities.add_id_to_pose to add an id to your different poses
    from cubicsym.utilities import add_id_to_pose
    add_id_to_pose(pose, id="1")

    # Instantiate the class object
    ccsc = CloudContactScoreContainer(...)

    # This must always be called before applying a pose to the object as this matches a particular CloudContactScore
    # and CubicMonteCarlo object to that pose through its 'id'. It makes sure that when using ccsc.cmc and ccsc.ccs they corresponds
    # to the correct backbone of the pose.
    ccsc.set_ccs_and_cmc(pose)

    Example for a sliding protocol
    ------------------------
    # If using it in a sliding protocol you can work on the CloudContactScore directly through the ccsc.ccs variable
    # do slide move
    ccsc.ccs.number_of_clashes(pose)
    # do slide move
    ccsc.ccs.number_of_clashes(pose)
    # do slide move
    ccsc.ccs.number_of_clashes(pose)
    # ...

    Example for a docking protocol
    ------------------------
    # If using it in a docking protocol and you want to use it a MonteCarlo Fashion:
    # First reset the CubicMonteCarlo object:
    ccsc.cmc.reset(pose)
    # Then do rigid body moves and when ready to evaluate the energy do:
    ccsc.cmc.apply(pose)
    # Finally when you have exhausted all your rigid body moves call this to get the lowest scored pose:
    ccsc.cmc.recover_lowest_scored_pose(pose)
    """

    def __init__(self, pose, cubicsetup: CubicSetup, low_memory=False, verbose=False, **cloudcontactscore_param):
        """Instantiate an object of the class"""
        self.sds = SymDefSwapper(pose, cubicsetup)
        self.ccs = None
        self.cmc = None
        self.pointer_to_ccs, self.pointer_to_cmc = {}, {}
        self.low_memory = low_memory
        self.dir_ref = f"{tempfile.gettempdir()}/{uuid.uuid4()}"
        self.verbose = verbose
        self.cloudcontactscore_param = cloudcontactscore_param

    def get_cubic_setup_from_base(self, base):
        """Returns the cubicsetup that corresponds to the base"""
        if base == "HF":
            return self.sds.foldHF_setup
        elif base == "3F":
            return self.sds.fold3F_setup
        elif base == "2F":
            return self.sds.fold2F_setup

    def construct_ccs(self, pose):
        """Constructs a CloudContactScore object based on the pose."""
        base = CubicSetup.get_base_from_pose(pose)
        cubicsetup = self.get_cubic_setup_from_base(base)
        ccs = CloudContactScore(pose=pose, cubicsetup=cubicsetup,
                                use_atoms_beyond_CB=False, use_neighbour_ss=False, **self.cloudcontactscore_param)
        return ccs

    def construct_cmc(self, pose, ccs):
        """Constructs a CubicMonteCarlo object based on the pose and the CloudContactScore"""
        cmc = CubicMonteCarlo(scorefunction=ccs, dofspec=DofSpec(pose))
        return cmc

    def get_pose_id(self, pose):
        """Returns the pose id stored in the datacache."""
        if pose.data().has(CacheableDataType.ARBITRARY_STRING_DATA):
            stringmap = pose.data().get_ptr(CacheableDataType.ARBITRARY_STRING_DATA)
            id_ = stringmap.map()["id"]
        else:
            raise ValueError("The pose should have an id")
        return id_

    def set_ccs_and_cmc_low_memory(self, pose, full_id):
        """Constructs ccs (a CloudContactScore object) inside cmc (a CubicMonteCarlo object) in low memory mode,
        meaning the CloudContactScore is stored on disk and a CubicMonteCarlo object instance is created from that
        each time when loaded in. In this case there will only be 1 instance of each in memory at any given time."""
        if full_id in self.pointer_to_ccs:
            # pickle load the cloudcontactscore object from file and create a new CubicMonteCarlo object from
            # we only have 1 instance of the CloudContactScore object in memory in this case
            self.ccs = pickle_load_cloudcontactscore(pose, self.pointer_to_ccs[full_id])
            self.cmc = CubicMonteCarlo(scorefunction=self.ccs, dofspec=DofSpec(pose))
        else:
            if self.verbose:
                print(f"Constructing ccs and cmc for id: {full_id} and stores ccs on disk.")
            ccs_path = self.dir_ref + "_ccs_" + full_id
            self.ccs = self.construct_ccs(pose)
            pickle_dump_cloudcontactscore(self.ccs, ccs_path)
            self.pointer_to_ccs[full_id] = ccs_path

    def set_ccs_and_cmc_high_memory(self, pose, full_id):
        """Constructs ccs (a CloudContactScore object) inside cmc (a CubicMonteCarlo object) in high memory mode,
        meaning the different CloudContactScore and CubicMonteCarlo objects are always stored in memory. This makes
        it more effecient, but also more memory consuming."""
        if full_id in self.pointer_to_ccs:
            self.ccs = self.pointer_to_ccs[full_id]
            self.cmc = self.pointer_to_cmc[full_id]
        else:
            if self.verbose:
                print(f"Constructing ccs and cmc for id: {full_id} and stores them both in memory")
            self.ccs = self.construct_ccs(pose)
            self.cmc = self.construct_cmc(pose, self.ccs)
            self.pointer_to_ccs[full_id] = self.ccs
            self.pointer_to_cmc[full_id] = self.cmc

    def set_ccs_and_cmc(self, pose):
        """Constructs ccs (a CloudContactScore object) inside cmc (a CubicMonteCarlo object) that will be used for the given pose."""
        # Get the id in the pose datacache if it exists
        full_id = self.get_pose_id(pose)
        # If no id is in the pose datacache then construct new instances of a CloudContactScore and CubicMonteCarlo object
        # this is either done in low memory mode or not. In the low memory mode the CloudContactScore is stored on
        # disk and a CubicMonteCarlo object instance is created from that each time when loaded in.
        # In regular mode both the CloudContactScore and CubicMonteCarlo object is stored in memory.
        if self.low_memory:
            self.set_ccs_and_cmc_low_memory(pose, full_id)
        else:
            self.set_ccs_and_cmc_high_memory(pose, full_id)


