#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: mads
@Date: 2/14/23 
"""


# import numpy as np
# a =  np.random.random((1000, 1000)) # , dtype=float)
# from pyrosetta import pose_from_file, init
# init("-initialize_rigid_body_dofs")
# pose = pose_from_file("/Users/mads/projects/cloudcontactscore/tests/input/pdbs/2CC9.cif")
# symdef = "/Users/mads/projects/cloudcontactscore/tests/input/pdbs/2CC9.symm"
# SetupForSymmetryMover(symdef).apply(pose)
def test_cloudcontactscorecontainer_low_memory():
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from cloudcontactscore.cloudcontactscorecontainer import CloudContactScoreContainer
    from simpletestlib.test import setup_test
    from cubicsym.kinematics import randomize_all_dofs
    import pandas as pd
    from cubicsym.paths import DATA
    from cubicsym.cubicsetup import CubicSetup
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    from cubicsym.utilities import add_id_to_pose_w_base
    pose, pmm, cmd, symdef = setup_test(name="T", file="2CC9", mute=True, return_symmetry_file=True, symmetrize=True)


    cs = CubicSetup(symdef=symdef)
    csss = []
    # for i in range(100):
    css = CloudContactScoreContainer(pose, symdef, low_memory=True)
    add_id_to_pose_w_base(pose, id_="test")
    css.set_ccs_and_cmc(pose)
    csss.append(css)