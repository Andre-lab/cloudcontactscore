#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests CloudContactScore on Octahedral symmetric structures
@Author: Mads Jeppesen
@Date: 10/27/22
"""

def test_css_on_O():
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from shapedesign.src.visualization.visualizer import Visualizer
    from simpletestlib.test import setup_test
    pose, pmm, cmd, symdef = setup_test(name="O", file="3LVX", return_symmetry_file=True, mute=True)
    css = CloudContactScore(pose, symdef=symdef, atom_selection="surface", use_atoms_beyond_CB=False)
    score_css = css.score(pose)
    css.show_in_pymol(pose, Visualizer())
    assert score_css > 1