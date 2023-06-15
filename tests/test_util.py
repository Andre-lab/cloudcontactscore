#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen 
@Date: 2/15/23 
"""

def test_pickle_cloudcontactscore():
    from cloudcontactscore.util import pickle_dump_cloudcontactscore, pickle_load_cloudcontactscore
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from simpletestlib.test import setup_test
    from cubicsym.cubicsetup import CubicSetup
    import numpy as np

    pose, pmm, cmd, symdef = setup_test(name="T", file="2CC9", mute=True, return_symmetry_file=True, symmetrize=True)
    css = CloudContactScore(pose, cubicsetup=CubicSetup(symdef))
    file = "output/ccs.pickle"
    pickle_dump_cloudcontactscore(css, file)
    css_new = pickle_load_cloudcontactscore(pose, file)

    # now assert they are the same
    assert all([np.isclose(vi, vj).all() for (ki, vi), (kj, vj) in zip(css.point_clouds.items(), css_new.point_clouds.items())])
    assert css.point_cloud_size == css_new.point_cloud_size
    assert np.isclose(css.main, css_new.main).all()
    assert np.isclose(css.rest, css_new.rest).all()
    assert css.matrix_shape == css_new.matrix_shape
    assert np.isclose(css.neighbour_mask, css_new.neighbour_mask).all()
    assert np.isclose(css.donor_acceptor_mask, css_new.donor_acceptor_mask).all()
    assert np.isclose(css.symmetric_wt, css_new.symmetric_wt).all()
    assert np.isclose(css.coordination_wt, css_new.coordination_wt).all()
    assert np.isclose(css.masked_symmetric_coordination_wt, css_new.masked_symmetric_coordination_wt).all()
    assert np.isclose(css.masked_coordination_wt, css_new.masked_coordination_wt).all()
    assert np.isclose(css.clash_limit_matrix, css_new.clash_limit_matrix).all()
    assert css.symmetry_type == css_new.symmetry_type
    assert css.symmetry_base == css_new.symmetry_base
    assert css.use_atoms_beyond_CB == css_new.use_atoms_beyond_CB
    assert css.chain_names_in_use == css_new.chain_names_in_use
    assert css.core_atoms_str == css_new.core_atoms_str
    assert css.core_atoms_index == css_new.core_atoms_index
    assert css.no_clash == css_new.no_clash
    assert np.isclose(css.distances, css_new.distances).all()
    assert css.apply_symmetry_to_score == css_new.apply_symmetry_to_score
    assert css.neighbour_dist == css_new.neighbour_dist
    assert css.neighbour_anchorage == css_new.neighbour_anchorage
    assert css.neighbour_ss == css_new.neighbour_ss
    assert css.dssp == css_new.dssp
    assert css.clash_penalty == css_new.clash_penalty
    assert css.use_hbonds == css_new.use_hbonds
    assert css.atom_selection == css_new.atom_selection
    assert css.clash_dist_str == css_new.clash_dist_str
    assert css.clash_dist_int == css_new.clash_dist_int
    assert css.sym_ri_ai_map == css_new.sym_ri_ai_map
    assert css.connected_jumpdof_map == css_new.connected_jumpdof_map
