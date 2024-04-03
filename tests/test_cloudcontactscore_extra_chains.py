
def test_CloudContactScore_with_extra_chains():
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from simpletestlib.setup import setup_test
    from cubicsym.kinematics import randomize_all_dofs
    from cubicsym.cubicsetup import CubicSetup
    from proteinshapedesign.posevisualizer import PoseVisualizer
    sym = "I"
    for pdbid, hand in {'1STM': True, '1NQW': False, '1B5S': True, '6S44': False}.items():
        pose, pmm, cmd, symdef = setup_test(name=sym, file=pdbid, mute=True, return_symmetry_file=True)
        pmm.apply(pose)
        cs = CubicSetup(symdef)
        pose = cs.make_asymmetric_pose(pose)
        cs_new = cs.add_extra_chains()
        cs_new.make_symmetric_pose(pose)
        css = CloudContactScore(pose, cubicsetup=cs_new, atom_selection="surface", use_neighbour_ss=False, use_atoms_beyond_CB=False,
                                extra_chain_interaction_disfavored=True)
        css.score(pose)
        for i in range(100):
            randomize_all_dofs(pose)
            try:
                assert css.pose_atoms_and_cloud_atoms_overlap(pose)
            except:
                raise AssertionError
        print(sym, "OK!")

        css = CloudContactScore(pose, cubicsetup=cs_new, atom_selection="surface", use_neighbour_ss=False,
                                use_atoms_beyond_CB=False)
        css.score(pose)
        for i in range(100):
            randomize_all_dofs(pose)
            try:
                assert css.pose_atoms_and_cloud_atoms_overlap(pose)
            except:
                raise AssertionError
        print(sym, "OK!")
