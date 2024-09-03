
def test_cyclical_symmetry():
    from symmetryhandler.symmetrysetup import SymmetrySetup
    from symmetryhandler.util import randomize_dofs
    import math
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from pyrosetta import pose_from_file, init
    from simpletestlib.setup import setup_pymol
    init("-symmetry:initialize_rigid_body_dofs true -pdb_comments")

    pmm, cmd = setup_pymol(return_cmd=True)

    # Cyclical symmetry
    pose = pose_from_file("cyclical/input_ref_INPUT.pdb")
    symdef = "cyclical/input_ref.symm"
    ss = SymmetrySetup(symdef=symdef)
    ss.make_symmetric_pose(pose)
    ss.visualize(ip="10.8.0.10")
    ccs = CloudContactScore(pose, ss)
    score = ccs.score(pose)
    assert math.isclose(score, -476, abs_tol=1)

    # check that is okay when we move them around
    for i in range(10):
        randomize_dofs(pose, 100)
        try:
            assert ccs.pose_atoms_and_cloud_atoms_overlap(pose)
        except:
            raise AssertionError

    # Visualization
    pmm.apply(pose)
    ccs.output_point_cloud_as_pdb("output/cyclical_points.pdb")
    cmd.load("/Users/mads/mounts/mailer/home/mads/projects/cloudcontactscore/tests/output/cyclical_points.pdb")
    cmd.do(f"show spheres, cyclical_points")
    cmd.do(f"set sphere_scale, 0.2")