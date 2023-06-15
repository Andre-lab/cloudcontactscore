#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for CloudContactScore
@Author: Mads Jeppesen
@Date: 12/6/21
"""

def test_CloudContactScore_on_normalized():
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from simpletestlib.test import setup_test
    from cubicsym.kinematics import randomize_all_dofs
    import pandas as pd
    from cubicsym.paths import DATA
    from cubicsym.cubicsetup import CubicSetup
    from symmetryhandler.reference_kinematics import set_jumpdof_str_str
    df = pd.read_csv("/home/mads/projects/cubicsym/tests/outputs/global_from_mer_native/results.csv")
    df_ = pd.read_csv("/home/mads/projects/cubicsym/tests/outputs/normalization_info_benchmark.csv")
    for pdb_base in df["pdb_base"].unique():
        pdb, base = pdb_base.split("_")
        # if pdb != "7OHF":
        #     continue
        sym = df[df["pdb_base"] == pdb_base]["symmetry"].unique()[0]
        pose, pmm, cmd = setup_test(name=sym, file=pdb, mute=True, return_symmetry_file=False, symmetrize=False)
        symdef = str(DATA) + f"/{sym}/{sym}_{base}_norm.symm"
        cs = CubicSetup(symdef)
        cs.make_symmetric_pose(pose)
        # transfer
        for jump, dof, val in zip(df_[df_["pdb_base"] == pdb_base]["jump"].values, df_[df_["pdb_base"] == pdb_base]["dof"].values, df_[df_["pdb_base"] == pdb_base]["val"].values):
            set_jumpdof_str_str(pose, jump, dof, val)
        css = CloudContactScore(pose, cubicsetup=cs, atom_selection="surface", use_neighbour_ss=False, use_atoms_beyond_CB=False)
        # save poses so that i can look at them
        identifier = f"{pdb}_{css.cubicsetup.get_base()}"
        out = f"/home/mads/projects/cubicsym/tests/outputs/cloudcontact_output_norm/{identifier}.pdb"
        pose.dump_pdb(out)
        out = f"/home/mads/projects/cubicsym/tests/outputs/cloudcontact_output_norm/{identifier}_cloud.pdb"
        css.output_point_cloud_as_pdb(out)
        # TEST 1: test that the point cloud follows the pose as it changes to new positions
        # from shapedesign.src.visualization.visualizer import Visualizer
        # css.show_in_pymol(pose, Visualizer())
        for i in range(10):
            randomize_all_dofs(pose)
            try:
                assert css.pose_atoms_and_cloud_atoms_overlap(pose)
            except:
                raise AssertionError
        print(sym, pdb, "OK!")

def test_CloudContactScore_on_all_fold_kinds():
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from simpletestlib.test import setup_test
    from cubicsym.kinematics import randomize_all_dofs
    from cubicsym.actors.symdefswapper import SymDefSwapper
    sym_files = {
        "T": ["7JRH", "1MOG", "1H0S", "4KIJ", "2VTY"],
        "I": ["1STM", "1B5S", "1NQW", "6S44", "5CVZ"],
        "O": ["5GU1", "3R2R", "1BG7", "1AEW", "3F36"]
    }
    for sym, files in sym_files.items():
        for file in files:
            pose_HF, pmm, cmd, symdef_HF = setup_test(name=sym, file=file, mute=True, return_symmetry_file=True)
            sds = SymDefSwapper(pose_HF, symdef_HF)
            pose_3F, pose_2F = sds.create_3fold_pose_from_HFfold(pose_HF), sds.create_2fold_pose_from_HFfold(pose_HF)
            sds.fold3F_setup.output("/tmp/3fold.symm") #, use_stored_anchor=True)
            sds.fold2F_setup.output("/tmp/2fold.symm") #, use_stored_anchor=True)
            for pose, symdef in zip((pose_HF, pose_3F, pose_2F), (symdef_HF, "/tmp/3fold.symm", "/tmp/2fold.symm")):
                css = CloudContactScore(pose, cubicsetup=symdef, atom_selection="surface", use_neighbour_ss=False, use_atoms_beyond_CB=False)
                # save poses so that i can look at them
                identifier = f"{file}_{css.cubicsetup.get_base()}"
                out = f"/home/mads/projects/cubicsym/tests/outputs/cloudcontact_output/{identifier}.pdb"
                pose.dump_pdb(out)
                out = f"/home/mads/projects/cubicsym/tests/outputs/cloudcontact_output/{identifier}_cloud.pdb"
                css.output_point_cloud_as_pdb(out)
                # TEST 1: test that the point cloud follows the pose as it changes to new positions
                # from shapedesign.src.visualization.visualizer import Visualizer
                # css.show_in_pymol(pose, Visualizer())
                for i in range(10):
                    randomize_all_dofs(pose)
                    try:
                        assert css.pose_atoms_and_cloud_atoms_overlap(pose)
                    except:
                        raise AssertionError
                print(sym, file, "OK!")

def test_CloudContactScore():
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from simpletestlib.test import setup_test
    from cubicsym.kinematics import randomize_all_dofs
    sym_files = {
        "T": ["7JRH", "1MOG", "1H0S", "4KIJ", "2VTY"],
        "I": ["1STM", "1B5S", "1NQW", "6S44", "5CVZ"],
        "O": ["5GU1", "3R2R", "1BG7", "1AEW", "3F36"]
    }
    for sym, files in sym_files.items():
        for file in files:
            pose, pmm, cmd, symdef = setup_test(name=sym, file=file, mute=True, return_symmetry_file=True)
            css = CloudContactScore(pose, cubicsetup=symdef, atom_selection="surface", use_neighbour_ss=False, use_atoms_beyond_CB=False)

            # TEST 1: test that the point cloud follows the pose as it changes to new positions
            from shapedesign.src.visualization.visualizer import Visualizer
            css.show_in_pymol(pose, Visualizer())
            for i in range(10):
                randomize_all_dofs(pose)
                try:
                    assert css.pose_atoms_and_cloud_atoms_overlap(pose)
                except:
                    raise AssertionError
            print(sym, file, "OK!")


def test_css_score():
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from simpletestlib.test import setup_test
    pose, pmm, cmd, symdef = setup_test(name="I", file="1STM", return_symmetry_file=True, mute=False)
    css = CloudContactScore(pose, None)
    score_css = css.score(pose)
    assert score_css < 0

def test_css_other():
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from simpletestlib.test import setup_test
    pose, pmm, cmd, symdef = setup_test(name="I", file="1STM", return_symmetry_file=True, mute=False)
    css = CloudContactScore(pose, cubicsetup=symdef)
    score_css = css.score(pose)
    hf_n = css.hf_neighbours()
    n_clashes = css._compute_clashes()

def test_css_produces_the_same_as_clashchecker():
    from shapedesign.src.movers.clashchecker import ClashChecker
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from simpletestlib.test import setup_test
    pose, pmm, cmd, symdef = setup_test(name="I", file="1STM", return_symmetry_file=True, mute=True)
    css = CloudContactScore(pose, None, use_hbonds=False)  #jump_apply_order=['JUMPHFfold1', 'JUMPHFfold111', 'JUMPHFfold111_z'],
                            # jump_connect_to_chain = "JUMP5fold111_z")

    clc = ClashChecker(pose)
    score_css = css.score(pose)
    score_clc = clc.score(pose)
    print(score_css - score_clc)

def test_cloudcontactscore_time():
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from shapedesign.src.utilities.tests import setup_test
    pose, pmm, cmd, symdef = setup_test(name="4v4m", return_symmetry_file=True, mute=True)
    print("construction time:", timeit.timeit(lambda: CloudContactScore(pose, None), number=1) / 1) # around 15 ms
    css = CloudContactScore(pose, None)
    print("score time:", timeit.timeit(lambda: css.score(pose), number=100) / 100) # around 15 ms


def test_cloudcontactscore_for_1stm():
    from shapedesign.src.utilities.tests import setup_test
    from pyrosetta import init
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    from symmetryhandler.utilities import add_symmetry_as_comment
    from symmetryhandler.symmetryhandler import SymmetrySetup
    from shapedesign.src.movers.randompertubmover import RandomMover

    init("-symmetry:initialize_rigid_body_dofs true -out:file:output_pose_energies_table false -pdb_comments")
    # pmm, cmd = setup_pymol_server()
    pose, symdef = setup_test(name="1stm", pymol=False, return_symmetry_file=True)
    # pose = pose_from_file("/home/niels/ft_np/4merise/lebar_p_mads.pdb")
    # symdef = "/home/mads/projects/symmetryhandler/tests/out/c4_new.symmdef"
    SetupForSymmetryMover(symdef).apply(pose)
    symmetrysetup = SymmetrySetup()
    symmetrysetup.read_from_file(symdef)
    add_symmetry_as_comment(pose, symdef)
    rndm = RandomMover(pose, scorefunction= {"cloudcontactscore": {"jump_apply_order": ["JUMP5fold1", "JUMP5fold111", "JUMP5fold1111"],
                                                                   "jump_connect_to_chain": "JUMP5fold1111_subunit",
                                                                   "chain_ids_in_use": [1, 2, 3, 8, 7, 6]}},
                       symmetrysetup = symmetrysetup,
                       dofs= {
                           "JUMP5fold1": {
                               "z": {"lower": 1, "upper": 10},
                               "angle_z": {"lower": 5, "upper": 10}
                               # "angle_z": { "lower": "max", "upper": "max"}
                           },
                           "JUMP5fold111": {
                               "x": {"lower": 5, "upper": 50},
                           },
                           "JUMP5fold1111": {
                               "angle_x": {"lower": 10, "upper": 100},
                               "angle_y": {"lower": 10, "upper": 100},
                               "angle_z": {"lower": 10, "upper": 100}
                           }
                       },
                       visualizer=None, # Visualizer(),
                       )
    Visualizer()
    for i in range(10):
        rndm.apply(pose)
    rndm.sfxn.show_in_pymol(pose, show_clashes=False, dict_to_visualizer={"name": f"pose_in_css", "reinitialize": False, "store_scenes": False})
    for i in range(10):
        rndm.apply(pose)
    rndm.sfxn.show_in_pymol(pose, show_clashes=False, dict_to_visualizer={"name": f"pose_in_css2", "reinitialize": False, "store_scenes": False})
    ...

def test_cloudcontactscore_for_4mer():
    from pyrosetta import pose_from_file, init
    from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
    from symmetryhandler.utilities import add_symmetry_as_comment
    from symmetryhandler.symmetryhandler import SymmetrySetup
    from shapedesign.src.movers.randompertubmover import RandomMover

    init("-symmetry:initialize_rigid_body_dofs true -out:file:output_pose_energies_table false -pdb_comments")
    # pmm, cmd = setup_pymol_server()
    pose = pose_from_file("/home/niels/ft_np/4merise/lebar_p_mads.pdb")
    symdef = "/home/mads/projects/symmetryhandler/tests/out/c4_new.symmdef"
    SetupForSymmetryMover(symdef).apply(pose)
    symmetrysetup = SymmetrySetup()
    symmetrysetup.read_from_file(symdef)
    add_symmetry_as_comment(pose, symdef)
    rndm = RandomMover(pose, scorefunction= {"cloudcontactscore": {"jump_apply_order": ["JUMP1", "JUMP11"],
                                                                   "jump_connect_to_chain": "JUMP111",
                                                                   "chain_ids_in_use": [1, 2, 3]}},
                       symmetrysetup = symmetrysetup,
                       dofs= {
                           "JUMP1": {
                               "x": {"lower": 0, "upper": 5},
                           },
                           "JUMP11":{
                               "angle_x": {"lower": 50, "upper": 100},
                               "angle_y": {"lower": 50, "upper": 100},
                               "angle_z": {"lower": 50, "upper": 100}
                           }
                       },
                       visualizer=None,  # Visualizer(),
                       )
    Visualizer()
    for i in range(10):
        rndm.apply(pose)
    rndm.sfxn.show_in_pymol(pose, show_clashes=False,
                            dict_to_visualizer={"name": f"pose_in_css", "reinitialize": False, "store_scenes": False})
    for i in range(10):
        rndm.apply(pose)
    rndm.sfxn.show_in_pymol(pose, show_clashes=False,
                            dict_to_visualizer={"name": f"pose_in_css2", "reinitialize": False, "store_scenes": False})
    ...

    # css = CloudContactScore(pose, jump_apply_order=['JUMP1', "JUMP11"],
    #                         jump_connect_to_chain="JUMP111",  chain_ids_in_use=[1, 2, 3])#, symdef=symdef)
    # css.score(pose)
    # add_symmetry_as_comment(pose, symdef)
    # css.show_in_pymol(pose, dict_to_visualizer={"name": "init"})
    # # for i in range(5):
    # #     perturb_jumpdof(pose, "JUMP1", 1, 10)
    # #     css.show_in_pymol(pose, dict_to_visualizer={"name": f"trans_{i}", "reinitialize": False})
    # # for i in range(2):
    # #     perturb_jumpdof(pose, "JUMP11", 4, 100)
    # #     css.show_in_pymol(pose, dict_to_visualizer={"name": f"rotx{i}", "reinitialize": False})
    # # for i in range(2):
    # #     perturb_jumpdof(pose, "JUMP11", 5, 100)
    # #     css.show_in_pymol(pose, dict_to_visualizer={"name": f"roty{i}", "reinitialize": False})
    # # for i in range(2):
    # #     perturb_jumpdof(pose, "JUMP11", 6, 100)
    # #     css.show_in_pymol(pose, dict_to_visualizer={"name": f"rotz{i}", "reinitialize": False})
    # for i in range(2):
    #     perturb_jumpdof(pose, "JUMP1", 1, -5)
    #     perturb_jumpdof(pose, "JUMP11", 4, -100)
    #     perturb_jumpdof(pose, "JUMP11", 5, -100)
    #     perturb_jumpdof(pose, "JUMP11", 6, -100)
    #     # css.score(pose)
    #     css.show_in_pymol(pose, dict_to_visualizer={"name": f"combo{i}", "reinitialize": False})

def test_cloudcontactscore():
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from shapedesign.src.utilities.tests import setup_test
    pose, pmm, cmd, symdef = setup_test(name="4v4m", return_symmetry_file=True, mute=True)
    css = CloudContactScore(pose, None)
    css.score(pose)
    css.show_in_pymol(pose)

def test_we_have_low_energies():
    from shapedesign.benchmark.src.prepare_benchmark import get_benchmark_proteins
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from shapedesign.src.utilities.tests import setup_test
    from shapedesign.benchmark.src.benchmarkhandler import BenchmarkHandler
    from pathlib import Path
    proteins = get_benchmark_proteins("", BenchmarkHandler().db)[0] # recovery inputs
    for protein in proteins:
        pose, symdef = setup_test(name=str(Path(protein).stem), return_symmetry_file=True, mute=True, pymol=False)
        css = CloudContactScore(pose, None, clash_dist={"CB": 1.65})
        print(protein, css.breakdown_score(pose))
        assert css.clash_score() == - 0.001, "no clashes should happen for the native structures"

def make_mc_mover(pose, symdef):
    from shapedesign.src.movers.mcdockmover import MCDockMover
    from shapedesign.src.movers.randompertubmover import RandomMover
    from symmetryhandler.symmetryhandler import SymmetrySetup
    ss = SymmetrySetup()
    ss.read_from_file(symdef)
    dofs = {"JUMP5fold1": {
        "z": {"lower": -5, "upper": 5},
        "angle_z": {"lower": -20, "upper": 20}
        # "angle_z": { "lower": "max", "upper": "max"}
    },
        "JUMP5fold111": {
            "x": {"lower": -5, "upper": 5},
        },
        "JUMP5fold1111": {
            "angle_x": {"lower": -20, "upper": 20},
            "angle_y": {"lower": -20, "upper": 20},
            "angle_z": {"lower": -20, "upper": 20}
        }}
    rndmover = RandomMover(pose, ss, dofs)
    mc_ref = {"JUMP5fold1": {
        "z": {"param1": 3},
        "z_angle": {"limit_movement": True, "param1": 3},
    },
        "JUMP5fold111": {
            "x": {"param1": 3},
        },
        "JUMP5fold1111": {
            "x_angle": {"param1": 5},
            "y_angle": {"param1": 5},
            "z_angle": {"param1": 5}}}
    # vis = Visualizer(store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False)
    ref = {"cloudcontactscore": {"symdef": symdef}}
    mcm = MCDockMover(pose.clone(), ss, ref, dofs=mc_ref, iterations=5000)
    return rndmover, mcm

def test_we_cant_find_other_low_energies_around_native_states():
    from shapedesign.benchmark.src.prepare_benchmark import get_benchmark_proteins
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from shapedesign.src.utilities.tests import setup_test
    from shapedesign.src.utilities.pose import CA_rmsd_without_alignment
    proteins = get_benchmark_proteins()[0]  # recovery inputs
    for protein in proteins:
        # pose, pmm, cmd, symdef = setup_test(name=protein, return_symmetry_file=True, mute=True, pymol=False)
        pose, symdef = setup_test(name=protein, return_symmetry_file=True, mute=True, pymol=False)
        css = CloudContactScore(pose, cubicsetup=symdef)
        rnm, mcm = make_mc_mover(pose, symdef)
        init_pose = pose.clone()
        mcm.apply(pose)
        rmsd = CA_rmsd_without_alignment(css.get_asymmetric_pose(pose), css.get_asymmetric_pose(init_pose))
        assert rmsd < 0.1

def test_cloudcontactscore_moves():
    from cloudcontactscore.cloudcontactscore import CloudContactScore
    from shapedesign.src.utilities.tests import setup_test
    pose, pmm, cmd, symdef = setup_test(name="4v4m", return_symmetry_file=True, mute=True)
    css = CloudContactScore(pose, cubicsetup=symdef)
    css.show_in_pymol(pose, show_clashes=False, dict_to_visualizer = {"name": "css_before", "reinitialize": True,
                                                "store_scenes":False, "store_states":True, "representation":["cartoon", "sticks"]})
    # ['JUMP5fold1111', 'JUMP5fold111', 'JUMP5fold1']
    visa = Visualizer(name="actual", store_scenes=False, store_states=True, representation=["cartoon", "sticks"],  reinitialize=False)
    perturb_jumpdof(pose, 'JUMP5fold1', 3, 180)
    perturb_jumpdof(pose, 'JUMP5fold1', 6, -96)
    set_jumpdof(pose, 'JUMP5fold111', 1, 120)
    perturb_jumpdof(pose, 'JUMP5fold1111', 4, 120)
    perturb_jumpdof(pose, 'JUMP5fold1111', 5, -50)
    perturb_jumpdof(pose, 'JUMP5fold1111', 6, 170)
    # visa.send_pose(pose)
    css.show_in_pymol(pose, show_clashes=False, dict_to_visualizer={"name": "css_after", "reinitialize": False,
            "store_scenes": False, "store_states": True, "representation": ["cartoon", "sticks"]})
    # perturb_jumpdof(pose, 'JUMP5fold1', 6, 5)
    # css.show_in_pymol(pose, show_clashes=False, dict_to_visualizer={"name": "css_after2", "reinitialize": False,
    #                                                                 "store_scenes": False, "store_states": True, "representation": ["cartoon", "sticks"]})

    perturb_jumpdof(pose, 'JUMP5fold1', 3, 15)
    perturb_jumpdof(pose, 'JUMP5fold1', 6, 15)
    set_jumpdof(pose, 'JUMP5fold111', 1, -15)
    perturb_jumpdof(pose, 'JUMP5fold1111', 4, -15)
    perturb_jumpdof(pose, 'JUMP5fold1111', 5, 15)
    perturb_jumpdof(pose, 'JUMP5fold1111', 6, -15)
    # visa.send_pose(pose)
    css.show_in_pymol(pose, show_clashes=False, dict_to_visualizer={"name": "css_after_2", "reinitialize": False,
                                                                    "store_scenes": False, "store_states": True, "representation": ["cartoon", "sticks"]})

def test_css_with_docking():
    from shapedesign.src.utilities.tests import setup_test
    pose, pmm, cmd, symdef = setup_test(name="4v4m", return_symmetry_file=True, mute=True)
    from shapedesign.src.movers.mcdockmover import MCDockMover
    from shapedesign.src.movers.randompertubmover import RandomMover
    from symmetryhandler.symmetryhandler import SymmetrySetup
    ss = SymmetrySetup()
    ss.read_from_file(symdef)
    dofs ={"JUMP5fold1": {
            "z": { "lower": -5, "upper": 5},
            "angle_z": { "lower": -20, "upper":20}
            # "angle_z": { "lower": "max", "upper": "max"}
            },
            "JUMP5fold111": {
                "x": {"lower": -5, "upper": 5},
            },
            "JUMP5fold1111": {
                "angle_x": {"lower": -20, "upper": 20},
                "angle_y": {"lower": -20, "upper": 20},
                "angle_z": {"lower": -20, "upper": 20}
            }}
    Visualizer(name="native", store_scenes=False, store_states=True, representation=["cartoon"], chain_color="grey",
               color_by_chains=False).send_pose(pose)
    RandomMover(pose, ss, dofs).apply(pose)
    Visualizer(name="start", store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False).send_pose(pose)

    mc_ref = {"JUMP5fold1": {
            "z": {"param1":3},
            "z_angle": {"limit_movement": True, "param1":3},
    },
        "JUMP5fold111": {
            "x": {"param1":3},
        },
        "JUMP5fold1111": {
            "x_angle": {"param1":5},
            "y_angle": {"param1":5},
            "z_angle": {"param1":5}}}

    vis = Visualizer(store_scenes=False, store_states=True, representation=["cartoon"], reinitialize=False)
    ref =  {"cloudcontactscore": {"clash_dist": {"CB": 1.6}, "lj_overlap":10 }}
    mcm = MCDockMover(pose.clone(), ss, ref, dofs=mc_ref, iterations=5000, visualizer=vis, visual_update_interval=50, fixed_pose_name="css")
    mcm.apply(pose)
    #
    # ref =  {"clashchecker": {}}
    # mcm = MCDockMover(pose.clone(), ss, ref, dofs=mc_ref, iterations=5000, visualize=vis, visual_update_interval=50, fixed_pose_name="clash")
    # mcm.apply(pose)

    # print("with symmetry: ", timeit.timeit(lambda: move_and_apply_score(pose, css.score, apply_symweights=True), number=100))
    # print("without symmetry", timeit.timeit(lambda: move_and_apply_score(pose, css.score, apply_symweights=False), number=100))
    #
    # from shapedesign.src.utilities.score import create_sfxn_from_terms, create_score_from_name
    # hbond_score = create_sfxn_from_terms(terms =("hbond_sr_bb", "hbond_lr_bb"), weights=(1,1))
    #
    # ref2015 = create_score_from_name("ref2015")
    # interchain_cen = create_score_from_name("interchain_cen")
    #
    # print("hbond", timeit.timeit(lambda: move_and_apply_score(pose, hbond_score), number=100))
    # print("ref2015", timeit.timeit(lambda: move_and_apply_score(pose, ref2015), number=100))
    #
    # from shapedesign.src.movers.clashchecker import ClashChecker
    # clashchecker = ClashChecker(pose, clash_distance=3.5, lj_percentage_overlap=25)
    #
    # print("clashchecker: ", timeit.timeit(lambda: move_and_apply_score(pose, clashchecker.score), number=100))
    #
    # # change
    # switch = SwitchResidueTypeSetMover("centroid")
    # switch.apply(pose)
    # print("interchain_cen", timeit.timeit(lambda: move_and_apply_score(pose, interchain_cen), number=100))
