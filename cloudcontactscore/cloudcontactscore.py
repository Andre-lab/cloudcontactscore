#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CloudContactScore class
@Author: Mads Jeppesen
@Date: 12/6/21
"""
import math
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory, ScoreTypeManager
from cloudcontactscore.modify_aa import build_new_aa_repr
import numpy as np
from io import StringIO
from itertools import cycle
from pyrosetta import AtomID, Pose, Vector1
from pyrosetta.rosetta.core.pose.symmetry import sym_dof_jump_num, jump_num_sym_dof, extract_asymmetric_unit
from pyrosetta.rosetta.core.scoring.sasa import SasaCalc
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from symmetryhandler.reference_kinematics import set_jumpdof_int_int, get_jumpdof_int_int
from pyrosetta.rosetta.core.scoring.dssp import Dssp
from symmetryhandler.utilities import get_updated_symmetry_file_from_pose, get_symmetry_file_from_pose
from symmetryhandler.reference_kinematics import set_all_dofs_to_zero
from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover
from pyrosetta.rosetta.core.pose.datacache import CacheableDataType
from pyrosetta.rosetta.core.select.util import SelectResiduesByLayer
from cubicsym.cubicsetup import CubicSetup
from cubicsym.utilities import get_chain_map_as_dict

class CloudContactScore:
    """Fast score function for fixed backbone coarse grained docking or design of cubic symmetrical proteins.

    General
    -----------------------------
    CloudContactScore is meant to be used on Rosetta Pose objects but does not inherit from the ScoreFunction class in Rosetta and can
    therefor not be used in place of a regular Rosetta score function. All the features of the CloudContactScore score function is set
    during the initialization of it (see __init__() below). Following this, each invocation of the 'score()' method will calculate the score
    of the symmetrical pose.

    Usage example
    -----------------------------
    ccs = CloudContactScore(pose)
    # move the pose to a new position
    ccs.score(pose)
    # move the pose to a new position
    ccs.score(pose)

    Details
    -----------------------------
    The score function has 5 components that either gives penalties (+) or bonuses (-) to the score:
    1. Clash penalty based on the distance between atoms.
    2. Neighbour bonus based on the distance between CB (if residue!=GLY) or CA (if residue=GLY) atoms.
    3. Backbone hydrogen bond bonuses for correct orientation of hydrogen bonds.
    4. Secondary structure interaction bonus. In DSSP terminology: L-interactions are worse than H- or E-interactions.
    5. Anchorage interaction bonus. Residues that are anchored to the subunit (calculated through internal interactions) gives higher bonus.
    Each of these score term contributions can be turned on and off and their weight/behavior modified.
    See "Score term details" below for more information

    CloudContactScore uses a point cloud representation where each points in the cloud represents a particular portion of a protein.
    By default (atom_selection='surface') this is the surface of protein without the presence of surface side chains beyond CB atoms. This
    cloud is stored as numpy arrays and moved around in space in relation to the internal symmetry change in a fixed backbone pose.
    The point cloud is stored during initialization of the mover and any change to backbone of the pose afterwards will not be recognized
    by CloudContactScore.

    CloudContactScore can also calculate the amount of what I call 'High Fold' interactions or HF-interactions. In this definition the
    'high folds' are the symmetry folds present in the symmetry file with the highest order. For I-symmetry it is the 5-fold and for O-symmetry
    it is the 4-fold and so on. The interactions I refer to are the interactions between the main/master subunit and other subunits that
    are part of the other high-folds that the main/master subunit is NOT part of. This is useful to check the connections of the cubic
    symmetrical structures as they must connect in order for a cubic symmetrical structure to form.

    Score term details
    -----------------------------
    1. To calculate the clash penalty between two atoms the identity of atoms and their Lennard Jones (LJ) sphere is taken into account.
       For all non Oxygen-Nitrogen interactions ("NO-interactions") a clash is detected if their LJ-sphere overlaps
       by a set overlap percentage (20% by default). NO-interactions can get closer than this and is set by another parameter (1.2 Å by default).
       The penalty is given as on/off so there is no gradient involved. The LJ-sphere uses the Rosetta definition. Any heavy atoms beyond CB
       in the side chains, including CB itself will have their clash distance set to 1.5 Å by default (originally CB ≈ 1.61) to allow the
       backbones to get closer.
    2. To calculate neighbour bonuses, each connection between CB (if residue!=GLY) or CA (if residue=GLY) between the main/master subunit
       and opposing subunits that are within a certain distance (12 Å by default) adds -1 to the neighbour score.
    3. The hydrogen bond score is done through the "hbond_sr_bb" and "hbond_lr_bb" score terms in Rosetta.
    4. The secondary structure is added as a weight through the neighbour score. If the neighbour score is not turn on neither will this
       contribution. If the potential neighbour is either E or H (in DSSP terminilogy) the neighbour score is weighted higher than if it is
       L.
    5. The anchorage contribution also happens through the neighbour score. If the neighbour score is not turn on neither will this
       contribution. The higher the internal neighbour count the higher the weight of the score.
    The total score will, if set of course, be reweighed by the E-line in the associated symmetry file as per usual in Rosetta.

    Recommendations
    -----------------------------
    For docking:
    use_atoms_beyond_CB=False  *This removes any rotamer information from the pose.
    use_neighbour_ss=False

    For design:
    use_atoms_beyond_CB=True
    use_neighbour_ss=True
    """

    def __init__(self, pose, cubicsetup, atom_selection="surface", clash_dist: dict = None, neighbour_dist=12, no_clash=1.2,
                 use_neighbour_anchorage=True, use_neighbour_ss=True, apply_symmetry_to_score=True, clash_penalty=100000, use_hbonds=True,
                 interaction_bonus: dict = None, lj_overlap=20, use_atoms_beyond_CB=True, verbose=False):
        """Initialization of a CloudContactScore object.

        :param pose: Pose to use for scoring. CloudContactScore will use the main/master subunit to create the internal point cloud.
        :param cubicsetup: CubicSetup associated with the cubic symmetrical pose.
        :param atom_selection: Selection method to use when selecting the point cloud. Options are 'surface', 'core' and 'all'. 'surface' is
               highly recommend as the others are not well tested.
        :param clash_dist: Clash distances to use instead of the default ones. 
        :param neighbour_dist: Neighbour distance to use. If 2 atoms are within this distance they are considered neighours. 
        :param no_clash: The Nitrogen-Oxygen clash distance to use.
        :param use_neighbour_anchorage: Use neighbour anchorage interaction bonus in the score.
        :param use_neighbour_ss: Use nieghour secondary structure interaction bonus in the score
        :param apply_symmetry_to_score: Apply symmetry weights to the score.
        :param clash_penalty: The score added to final score for each clash detected in the pose
        :param use_hbonds: Use hydrogen bond score in the final score.
        :param interaction_bonus: Add interaction bonuses between different subunits. Useful if you want to design specific interactions.
               The keys of the dictionary are the Pose chain numbers and the values the weights. Example: {2: 1, 3: 1, 6: 3, 7:1, 8: 3}
        :param lj_overlap: LJ_overlap to use when calculating the default clash distances.
        :param use_atoms_beyond_CB: Use atoms beyond CB in the point cloud for residues not designated as surface.
        :param verbose: Output information about what is going on
        """
        self.cubicsetup = cubicsetup
        self.symmetry_type = CubicSetup.cubic_symmetry_from_pose(pose)
        self.symmetry_base = CubicSetup.get_base_from_pose(pose)
        self.map_chains()
        self.use_atoms_beyond_CB = use_atoms_beyond_CB
        self.chain_names_in_use = [pose.pdb_info().chain(pose.chain_end(i)) for i in self.chain_ids_in_use]
        self.core_atoms_str = ["CB", "N", "O", "CA", "C"]
        self.core_atoms_index = self._get_atom_indices(self.core_atoms_str)
        self.no_clash = no_clash
        self.distances = None
        self.apply_symmetry_to_score = apply_symmetry_to_score
        self.neighbour_dist = neighbour_dist
        self.neighbour_anchorage = use_neighbour_anchorage
        self.neighbour_ss = use_neighbour_ss
        self.verbose = verbose
        if self.neighbour_ss:
            self.dssp = self._create_dssp(pose)
        else:
            self.dssp = None
        self.clash_penalty = clash_penalty
        self.use_hbonds = use_hbonds
        if self.use_hbonds:
            self.hbond_score = self._create_hbonds_scores()
        else:
            self.hbond_score = None
        assert atom_selection in ("surface", "all", "core")
        # creating point cloud
        self.atom_selection = atom_selection
        self.clash_dist_str, self.clash_dist_int = self._create_default_clash_distances(pose, clash_dist=clash_dist, lj_overlap=lj_overlap)
        self.sym_ri_ai_map = self._select_atoms(pose, atom_selection)
        self.symdef = cubicsetup
        self.connected_jumpdof_map = self._create_connectted_jumpdof_map(pose)
        self.point_clouds = self._get_point_clouds(pose)
        self.point_cloud_size = len(self.point_clouds[1])
        self.main = np.zeros(self.point_clouds[1].shape)
        self.rest = np.zeros((self.point_clouds[1].shape[0]*len(self.chain_ids_in_use[1:]), 3))
        self.matrix_shape = (self.main.shape[0], self.rest.shape[0])
        # masks and weights
        self.neighbour_mask, self.donor_acceptor_mask, self.hf_masks = self._create_masks(pose)
        self.symmetric_wt = self._create_symweigts(pose)
        self.coordination_wt = self._create_coordination_weights(pose, interaction_bonus)
        self.masked_symmetric_coordination_wt = self.symmetric_wt[self.neighbour_mask] * self.coordination_wt[self.neighbour_mask]
        self.masked_coordination_wt = self.coordination_wt[self.neighbour_mask]
        # other
        self.clash_limit_matrix = self._create_clash_limit_matrix()
        # _internal_update is not strictly nescesarry but will give meaning for some attributes such as self.main, self.rest
        # straight after initialization
        self._internal_update(pose)

    @staticmethod
    def _create_dssp(pose):
        return Vector1(Dssp(pose).get_dssp_secstruct())

    def map_chains(self):
        """"""
        # this is the general setup if the base is HF
        if self.symmetry_type in ("I", "O"):
            self.chain_ids_in_use = [1, 2, 3, 8, 7, 6]
            self.hfs = {0: (2, 3), 1: (8,), 2: (7, 6)}
        else:
            assert self.symmetry_type == "T"
            self.chain_ids_in_use = [1, 2, 4, 6, 5]
            self.hfs = {0: (2,), 1: (4,), 2: (6, 5)}
        # The following varies depending on which base we have. If the base is not HF, we have to change
        # self.chain_ids_in_use and self.hfs
        jid = self.cubicsetup.get_jumpidentifier()
        self.jump_apply_order = [f'JUMP{jid}fold1', f'JUMP{jid}fold1_z', f'JUMP{jid}fold111', f'JUMP{jid}fold111_x', f'JUMP{jid}fold111_y', f'JUMP{jid}fold111_z']
        if self.cubicsetup.get_base() != "HF":
            chain_map = get_chain_map_as_dict(symmetry_type=self.symmetry_type, is_righthanded=self.cubicsetup.righthanded)
            self.chain_ids_in_use = [chain_map[c][self.symmetry_base] for c in self.chain_ids_in_use]
            self.hfs = {k: tuple(chain_map[c][self.symmetry_base] for c in v) for k, v in self.hfs.items()}

    # --- score functions, either total or individual ones --- #

    # fixme: Applying symweights makes it about 20% slower - this can be precalculated
    def score(self, pose):
        """Computes the total score w/wo hbonds"""
        self._internal_update(pose)
        score = self.clash_score() + self.neighbour_score()
        if self.use_hbonds:
            score += self.hbond_score.score(pose)
        return score

    def breakdown_score(self, pose):
        """Breaks down the total score into its individual components (neighbours, clashes and hbonds)."""
        self._internal_update(pose)
        string = f"clash_score: {self.clash_score()}, neighbour_score: {self.neighbour_score()}"
        if self.use_hbonds:
            string += f", hbond_score: {self.hbond_score.score(pose)}"
        string += f", total: {self.score(pose)}"
        return string

    def breakdown_score_as_dict(self, pose):
        """Breaks down the total score into its individual components (neighbours, clashes and hbonds)."""
        self._internal_update(pose)
        d = {"clash_score": self.clash_score(), "neighbour_score": self.neighbour_score()}
        if self.use_hbonds:
            d["hbond_score"] = self.hbond_score.score(pose)
        d["total"] = self.score(pose)
        return d

    def clash_score(self):
        """Computes the clash score."""
        return self._compute_clashes() * self.clash_penalty - 0.001  # +0.01 because 0 can lead to zero division errors

    def neighbour_score(self):
        """Computes the neighbour score."""
        results = self._compute_neighbours()
        if self.apply_symmetry_to_score:
            return - np.sum(results * self.masked_symmetric_coordination_wt)
        else:
            return - np.sum(results * self.masked_coordination_wt)

    # --- Functions that computes things that might be relevant outside the scoring it self --- #

    def number_of_clashes(self, pose) -> int:
        """Only computes the current number clashes."""
        self._internal_update(pose)
        return self._compute_clashes()

    def hf_clashes(self, pose) -> dict:
        """returns the number of HF clashes for each HF"""
        self._internal_update(pose)
        hf_clashes = {}
        for hf, hf_mask in self.hf_masks.items():
            hf_clashes[hf] = int(np.sum((self.distances[hf_mask] < 0) * self.symmetric_wt[hf_mask]))
        return hf_clashes

    def hf_neighbours(self) -> dict:
        """returns the number of HF neighbours for each HF"""
        hf_neighbours = {}
        for hf, hf_mask in self.hf_masks.items():
            hf_neighbours[hf] = np.sum(np.where(self.distances[hf_mask] <= self.neighbour_dist, 1, 0)) #int(np.sum((self.distances[hf_mask] < 0) * self.symmetric_wt[hf_mask]))
        return hf_neighbours

    # --- Debug functions --- #
    # It starts to fail at 1e-5
    def pose_atoms_and_cloud_atoms_overlap(self, pose, atol=1e-4):
        """Checks that the atom xyzs in the cloud matches the same positions in the pose."""
        self._internal_update(pose)
        return np.all(np.isclose(np.concatenate((self.main, self.rest)), self.get_cloud_atoms_from_pose(pose), atol=atol))

    def get_cloud_atoms_from_pose(self, pose):
        """Gets the atom positions that is in the cloud extracted from the pose directly."""
        xyzs = []
        for chain, resi_ais_map in self.sym_ri_ai_map.items():
            for resi, ais in resi_ais_map.items():
                for ai in ais:
                    xyzs.append(list(pose.residue(resi).atom(ai).xyz()))
        return np.array(xyzs)

    # --- Visualization functions --- #

    # TODO: hide personal paths and potentially make modular
    def show_in_pymol(self, pose, visualizer, show_clashes=True):
        self._internal_update(pose)
        # fixme: why do we call this again? this is called in internal_update
        # self._apply_symmetry_to_point_cloud(pose)
        # self._store_distances()
        pose_repr = self._get_pose_representation(pose)
        visualizer.construct_pose_name(pose_repr)
        visualizer.pmm.apply(pose_repr)
        visualizer.show_only_chains(self.chain_names_in_use)
        cloud_name = "/home/mads/local_tmp/point_cloud.pdb"
        self.output_point_cloud_as_pdb(cloud_name)
        name = pose.pdb_info().name() + "_"
        if name == "_":
            name = ""
        point_cloud_name = f"{name}point_cloud"
        file_2_load = f"/Users/mads/mounts/mailer/{cloud_name}"
        visualizer.cmd.do(f"load {file_2_load}, {point_cloud_name}")
        visualizer.cmd.do(f"set sphere_scale, 0.2")
        visualizer.cmd.do(f"show spheres, {point_cloud_name}")
        visualizer.cmd.do(f"util.cbc {point_cloud_name}")
        visualizer.cmd.do("set retain_order, 1") # so we can select by the order of the pdb file
        if show_clashes:
            clash_points = self._get_clashing_points()
            # the point
            for main, rest in zip(clash_points[0], clash_points[1]):
                visualizer.mark_distance_by_rank(main, rest + self.point_cloud_size, f"{point_cloud_name}")

    def output_point_cloud_as_pdb(self, filename, storing_cloud=False):
        if self.use_atoms_beyond_CB == True:
            raise NotImplementedError("This representation will not look right with self.use_atoms_beyond_CB == False!")
        txt = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n"
        ATOM_NAME = ["N", "CA", "C", "O", "CB"]
        ELEMENT = ["N", "C", "C", "O", "C"]
        C = 1
        with open(filename, "w") as f:
            for n, chain in enumerate(self.chain_names_in_use ):
                resi = 0
                if storing_cloud:
                    points = self.point_clouds[chain]
                else:
                    if n == 0:
                        points = self.main
                    else:
                        points = self.rest[(n-1)*self.point_cloud_size:n*self.point_cloud_size]
                for n, i in enumerate(points, 1):
                    f.write(
                        txt.format(
                            "ATOM ",
                            n,
                            ATOM_NAME[n % len(ATOM_NAME)],
                            " ",
                            "ALA",
                            chain,
                            resi,
                            " ",
                            i[0],
                            i[1],
                            i[2],
                            1,
                            0,
                            ATOM_NAME[n % len(ATOM_NAME)],
                            " ",
                        )
                    )
                    if n % len(ATOM_NAME) == 0:
                        resi += 1
                f.write("TER\n")

# --- internal/private functions that does stuff under the hood --- #

    @staticmethod
    def _create_hbonds_scores():
        scorefxn = ScoreFunctionFactory.create_score_function("empty")
        terms, weights = ("hbond_sr_bb", "hbond_lr_bb"), (1, 1)
        for term, weight in zip(terms, weights):
            scoretype = ScoreTypeManager().score_type_from_name(term)
            scorefxn.set_weight(scoretype, weight)
        return scorefxn

    def _internal_update(self, pose):
        self._apply_symmetry_to_point_cloud(pose)
        self._store_distances()

    def _store_distances(self):
        self.distances = self._compute_distances()

    def _compute_distances(self):
        """Computes the euclidian distances between self.main (chain 1 / main chain) and self.rest (all other chains) in the point cloud."""
        return cdist(self.main, self.rest, 'euclidean') - self.clash_limit_matrix

    def _compute_clashes(self):
        """Computes number of clahses neighbours."""
        if self.apply_symmetry_to_score:
            return int(np.sum((self.distances < 0) * self.symmetric_wt))
        else:
            return int(np.sum(self.distances < 0))

    def _compute_neighbours(self):
        """Computes neighbours """
        return np.where(self.distances[self.neighbour_mask] <= self.neighbour_dist, 1, 0)

    # fixme: just go through a single chain
    def _calculate_neighbours(self, pose, resi):
        """Calculates the neighbours to a residue"""
        resi_nbrs = 0
        actual_resi = pose.residue(resi)
        for resj in range(1, pose.size() + 1):
            # dont calculate with itself or residues that are on other chains
            if resj == resi or pose.chain(resi) != pose.chain(resj):
                continue
            actual_resj = pose.residue(resj)
            distance = actual_resi.xyz( actual_resi.nbr_atom() ).distance( actual_resj.xyz( actual_resj.nbr_atom()))
            if distance <= self.neighbour_dist:
                resi_nbrs += 1
        return resi_nbrs

    def _create_coordination_weights(self, pose, interaction_bonus):
        if self.verbose:
            print("creating coordination weigths")
        coordination_wts = []
        # 1. add weights according to the anchorage of each residue on its own chain and the secondary structure type it is from
        for ri, aii in self.sym_ri_ai_map[1].items():
            wt = 1.0
            if self.neighbour_anchorage:
                wt *= min(1.0, self._calculate_neighbours(pose, ri) / 20.0)
            if self.neighbour_ss and self.dssp[ri] == "L":
                wt /= 3.0
            for _ in aii:
                coordination_wts.append(wt)
        coordination_wts = np.array(coordination_wts)
        other_chains = len(self.chain_ids_in_use[1:])
        coordination_wts = coordination_wts[:, np.newaxis] * np.repeat(coordination_wts, other_chains)
        # 2. add bonus to the weights according to the which chain interaction it is from
        if interaction_bonus:
            coordination_wts *= self._bcast_bonus(interaction_bonus)
        if self.verbose:
            print("DONE!")
        return coordination_wts

    def _bcast_bonus(self, interaction_bonus):
        """Broadcast the interaction_bonus (dict) so that it can be added to the coordination weights"""
        # chains must be specified in the same order as chain_in_use
        bonuses = []
        for chaini in self.chain_ids_in_use[1:]:
            for chainj, bonus in interaction_bonus.items():
                if chaini == chainj:
                    bonuses.append(bonus)
        return np.array([bonus for bonus in bonuses for _ in range(self.point_cloud_size)])

    def _create_symweigts(self, pose):
        if self.verbose:
            print("creating symmetry weights")
        si = pose.conformation().Symmetry_Info()
        symweights = []
        for ri, aii in self.sym_ri_ai_map[1].items():
            for ai in aii:
                symweights_row = []
                for chain in self.chain_ids_in_use[1:]:
                    for rj, ajj in self.sym_ri_ai_map[chain].items():
                        for aj in ajj:
                            symweights_row.append(si.score_multiply(ri, rj))
                symweights.append(symweights_row)
        if self.verbose:
            print("DONE!")
        return np.array(symweights)

    def _assign_to_hf(self, chain):
        for hf, chains in self.hfs.items():
            if chain in chains:
                return hf
        raise ValueError("chain not present in hfs!")

    def _finalize_hf_mask(self, hfs):
        hf_mask = {}
        for hf, chains in self.hfs.items():
            hf_mask[hf] = hfs == hf
        return hf_mask

    # TODO: perhaps init the arrays with numpy and then calculate a function over the arrays to make it faster
    def _create_masks(self, pose):
        if self.verbose:
            print("creating masks")
        neighbour, no, hf = [], [], []
        for ri, aii in self.sym_ri_ai_map[1].items():
            rir = pose.residue(ri)
            for ai in aii:
                neighbour_row, no_row, hf_row = [], [], []
                for chain in self.chain_ids_in_use[1:]:
                    for rj, ajj in self.sym_ri_ai_map[chain].items():
                        rjr = pose.residue(rj)
                        for aj in ajj:
                            neighbour_row.append(self._is_neighbour_interaction(rir, ai, rjr, aj))
                            no_row.append(self._is_donor_acceptor_pair(rir, ai, rjr, aj))
                            hf_row.append(self._assign_to_hf(chain))
                neighbour.append(neighbour_row)
                no.append(no_row)
                hf.append(hf_row)
        neighbour = np.array(neighbour)
        no = np.array(no)
        hf = np.array(hf)
        if self.verbose:
            print("DONE!")
        return neighbour, no, self._finalize_hf_mask(hf)

    def _is_neighbour_interaction(self, rir, ai, rjr, aj):
        """Checks if the atom pairs ai and aj on residue rir and rjr are considered a neighbour. They are both are either the CA atom of
        G or CB atom any other residue."""
        return ((rir.name1() == "G" and ai == 2) or ai == 5) and ((rjr.name1() == "G" and aj == 2) or aj == 5)

    def _is_donor_acceptor_pair(self, rir, ai, rjr, aj):
        ai_donor = rir.type().atom_type(ai).is_donor()
        ai_acceptor = rir.type().atom_type(ai).is_acceptor()
        aj_donor = rjr.type().atom_type(aj).is_donor()
        aj_acceptor = rjr.type().atom_type(aj).is_acceptor()
        return (ai_donor and aj_acceptor) or (ai_acceptor and aj_donor)

    def _create_clash_limit_matrix(self):
        ri_ai_map = self.sym_ri_ai_map[1]
        ri_radiis_map = {k:[self.clash_dist_int[v] if v in self.core_atoms_index else self.clash_dist_int[99] for v in vv] for k,vv in ri_ai_map.items()}
        # fixme: modify so that N O interactions are less
        radii_main = np.array([radii for ri, radiis in  ri_radiis_map.items() for radii in radiis])
        radii_rest = np.array([radii for _ in range(len(self.chain_ids_in_use) - 1) for radii in radii_main])
        radii_all = radii_main[:, np.newaxis] + radii_rest
        radii_all[self.donor_acceptor_mask] = self.no_clash
        return radii_all

    # fixme call the the residues found "subset" as that is what I call it in the docs
    def _mark_surface_atoms(self, aspose, core_sasa_limit=20.0):
        """Marks only atoms on the surface.

        Uses classical Rosetta based layer selection with a combination of SASA and CA-CB cone neighbour detection.

        The algorithm is as follows:

        1. Calculate SASA for all atoms in the pose. All residues with a total sasa (add up their atom sasas) of more than 20 are changed to
        ALA, except if they are already GLY.

        2. Calculate CA-CB cone for all residues. All residues with less than 2.0 CA atoms in that cone are also changed to ALA,
        except if they are already GLY.

        Step 1 and 2 are done to remove all side chain information of residues on the surface of the protein. Side chain atoms
        inside the core of the protein will remain and potentially some side chain atoms in boundary. The user might want to keep these
        boundary side chain atoms depending on the task at hand. If the task is to design the surface of the protein based on an existing
        crystal structure without prepacking, the boundary atoms might be useful to specify in the atom subset for 2 reasons: 1. We have
        confidence that the rotamer is likely correct (not always the case though) and, since the are on the boundary they interact with
        the monomer it self and helps in the stability and structural integrity of the monomer. In the design case it is advantages to not
        introduce to many mutations and keeping the residue and its rotamer in there will help in accomplishing that gaol.
        However, when doing structure predicition, we have less confidence in the rotamer, either because we use prepacking on a bound
        structure for benchmarking or we start with an unbound structure that has to conform to a bound state.

        If the user want to remove all sidechain information there's an option to include only BB + CB atoms in the final subset so that
        no rotamer information is saved. This is done by use_atoms_beyond_CB.

        3. This step depends on the goal of the user. If the user design/prediction

        4. Calculate SASA on all atoms on this modified pose. Pick all heavy atoms that have 0 SASA and use that subset
        """

        # 1 find surface residues based on SASA
        s = SasaCalc()
        s.calculate(aspose)
        per_residue_sasa = np.array(s.get_residue_sasa())
        surface_residues_from_sasa = set(np.where(per_residue_sasa > core_sasa_limit)[0] + 1)
        # print(f"sele sasa_surf_res, chain A and resi {'+'.join(map(str, surface_residues_from_sasa))}")
        # print(f"sele diff_surf, chain A and resi {'+'.join(map(str, surface_residues_from_cone.difference(surface_residues_from_sasa)))}")


        # 2 find surface residues based on CA->CB cone
        srbl = SelectResiduesByLayer(False, False, True)
        srbl.use_sidechain_neighbors(True)
        srbl.sasa_core(5.2)
        srbl.sasa_surface(2.0)
        try:
            surface_residues_from_cone = set(srbl.compute(aspose))
        except RuntimeError as e:
            # This produces the following error: /source/src/numeric/xyzVector.hh:654 Cannot create normalized xyzVector from vector of length() zero.
            # I believe this is when you have a very small protein that does not have any core (I get it when i run pdbid: 7JRH)
            # so in this case all atoms should actually be chosen
            surface_residues_from_cone = surface_residues_from_sasa

        # print(f"sele cone_surf_res, chain A and resi {'+'.join(map(str, surface_residues_from_cone))}")

        # combine them:
        all_surface_residues = set(surface_residues_from_sasa).union(set(surface_residues_from_cone))

        # print(f"sele core_res, chain A and resi {'+'.join(map(str, core_residues))}")
        # ModifyRepresentation(residues=all_surface_residues, unaffected="G").apply(aspose)
        build_new_aa_repr(aspose, mutant_aa="A", residues=all_surface_residues, types_not_to_mutate="G")

        # 3 Calculate Sasa again on the modified aspose and get all the atoms that have 0 sasa
        s.calculate(aspose)

        atom_sasa = s.get_atom_sasa()
        resi_atom_sasa_map = {ri:{ai: atom_sasa.get(AtomID(ai, ri)) for ai in self._get_heavy_ai(aspose, ri)} for ri in range(1, aspose.size() + 1)}
        if not self.use_atoms_beyond_CB: # then put all the sasa beyond CB (>5) onto CB and replace the atom with 0 sasa
            for resi, atom_sasa_map in resi_atom_sasa_map.items():
                for atomn, sasa in atom_sasa_map.items():
                    if atomn > 5:
                        atom_sasa_map[5] += sasa
                        atom_sasa_map[atomn] = 0
        atom_size_unclean = len([i for ii in [list(m.keys()) for resi, m in resi_atom_sasa_map.items()] for i in ii])
        core_atom_size_unclean = len([i for ii in [list(m.keys()) for resi, m in resi_atom_sasa_map.items()] for i in ii if i in self.core_atoms_index])
        # atom_sasa_arr = np.array(list(atom_sasa_map.values()))
        # remove all atoms that have 0
        resi_atom_sasa_map_clean = {}
        for resi, atom_sasa_map in resi_atom_sasa_map.items():
            resi_atom_sasa_map_clean[resi] = [atom for atom, sasa in atom_sasa_map.items() if sasa != 0]
        # atom_sas_map
        atom_size_clean = len([i for ii in [m for resi, m in resi_atom_sasa_map_clean.items()] for i in ii])
        if self.verbose:
            print(f"All atoms: {atom_size_unclean}")
            print(f"{','.join(self.core_atoms_str)}: {core_atom_size_unclean}")
            print(f"kept atoms: {atom_size_clean}")
        return resi_atom_sasa_map_clean

    # def _mark_surface_cone_atoms(self, aspose):
    #     """Use CA-CB cone surface selection """
    #     from shapedesign.src.visualization.visualizer import Visualizer
    #     for sel, typ in zip(([True, False, False], [False, True, False], [False, False, True]), ("core", "bound", "surf")):
    #         srbl = SelectResiduesByLayer(*sel)
    #         srbl.use_sidechain_neighbors(True)
    #         srbl.sasa_core(5.2)
    #         srbl.sasa_surface(2.0)
    #         srbl.compute(aspose)
    #         print(f"sele {typ}_res, chain A and resi {'+'.join(map(str, srbl.compute(aspose)))}")
    #     from shapedesign.src.visualization.visualizer import Visualizer
    #     Visualizer(reinitialize=False).send_pose(aspose, name="aspose")
    #     # vec = calc_sc_neighbors(aspose)
    #
    #
    #
    #     ls_core = LayerSelector()
    #     ls_core.set_layers(True, False, False)
    #     ls_core.apply(aspose)
    #     core = OperateOnResidueSubset(PreventRepackingRLT(), ls_core)
    #     task_core = TaskFactory()
    #     task_core.push_back(core)
    #
    #
    #     ls_bound = LayerSelector()
    #     ls_bound.set_layers(False, True, False) # core, boundary, surface
    #     ls_bound.apply(aspose)
    #
    #     ls_surf = LayerSelector()
    #     ls_surf.set_layers(False, False, True) # core, boundary, surface
    #     ls_surf.apply(aspose)
    #
    #     resi_atom_sasa_map = {ri:{ai: atom_sasa.get(AtomID(ai, ri)) for ai in self._get_heavy_ai(aspose, ri)} for ri in range(1, aspose.size() + 1)}
    #     atom_size_unclean = len([i for ii in [list(m.keys()) for resi, m in resi_atom_sasa_map.items()] for i in ii])
    #     core_atom_size_unclean = len([i for ii in [list(m.keys()) for resi, m in resi_atom_sasa_map.items()] for i in ii if i in self.core_atoms_index])
    #     # atom_sasa_arr = np.array(list(atom_sasa_map.values()))
    #     # remove all atoms that have 0
    #     resi_atom_sasa_map_clean = {}
    #     for resi, atom_sasa_map in resi_atom_sasa_map.items():
    #         resi_atom_sasa_map_clean[resi] = [atom for atom, sasa in atom_sasa_map.items() if sasa != 0]
    #     # atom_sas_map
    #     atom_size_clean = len([i for ii in [m for resi, m in resi_atom_sasa_map_clean.items()] for i in ii])
    #     print(f"All atoms: {atom_size_unclean}")
    #     print(f"{','.join(self.core_atoms_str)}: {core_atom_size_unclean}")
    #     print(f"kept atoms: {atom_size_clean}")
    #     return resi_atom_sasa_map_clean

    def _get_heavy_ai(self, pose, resi):
        residue = pose.residue(resi)
        natoms = residue.natoms()
        return [i for i in range(1, natoms + 1) if residue.type().atom_type(i).is_heavyatom()]

    def _create_default_clash_distances(self, pose, lj_overlap=20, clash_dist=None):
        """Create the default clash distance for each atom as 20% of the ."""
        # LJ results for 20% overlap between the radii (*0.8 = 20%)
        zip_ = zip([pose.residue(10).atom_type(x).lj_radius() for x in self.core_atoms_index], self.core_atoms_str)
        # default = {'CB': 1.6094080000000002, 'N': 1.4419616, 'O': 1.2324640000000002, 'CA': 1.6094080000000002, 'C': 1.5333288}
        default_str = {k: v * ((100 - lj_overlap)/100) for v, k in zip_}
        # reweighing of CB and SC
        default_str["CB"] = 1.5
        default_str["SC"] = 1.5
        # add custom
        if clash_dist:
            assert isinstance(clash_dist, dict), "clash_dist must be a dict"
            for k, v in clash_dist.items():
                default_str[k] = v
        default_int = {99: default_str["SC"]}
        default_int.update({self.core_atoms_index[self.core_atoms_str.index(k)]:v for k, v in default_str.items() if k != "SC"})
        return default_str, default_int

    def _get_atom_indices(self, atoms_str):
        ma = {k:v for v,k in zip([5, 1, 4, 2, 3], ["CB", "N", "O", "CA", "C"])}
        return [ma[k] for k in atoms_str]

    # def map_chain_name_to_id(self, pose):
    #     chain_name_to_chain_id = {pose.pdb_info().chain(ce): n for n, ce in enumerate(pose.conformation().chain_endings(), 1)}
    #     return [chain_name_to_chain_id[chain_name] for chain_name in self.chain_ids_in_use]

    def _get_connected_jump_from_vrt(self, vrt):
        for jump, vrts in self.symmdata.get_virtual_connects().items():
            if vrts[0] == vrt:
                return jump
        raise ValueError(f"{vrt} not connected by a JUMP")

    def _get_all_jump_ids(self, pose, jname):
        jumpids = []
        while jname != 'NOPARENT':
            jname = self.symmdata.get_parent_jump(jname)
            if jname != 'NOPARENT':
                jumpids.append(sym_dof_jump_num(pose, jname))
        return jumpids

    def _get_jumps_connected_to_chains(self, pose):
        """Get the jump ids that connects a vrt to a residue"""
        return [e.label() for e in pose.conformation().fold_tree().get_jump_edges() if e.stop() <= pose.conformation().Symmetry_Info().num_total_residues()]

    def _get_all_connected_jumps(self, pose, jc):
        jumps = [jc]
        jump = jc
        while True:
            start = pose.fold_tree().upstream_jump_residue(jump)
            jump_edges = pose.fold_tree().get_jump_edges()
            for edge in jump_edges:
                if edge.stop() == start:
                    start, jump = edge.start(), edge.label()
                    break
            # process
            if jump == jumps[-1]:
                break
            else:
                jumps.append(jump)
        return reversed(jumps) # we want the last vrt to be the one connected to the subunit

    def _create_connectted_jumpdof_map(self, pose):
        master_jumpdof_map = self._get_master_jumpdofs_map(pose)
        connected_jumpdof_map = {1: master_jumpdof_map}
        # subunit_jump = sym_dof_jump_num(pose, self.jump_connected_to_chain)
        subunit_jump_clones = self._get_jumps_connected_to_chains(pose)#pose.conformation().Symmetry_Info().jump_clones(subunit_jump)
        # all_movable_jump_clones = [i for ii in [pose.conformation().Symmetry_Info().jump_clones(j) for j in master_jumpdof_map.keys()] for i in ii]
        for chain_id in self.chain_ids_in_use:
            if chain_id == 1:
                continue
            # get the jump that is connected to the subunit (jc)
            jc = None
            for jc in subunit_jump_clones:
                chain = pose.chain(pose.fold_tree().downstream_jump_residue(jc))
                if chain == chain_id:
                    break
            jumpids = self._get_all_connected_jumps(pose, jc)
            slave_jumpdof_map = {}
            for m_jumpid, dofs in master_jumpdof_map.items():
                for s_jumpid in jumpids:
                    if s_jumpid in pose.conformation().Symmetry_Info().jump_clones(m_jumpid) or s_jumpid == m_jumpid:
                        slave_jumpdof_map[s_jumpid] = dofs
                        break
            assert len(master_jumpdof_map) == len(slave_jumpdof_map)
            connected_jumpdof_map[chain_id] = slave_jumpdof_map
        return connected_jumpdof_map

    def _get_clones_to_use(self, pose):
        si = pose.conformation().Symmetry_Info()
        jumpid_clones = list(si.jump_clones(sym_dof_jump_num(pose, self.jump_connected_to_chain)))
        return [self._get_chain_name_from_jumpid(pose, j) in self.chain_ids_in_use for j in jumpid_clones]

    def _get_chain_name_from_jumpid(self, pose, jumpid):
        return pose.chain(pose.fold_tree().downstream_jump_residue(jumpid))

    def _apply_symmetry_to_point_cloud(self, pose):
        diff = cycle(self._get_all_pertubation_difference(pose))
        for n, (chain, syminfo) in enumerate(self.connected_jumpdof_map.items()):
            cloud = np.copy(self.point_clouds[chain])
            for jumpid, dofs in syminfo.items():
                # fixme: could be faster if you check if the angles are not zero
                for dof in dofs:
                    # the trans applies internally to the clou
                    if dof == 1:
                        cloud += self._apply_x_trans(cloud, next(diff), pose, jumpid)
                    elif dof == 2:
                        cloud += self._apply_y_trans(cloud, next(diff), pose, jumpid)
                    elif dof == 3:
                        cloud += self._apply_z_trans(cloud, next(diff), pose, jumpid)
                    elif dof == 4:
                        cloud = self._apply_x_rot(cloud, next(diff), pose, jumpid)
                    elif dof == 5:
                        cloud = self._apply_y_rot(cloud, next(diff), pose, jumpid)
                    elif dof == 6:
                        cloud = self._apply_z_rot(cloud, next(diff), pose, jumpid)
            if n == 0:
                self.main = cloud
            else:
                self.rest[(n-1)*self.point_cloud_size:n*self.point_cloud_size] = cloud

    def _get_all_pertubation_difference(self, pose):
        return [get_jumpdof_int_int(pose, jumpid, dof) for jumpid, dofs in self.connected_jumpdof_map[1].items() for dof in dofs]
        #return [self._get_pertubation(pose, jumpid, dof) for jumpid, dofs in self.connected_jumpdof_map[1].items() for dof in dofs]

    def _get_master_jumpdofs_map(self, pose):
        dofs = pose.conformation().Symmetry_Info().get_dofs()
        jumpdofs = {sym_dof_jump_num(pose, k): [] for k in self.jump_apply_order}
        for jump_id, symdof in dofs.items():
            jump_name = jump_num_sym_dof(pose, jump_id)
            if jump_name in self.jump_apply_order:
                for dof in range(1, 7):
                    if symdof.allow_dof(dof):
                        jumpdofs[jump_id].append(dof)
        return jumpdofs

    def _get_all_core_atoms(self, aspose):
        return {ri: [ai for ai in self._get_heavy_ai(aspose, ri) if ai in self.core_atoms_index] for ri in range(1, aspose.size() + 1)}

    def _get_all_atoms(self, aspose):
        return {ri: [ai for ai in self._get_heavy_ai(aspose, ri)] for ri in range(1, aspose.size() + 1)}

    # FIXME: IS GOING TO BREAK THE ENTIRE CODE IF THE MAIN CHAIN IS NOT CHAIN A
    def _symmetrize_ri_ai_map(self, pose, aspose, ri_ai_map):
        sym_ri_ai_map = {chain: {} for chain in self.chain_ids_in_use}
        sym_ri_ai_map[1] = ri_ai_map
        for ri in range(aspose.size() + 1, pose.size() + 1):
            chain = pose.chain(ri)
            if chain in self.chain_ids_in_use:
                as_ri = pose.conformation().Symmetry_Info().get_asymmetric_seqpos(ri)
                ai_match = ri_ai_map.get(as_ri, None)
                if ai_match:
                    chain = pose.chain(ri)
                    sym_ri_ai_map[chain][ri] = ai_match
        return sym_ri_ai_map

    def _get_asymmetric_pose(self, pose, set_dofs_to_0=False):
        pose = pose.clone()
        if set_dofs_to_0:
           set_all_dofs_to_zero(pose)
        aspose = Pose()
        extract_asymmetric_unit(pose, aspose, False)
        return aspose

    def _select_atoms(self, pose, atom_selection):
        aspose = self._get_asymmetric_pose(pose)
        # get atoms we are interested in
        if atom_selection == "surface":
            ri_ai_map = self._mark_surface_atoms(aspose)
        # elif atom_selection == "surface_cone":
        #     ri_ai_map = self._mark_surface_cone_atoms(aspose)
        elif atom_selection == "core":
            ri_ai_map = self._get_all_core_atoms(aspose)
        elif atom_selection == "all":
            ri_ai_map = self._get_all_atoms(aspose)
        # map onto all chains!
        sym_ri_ai_map = self._symmetrize_ri_ai_map(pose, aspose, ri_ai_map)
        return sym_ri_ai_map

    def _get_point_clouds(self, pose):
        # TODO: coarse grain atoms
        pose_dofs_are_0 = pose.clone()
        # set all degrees of freedom to 0
        for jump, dofs in self.connected_jumpdof_map[1].items():
            for dof in dofs:
                set_jumpdof_int_int(pose_dofs_are_0, jump, dof, 0)
        # create a new pose object and fill it with the asymmetric pose
        clouds = {}
        for chain in self.chain_ids_in_use:
            clouds[chain] = self._get_point_cloud(pose_dofs_are_0, self.sym_ri_ai_map[chain])
        return clouds

    def _get_point_cloud(self, pose, ri_ai_map):
        points = []
        for ri, aii in ri_ai_map.items():
            for ai in aii:
                points.append(np.array(pose.residue(ri).atom(ai).xyz()))
        return np.array(points)

    def _get_atom_name(self, pose, resi, atom_index):
        return pose.residue(resi).atom_name(atom_index).replace(" ", "")

    # def _set_start_dof(self, pose):
    #     dofs = {27: {3: None, 6: None}, 29: {1: None}, 30: {4: None, 5: None, 6: None}}
    #     for jumpid, dofinfo in dofs.items():
    #         for dof, value in dofinfo.items():
    #             if dof <= 3:
    #                 trans = pose.jump(jumpid).get_translation()
    #                 dofs[jumpid][dof] = trans[dof - 1]
    #             elif dof == 4:
    #                 rot = pose.jump(jumpid).get_rotation()
    #                 dofs[jumpid][dof] = self.get_euler_angle_x(rot)
    #             elif dof == 5:
    #                 rot = pose.jump(jumpid).get_rotation()
    #                 dofs[jumpid][dof] = self.get_euler_angle_y(rot)
    #             elif dof == 6:
    #                 rot = pose.jump(jumpid).get_rotation()
    #                 dofs[jumpid][dof] = self.get_euler_angle_z(rot)
    #     return dofs

    # timeit.timeit(lambda :self.get_all_pertubation_difference(pose), number=100)
    # 0.007277664728462696
    # NOW SUPERSEDED BY get_jumpdof_int_int
    # def _get_pertubation(self, pose, jumpid, dof):
    #
    #     if dof <= 3:
    #         trans = pose.jump(jumpid).get_translation()
    #         return trans[dof - 1]
    #     elif dof == 4:
    #         rot = pose.jump(jumpid).get_rotation()
    #         return self._get_euler_angle_x(rot)
    #     elif dof == 5:
    #         rot = pose.jump(jumpid).get_rotation()
    #         return self._get_euler_angle_y(rot)
    #     elif dof == 6:
    #         rot = pose.jump(jumpid).get_rotation()
    #         return self._get_euler_angle_z(rot)

    def _translate_to_center(self, cloud, origo):
        return cloud - origo

    def _translate_to_origo(self, cloud, origo):
        return cloud + origo

    def _apply_rotation_at_origo(self, f_rot, cloud, angle, pose, jumpid):
        origo = self._get_vrt_origo(pose, jumpid)
        cloud = self._translate_to_center(cloud, origo)
        cloud = np.dot(cloud, f_rot(pose, jumpid, angle))
        cloud = self._translate_to_origo(cloud, origo)
        return cloud

    def _apply_x_rot(self, cloud, angle, pose, jumpid):
        return self._apply_rotation_at_origo(self._get_x_rot_matrix, cloud, angle, pose, jumpid)

    def _apply_y_rot(self, cloud, angle, pose, jumpid):
        return self._apply_rotation_at_origo(self._get_y_rot_matrix, cloud, angle, pose, jumpid)

    def _apply_z_rot(self, cloud, angle, pose, jumpid):
        return self._apply_rotation_at_origo(self._get_z_rot_matrix, cloud, angle, pose, jumpid)

    def _apply_x_trans(self, cloud, trans, pose, jumpid):
        # THIS IS MINUS BECAUSE OF A BUG/FEATURE IN ROSETTA
        return - self._get_x_vector(pose, jumpid) * trans

    def _apply_y_trans(self, cloud, trans, pose, jumpid):
        return self._get_y_vector(pose, jumpid) * trans

    def _apply_z_trans(self, cloud, trans, pose, jumpid):
        # THIS IS MINUS BECAUSE OF A BUG/FEATURE IN ROSETTA
        return - self._get_z_vector(pose, jumpid) * trans

    def _get_vrt_origo(self, pose, jumpid):
        return np.array(self._get_vector(pose, 1, jumpid))

    def _get_x_vector(self, pose, jumpid):
        x = np.array(self._get_vector(pose, 2, jumpid) - self._get_vector(pose, 1, jumpid))
        return x / np.linalg.norm(x)

    def _get_y_vector(self, pose, jumpid):
        y = np.array(self._get_vector(pose, 3, jumpid) - self._get_vector(pose, 1, jumpid))
        return y / np.linalg.norm(y)

    def _get_z_vector(self, pose, jump_id):
        # THIS IS MINUS BECAUSE OF A BUG/FEATURE IN ROSETTA
        z = np.cross(self._get_x_vector(pose, jump_id), self._get_y_vector(pose, jump_id))
        return z / np.linalg.norm(z)

    def _get_vector(self, pose, atomid, jumpid):
        return pose.residue(pose.fold_tree().upstream_jump_residue(jumpid)).atom(atomid).xyz()

    # timeit.timeit(lambda: r.from_rotvec([0, 0, 45], True).as_matrix(), number=100)
    # 0.0018161246553063393
    def _get_x_rot_matrix(self, pose, jumpid, angle):
        return R.from_rotvec(self._get_x_vector(pose, jumpid) * angle, True).as_matrix()

    def _get_y_rot_matrix(self, pose, jumpid, angle):
        # THIS IS MINUS BECAUSE OF A BUG/FEATURE IN ROSETTA
        return R.from_rotvec(- self._get_y_vector(pose, jumpid) * angle, True).as_matrix()

    def _get_z_rot_matrix(self, pose, jumpid, angle):
        return R.from_rotvec(self._get_z_vector(pose, jumpid) * angle, True).as_matrix()

    def _get_euler_angle_x(self, rot):
        """Variables as assigned in Rosetta:
        	xx_ = xy_ = xz_
    	    yx_ = yy_ = yz_
        	zx_ = zy_ = zz_

        :param rot:
        :return:
        """
        return math.degrees(math.atan2(rot.zy, rot.zz))

    def _get_euler_angle_y(self, rot):
        return math.degrees(math.atan2(-rot.zx, math.sqrt(rot.zy**2 + rot.zz**2)))

    def _get_euler_angle_z(self, rot):
        return math.degrees(math.atan2(rot.yx, rot.xx))

    def _get_pose_representation(self, pose):
        from shapedesign.src.utilities.pose import add_id_to_pose, get_pose_id
        if self.atom_selection == "all":
            raise NotImplementedError  # should be easy to do though
        elif self.atom_selection == "core":
            raise NotImplementedError  # should be easy to do though
        else:
            # NOTE: FOR SOME REASON set_dofs_to_0 CHANGES THE SASA???? so it is a bit different from the original one?????
            aspose = self._get_asymmetric_pose(pose, set_dofs_to_0=True)
            self._mark_surface_atoms(aspose)
        #
        # now symmetrize it
        tmp_symm = "/tmp/symdef.symm"
        if pose.data().has(CacheableDataType.STRING_MAP):
            self.symdef = StringIO(get_symmetry_file_from_pose(pose))
        else:
            assert self.symdef, "if the pose does not contain a symmetry comment then the symmetry file must be supplied in self.symdef"
        symdef = get_updated_symmetry_file_from_pose(pose, self.symdef.symmetry_name)
        with open(tmp_symm, "w") as f:
            f.write(symdef)
        SetupForSymmetryMover(tmp_symm).apply(aspose)
        try:
            add_id_to_pose(aspose, get_pose_id(pose))
        except ValueError:
            pass
        return aspose



    def _get_clashing_points(self):
        return np.where(self.distances < 0)


    # def get_affected_atoms(self, pose, cb_only=False, lj=False):
    #     if cb_only:
    #         if lj:
    #             self.compute_cb_with_lj(pose)
    #             results = self.get_all_clash_locations_with_lj()
    #         else:
    #             self.compute_cb(pose)
    #             results = self.get_all_clash_locations()
    #     else:
    #         if lj:
    #             self.compute_all_with_lj(pose)
    #             results = self.get_all_clash_locations_with_lj()
    #         else:
    #             self.compute_all(pose)
    #             results = self.get_all_clash_locations()
    #     tmp_main_clash, tmp_rest_clash = results
    #     # remove non clashses as one does in self.filter (TODO: make it nicer)
    #     main_clash, rest_clash = [],[]
    #     for mc, rc in zip(tmp_main_clash, tmp_rest_clash):
    #         if self.hbond_simple:
    #             if not self.clashes_allowed(pose, [mc], [rc]):
    #                 main_clash.append(mc)
    #                 rest_clash.append(rc)
    #         else:
    #             if self.some_clashes_are_not_involving_N_or_O([mc], [rc]) or\
    #                     self.some_clashes_are_not_a_results_of_hbonds(pose, [mc], [rc]):
    #                 main_clash.append(mc)
    #                 rest_clash.append(rc)
    #     main_resi = [ self.main_resi_iter[i] for i in main_clash]
    #     main_atoms = [ self.main_atom_str_iter[i] for i in main_clash]
    #     rest_resi = [ self.rest_resi_iter[i] for i in rest_clash]
    #     rest_atoms = [ self.rest_atom_str_iter[i] for i in rest_clash]
    #     return main_resi, main_atoms, rest_resi, rest_atoms




    #     mark_distance(pose, *self.get_affected_atoms(pose, cb_only=cb_only, lj=lj), cmd, name, color)
    #
    #     k
    #     if show_hbonds:
    #         self.show_hbonds(pose, cmd)
    #
    # def show_hbonds(self, pose, cmd):
    #     show_hbonds(pose, cmd, printout=True, only_BB=True, only_inter_chain=True)
