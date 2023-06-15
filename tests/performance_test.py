#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance tests for ccs
@Author: Mads Jeppesen
@Date: 10/11/22
"""

from cloudcontactscore.cloudcontactscore import CloudContactScore
from simpletestlib.test import setup_test
import pandas as pd
from pathlib import Path
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

current_struct = True

pdbs = None
if rank == 0:
    df = pd.read_csv("input/files/rcsb_pdb_custom_report_20221011033220.csv", header=1)
    del df["Unnamed: 7"]
    df = df[df["Symbol"].isin(("I", "O", "T"))]
    df = df[df["Total Number of Polymer Instances (Chains) per Assembly"].isin((60, 24, 12))]
    df = df.dropna()
    # make resolution float and take the maximum resolution
    df["Resolution (Å)"] = [float(i) if i.isdigit() else max([float(j) for j in i.split(", ")]) for i in df["Resolution (Å)"]]
    df = df[(df["Symbol"] == "I") & (df["Resolution (Å)"] < 3.5)]
    avail_ids = [path.stem for path in Path("/home/shared/databases/SYMMETRICAL/I/idealized/input/native").glob("*")]
    if current_struct:
        avail_ids = [i for i in avail_ids if Path(i).stem in ("1STM", "6RPO", "6S44", "6ZLO", "7NO0", "6JJA")]
    # hack for just looking at the current structs
    # print("Available ids:", ", ".join(avail_ids))
    df = df[df["Entry ID"].isin(avail_ids)]
    pdbs = df["Entry ID"].values
    print("Used ids:", ", ".join(pdbs))
    pdbs = np.array_split(pdbs, size)

pdbs = comm.scatter(pdbs, root=0)
for pdb in pdbs: #  pd.concat([df_old, df])
    pose, symdef = setup_test(name="I", file=pdb, return_symmetry_file=True, mute=True, pymol=False)
    print(f"reading in {pdb}")
    css = CloudContactScore(pose, None, use_hbonds=True)
    score = css.breakdown_score_as_dict(pose)
    score["pdb"] = pdb
    df = pd.DataFrame({k: [v] for k, v in score.items()})
    name = f"output/performance_data/{rank}_data{'_current_struct' if current_struct else ''}.csv"
    if Path(name).exists():
        df_old = pd.read_csv(name)
        df = pd.concat([df_old, df])
    df.to_csv(name, index=False)