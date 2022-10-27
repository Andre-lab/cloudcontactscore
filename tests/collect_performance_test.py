#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[Documentation]
@Author: Mads Jeppesen
@Date: 10/11/22
"""
from pathlib import Path
import pandas as pd

dfs = []
for path in Path("output/performance_data").glob("*"):
    dfs.append(pd.read_csv(path))