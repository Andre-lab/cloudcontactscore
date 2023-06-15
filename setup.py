#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen 
@Date: 2/14/23 
"""
from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='cloudcontactscore',
    version='0.1',
    packages=find_packages(where=".", exclude=("test",)),
    #packages=['cubicsym'],
    url='https://github.com/Andre-lab/cloudcontactscore',
    license='MIT',
    author='mads',
    author_email='mads.jeppesen@biochemistry.lu.se',
    description='Fast score function for coarse grained protein docking'
    # FIXME: mpi could be an optional in the the future
    #	install_requires = []
)

