#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen 
@Date: 2/15/23 
"""
import pickle
from cloudcontactscore.cloudcontactscore import CloudContactScore

def pickle_dump_cloudcontactscore(css: CloudContactScore, file: str):
    """Pickles a CloudContactScore object to a file that can be reloaded. CloudContactScore cannot be
    pickled directly as certain Rosetta objects have to be removed before doing so."""
    # delete all non picklable objects (These are Rosetta objects)
    hbond_score = None
    if css.hbond_score is not None:
        hbond_score = css.hbond_score
        del css.hbond_score
    # save the cloudcontactscore object to file
    with open(file, "wb") as f:
        pickle.dump(css, f)
    # reattach non-picklable objects onto the cloudcontactscore object again
    if hbond_score is not None:
        css.hbond_score = hbond_score

def pickle_load_cloudcontactscore(file: str) -> CloudContactScore:
    """Unpickles a CloudContactScore object from a file. CloudContactScore cannot be
    pickled directly as certain Rosetta objects have to be removed before doing so."""
    # load the cloudcontactscore object to file
    with open(file, "rb") as f:
        css = pickle.load(f)
    # reconstruct all non picklable objects (These are Rosetta objects)
    #css.hbond_score = CloudContactScore._create_hbonds_scores()
    css.hbond_score = None
    return css
