#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 12:16:30 2022

@author: gordonkoehn

Loads specified data.
"""

import pandas as pd

###############################################################################
###############################################################################
if __name__ == '__main__':
    
    replicaSpace = pd.read_pickle("classifyData/avg_N_100_t_10.0_probs_0.02_0.02_0.02_0.02_a_0_85_b_0_85_gi_85.0_85.0_ge_20.0_20.0_rep_3.pkl")
    
    classifyResults = pd.read_pickle("classifyData/N_100_t_10.0_probs_0.02_0.02_0.02_0.02_a_0_85_b_0_85_gi_85.0_85.0_ge_20.0_20.0_rep_3.pkl")