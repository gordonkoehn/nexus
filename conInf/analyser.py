#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements some methods to post process the spycon_result and test_metrics.

Created on Tue Aug  2 10:31:00 2022

@author: gordonkoehn
"""

# import common stuff
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx

# import inference method
sys.path.append('../tools/spycon/src')
from sci_sccg import Smoothed_CCG
### get conncectivity test 
from spycon_tests import load_test, ConnectivityTest
from spycon_result import SpikeConnectivityResult

sys.path.append('../tools')
import adEx_util as  adEx_util




def getInferedNxGraph(spycon_result : SpikeConnectivityResult):
    """Get infered networkX Graph from spycon_result with set threshold.
    
    Parameters
    ----------
    spycon_result : SpikeConnectivityResult
    
        
    Returns
    -------
    G_infered_nx : networkx.classes.graph.Graph
    
    """
    G_infered = pd.DataFrame(spycon_result.stats, columns=['id_from', 'id_to', 'weight'])
    G_infered = G_infered.assign(
        isEdge=np.where(
            G_infered['weight'] > spycon_result.threshold,
            True,False))
    #get only significant edges
    G_infered_sig = G_infered[G_infered['isEdge'] == True]
    #fix '/ make certain that the labels are still integers
    G_infered_sig.id_from =  G_infered_sig.id_from.astype(int)
    G_infered_sig.id_to = G_infered_sig.id_to.astype(int)
    
    G_infered_nx = nx.from_pandas_edgelist(G_infered_sig, source='id_from', target='id_to',create_using=nx.DiGraph())
    
    return G_infered_nx