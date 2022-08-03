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
import tools.adEx_util




def getInferedNxGraph(spycon_result : SpikeConnectivityResult, test_metrics):
    """Get infered networkX Graph from closest to PC threshold.
    
    Formerly it used spycon_result.threshold to set a threshold.
    
    The function getBestThreshold is used to get the threshold.
    
    Parameters
    ----------
    spycon_result : SpikeConnectivityResult
    
    test_metrics
    
        
    Returns
    -------
    G_infered_nx : networkx.classes.graph.Graph
    
    """
    G_infered = pd.DataFrame(spycon_result.stats, columns=['id_from', 'id_to', 'weight'])
    G_infered = G_infered.assign(
        isEdge=np.where(
            G_infered['weight'] > getBestThreshold(test_metrics)['threshold'], #spycon_result.threshold,
            True,False))
    #get only significant edges
    G_infered_sig = G_infered[G_infered['isEdge'] == True]
    #fix '/ make certain that the labels are still integers
    G_infered_sig.id_from =  G_infered_sig.id_from.astype(int)
    G_infered_sig.id_to = G_infered_sig.id_to.astype(int)
    
    G_infered_nx = nx.from_pandas_edgelist(G_infered_sig, source='id_from', target='id_to',create_using=nx.DiGraph())
    
    return G_infered_nx


def getBestThreshold(test_metrics):
    """
    Calculate best threshold.
    
    Point on ROC curve closest to the perfect classifier  (FP, TP)=  (0,1).
    
    Parameters
    ----------
    spycon_result : SpikeConnectivityResult
    
        
    Returns
    -------
    bestPoint :  dict
            bestPoint['threshold'] = threshold id in test_metrics
            bestPoint['fpr'] = list of flase positive rates
            bestPoint['tpr'] = list of true positive rates
    
    """
    # unpack testmetrics
    fpr, tpr, auc, thresholds = tuple(test_metrics[['fpr', 'tpr', 'auc', 'thresholds']].to_numpy()[0])
    
    idx_bestThreshold = np.argmin(np.sqrt((fpr**2) +((1-tpr)**2)))
    
    bestPoint = dict()
    bestPoint['threshold'] = thresholds[idx_bestThreshold]
    bestPoint['fpr'] = fpr[idx_bestThreshold]
    bestPoint['tpr'] = tpr[idx_bestThreshold]
    
    return bestPoint
    