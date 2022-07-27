#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:06:25 2022

@author: gordonkoehn
"""


import networkx as nx
import matplotlib.pyplot as plt

# custom modules
from analyzeNet import *
from modifyNet import *

###############################################################################
###############################################################################
if __name__ == '__main__':

    ###########################################################################
    ####################### Graph Generation ##################################

    # no of nodes
    n =100

    scaleFreeGraph = True
    randomGraph = True
    
    graphType = ""
    
    seed1 =4712
    

    if scaleFreeGraph: 
        # make a scale free graph of defined parameters
    
        a=0.41 # Prob of adding new node with connection --> existing node
        b=0.54 # Prob of adding edge from existing node to existing node
        g=0.05 # Prob of adding new node with connection <-- existing node
        
        d_in = 0.2
        d_out = 0
    
        genParams = {'alpha':a, 'beta':b, 'gamma':g, 'delta_in':d_in, 'delta_out':d_out}
        graphType = "scale-free"
    
        G = nx.scale_free_graph(n,alpha = a, beta=b, gamma=g, delta_in=d_in, delta_out=d_out, seed=seed1)
        
        G= removeSelfConns(G)

        ######################## Vizailize Graphs #################################
        # plot graphs
        ######################### Analyze Graphs ##################################
        # calculate and visualize graph characteristics
        
        GInspector = netInspector(G, graphType, genParams)
        GInspector.eval_all()
        GInspector.analyticsPanel()
            
        
        
    if randomGraph:
        
        p = 0.03 # Probability for edge creation
        
        graphType = "random"
        
        genParams = {'p':p}
    
        G = nx.erdos_renyi_graph(n, p, seed=seed1, directed=True)
        
        G= removeSelfConns(G)
        

        ######################## Vizailize Graphs #################################
        # plot graphs
        ######################### Analyze Graphs ##################################
        # calculate and visualize graph characteristics
        
        GInspector = netInspector(G, graphType, genParams)
        GInspector.eval_all()
        GInspector.analyticsPanel()
            
    

    
    
    
    
        
    
    
    
    