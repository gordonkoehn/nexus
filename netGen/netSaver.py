#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Saves Networks from networkX with needed info to a format readble by brian2.


Created on Fri Jul 22 14:15:13 2022

@author: gordonkoehn

"""


import numpy as np
import networkx as nx
import pandas as pd
import os

#custom modules
from netGen.genNet import *

 

def save_graph(G: nx.classes.graph.Graph, graph_type : str, genParams : dict):
    """Save graph with generation parameters."""
    
    G.graph['graph_type'] = graph_type
    G.graph['genParams'] = genParams
    
    

    root_dir = 'netData'

    curr_dir = root_dir + '/' 
    curr_dir += graph_type + '/'  

                                          
    if(not(os.path.exists(curr_dir))):
        os.makedirs(curr_dir)

    save_fol = curr_dir

    if (graph_type == "random"):
        save_name = '_'.join(["G", "p-" + str(genParams['p']) ]) + '.gpickle'
    elif (graph_type == "scale-free"):
        save_name = '_'.join(["G", "a-" + str(genParams['alpha']), "b-" + str(genParams['beta']), "g-" + str(genParams['gamma']), "d_in-" + str(genParams['delta_in']),"d_out-" + str(genParams['delta_out'])]) + '.gpickle'
    else:
        raise Exception("Unknown graph type - cannot save graph.")
    
    nx.write_gpickle(G, save_fol + '/' + save_name)

    output = "Graph succesfully saved under:\n "
    output += save_fol + '/' + save_name

    print(output)

###############################################################################
###############################################################################
if __name__ == '__main__':

  
    
    ###### generate graph
        
    seed1 = 4712
    
    ### Scale-Free
    # a=0.26 #0.41    # Prob of adding new node with connection --> existing node
    # b=0.54          # Prob of adding edge from existing node to existing node
    # g=0.20 #0.05     # Prob of adding new node with connection <-- existing node
    
    # d_in = 0.2
    # d_out = 0

    # genParams = {'alpha':a, 'beta':b, 'gamma':g, 'delta_in':d_in, 'delta_out':d_out}
    # graphType = "scale-free"

    # G = scale_free_net(a=0.16, b=0.54, g=0.30,seed =    seed1 = 4712)
    
    ### Random
    p = 0.02 # Probability for edge creation
    
    graphType = "random"
    
    genParams = {'p':p}

    G = random_net(p=p,seed1=seed1)
    
    
    save_graph(G, graphType, genParams)
    

    #### read graph with

     
    G2 = nx.read_gpickle("netData/random//G_p-0.02.gpickle")
    
    GInspector = netInspector(G2, "random", None)
    
    GInspector.analyticsPanel()