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

###############################################################################
###############################################################################
if __name__ == '__main__':

    ###########################################################################
    ####################### Graph Generation ##################################
    # make a scale free graph of defined parameters
   
    ## define output lists
    
    # a_min = 0.15
    # a_max = 0.85
    # b_min = 0.15
    # b_max = 0.85
    # g_min = 0.05
    # g_max = 0.85
   
    # step = 0.05
   
    # alphas = np.arange(a_min, a_max+step, step)
    # betas = np.arange(b_min, b_max+step, step)
    # gammas = np.arange(g_min, g_max+step, step)
    
    # abg_space = []
    
    # # loop through all conditions and analyze files
    # for a in alphas:
    #     for b in betas:
    #         for g in gammas:
    #             if (a+b+g == 1): 
    #                 abg_point = [a,b,g]
    #                 abg_space.append(abg_point)
                    
    # print(f"A total of {len(abg_space) : 3.0f} combindation of alpha, beat and gamma are tested.")
    
    a=np.random.uniform(0.05,0.95)
    b=np.random.uniform(0.05,0.95)
    g=np.random.uniform(0.05,0.95)
    
    abg_sum =a+b+g
    a= a/abg_sum
    b= b/abg_sum
    g= g/abg_sum
    
    
   # a=0.15 # Prob of adding new node with connection --> existing node
   # b=0.8 # Prob of adding edge from existing node to existing node
   # g=0.05 # Prob of adding new node with connection <-- existing node
    
    n =100
    d_in = 0.2
    d_out = 0

    # for a,b,g in abg_space:
    
        # make a scale-free directed graph with parameters
     
        #G = nx.erdos_renyi_graph(100,0.2)
        #G =G.to_directed()
        
        #G = nx.gn_graph(25, kernel=lambda x: x ** 1.5)  # A_k = k^1.5
        # print("hello")
    G = nx.scale_free_graph(n,alpha = a, beta=b, gamma=g, delta_in=d_in, delta_out=d_out)
   # G = nx.gnr_graph(25, 0.5)  # the GNR graph

    ######################## Vizailize Graphs #################################
    # plot graphs
    ######################### Analyze Graphs ##################################
    # calculate and visualize graph characteristics
    
    genParams = {'alpha':a, 'beta':b, 'gamma':g, 'delta_in':d_in, 'delta_out':d_out}

    GInspector = netInspector(G, "scale-free", genParams)
    GInspector.eval_all()
    GInspector.analyticsPanel()
        
        
    
    
    
    
        
    
    
    
    