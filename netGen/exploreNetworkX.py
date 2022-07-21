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
    
   
    # n = 25
    #alpha = 0.3
    #beta =0.3
    #gamma = 0.3

    # make a scale-free directed graph with parameters
    G = nx.scale_free_graph(20,alpha = 0.1, gamma=0.5, beta=0.4 ) 
    
    
    
    #G= nx.barabasi_albert_graph(15,10) #undirected
    #nx.draw(G, with_labels=True)
    #plt.show()
    
    
    #powerlaw_cluster_graph(n, m, p, seed=None)
    #G = nx.navigable_small_world_graph(4)
    
    
    ###########################################################################
    ######################## Vizailize Graphs #################################
    # plot graphs
    
    ## plotting options
    options = {
    'node_color': 'lightsteelblue',
    'node_size': 300,
    'width': 2,
    'arrowstyle': '-|>',
    'arrowsize': 12,
    }
    
    ## figure setup 
    fig, axes = plt.subplots(1,1, figsize = (7,7))
        # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    
    ## differnt plotting commands 
    #nx.draw(G)
    #nx.draw_networkx(G)
    nx.draw_random(G, arrows=True, **options, with_labels=True, font_weight='regular')
    #nx.draw_circular(G)
    #nx.draw_kamada_kawai(G)
    #nx.draw_random(G)
 
    
    ## show interactive plot
    #plt.show()
    
    ###########################################################################
    ######################### Analyze Graphs ##################################
    # calculate and visualize graph characteristics
    
    ## print out graph characteristics
    
    #degree_dist_print(G)
    
    
    
    G.graph['networkType'] = "random"
    
    
    GInspector = netInspector(G)
    
    GInspector.eval_all()
    
    print(GInspector)
    
    print(G.graph['networkType'])
    
    
    
    
        
    
    
    
    