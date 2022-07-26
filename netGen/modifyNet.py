#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modifies network.

This module shall contain various functions to modify networks:
    - delete self-connections


Created on Tue Jul 26 12:33:43 2022

@author: gordonkoehn
"""
import networkx as nx
import matplotlib.pyplot as plt


def removeSelfConns(G : nx.classes.graph.Graph):
    """Delete all self connections in a directed networkX graph.
    
    Parameters
    ----------
    G : nx.graph.Graph
       graph to analyze
        
    Returns
    -------
    G : nx.graph.Graph
       graph without self-connections
    """
    
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G




###############################################################################
###############################################################################
if __name__ == '__main__':

    ###########################################################################
    ####################### Graph Generation ##################################
    
    G = nx.DiGraph()
    G.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75), (1,1,1),(2,2,1)])
    
    #G = nx.erdos_renyi_graph(10, 0.5, seed=45, directed=True)
    
    #G = nx.scale_free_graph(10)
    
    
    
  
    ################ PLOT ######
    fig, ax = plt.subplots()

    ## plotting options
    options = {
    'node_color': 'lightsteelblue',
    'node_size': 250,
    'width': 1,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    }
    nx.draw_networkx(G,ax=ax, pos=nx.kamada_kawai_layout(G), arrows=True, **options, with_labels=True,font_size =10, font_weight='regular')
    #####################
    
            
  
    G= removeSelfConns(G)
    
    
    ################## PLOT #####
    fig, ax = plt.subplots()

    ## plotting options
    options = {
    'node_color': 'lightsteelblue',
    'node_size': 250,
    'width': 1,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    }
    nx.draw_networkx(G,ax=ax, pos=nx.kamada_kawai_layout(G), arrows=True, **options, with_labels=True,font_size =10, font_weight='regular')
    ################### 
    
        