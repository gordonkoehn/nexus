#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:43:19 2022.

@author: gordonkoehn


Utility functions to run simple AdEx models. 
"""
##### GENERAL IMPORTS ######
import numpy as np

############################




def make_marked_edges(ids, S):
    """
    Create marked egdes graph (true connections directed graph) as required by spycon_tests.ConectivityTest().
    
    Note self conncetions are allowed.
    
    Parameters
    ----------
    ids : array of (unique) ids of neurons
        list of ids of neurons in network
    S : Synapses
        holds true network definition     
    
    Returns
    -------
    marked_edges : np array
        contains array of all possible conncetions and wheater is exists in the true ground truth graph [[id_out, id_in, 1 or 0 for true or not],...]
    nodes : np array of ints
        list of all unique ids for neurons, i.e. nodes of the network
    """
    ##### FIND ALL POSSIBLE CONNCETIONS ####
    # get conncection array of all possible conncections
    nodes = np.unique(ids) # make sure that ids are unique
    
    #make mesh of all possible conncetions also self connections
    mesh = np.meshgrid(nodes, nodes)
    num_nodes = len(nodes)
    
    #bulit up array of all possible conncetions
    con_mat = np.zeros((num_nodes, num_nodes))
    marked_edges = np.vstack([mesh[1].flatten(), mesh[0].flatten(), con_mat.flatten()]).T
    marked_edges = marked_edges[np.logical_not(np.isnan(marked_edges[:,2]))]

    ###### ADD GROUND TRUTH TO POSSIBLE CONNCECTIONS ####    
    # mark true ground through connections in the connection array
    for true_conn in list(zip(S.i, S.j)): # for each Brian2 inizialied connection
        marked_edges[(marked_edges[:, 0] == true_conn[0]) & (marked_edges[:, 1] == true_conn[1]), 2] = 1   #add a one in thr thrird column
        
    return marked_edges, nodes