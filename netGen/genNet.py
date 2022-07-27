#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:06:25 2022

@author: gordonkoehn
"""


import networkx as nx
import matplotlib.pyplot as plt

# custom modules
from netGen.analyzeNet import *

from netGen.modifyNet import *


def scale_free_net(n=100,a=0.41,b=0.54,g=0.05,d_in = 0.2,d_out = 0,seed1 = 4812):
    """Make scale free graph.
    
    Parameters
    ----------
    parameters as nx.scale_free_graph()
        
    Returns
    -------
    G : nx.graph.Graph
       graph without self-connections
    """
    G = nx.scale_free_graph(n,alpha = a, beta=b, gamma=g, delta_in=d_in, delta_out=d_out, seed=seed1)

    G= removeSelfConns(G)
    
    return G


def random_net(n=100,p=0.02,seed1=4812):
    """Make random graph.
    
    Parameters
    ----------
    parameters as nx.erdos_renyi_graph()
        
    Returns
    -------
    G : nx.graph.Graph
       graph without self-connections
    """
    
    G = nx.erdos_renyi_graph(n, p, seed=seed1, directed=True)
    
    G= removeSelfConns(G)
    
    return G


def classifySynapses(G, NI=20, NE=80, inhibitoryHubs=True):
    """Generate classified (inhibitory / excitatory) graph/synapses.
    
    Can be set to prioretize hub neurons to be inhibitory neurons.
    
    Parameters
    ----------
    G : nx.graph.Graph
       graph to convert
       
    NI : int
        No of inhibitory neurons
        
    NE : int 
        No of excitatory neurons
        
    inhibitoryHubs : bool
        if true the NI neurons of highes degree will be inhibitory neurons
        
    Returns
    -------
    synapses : pd.dataframe
       synapses
    """
    ##### classify neuron as inhibitory / excitatory
    
    neurons = pd.DataFrame(G.degree, columns=['node', 'degree'])
    
    
    if inhibitoryHubs:
        # get inhibitory as largest
        id_inhibitory_neurons = neurons['degree'].nlargest(NI).index
    else:
        # get random indixes for inhibitory
        id_inhibitory_neurons = random.sample(range(0, NI+NE), 20)
    
    #set all to excitatory by default
    neurons['type'] = "excitatory"
    # overwrite with inhibitory
    neurons.loc[id_inhibitory_neurons,'type'] = "inhibitory"
    
    #ids of excitatory and inhibitory neurons
    ex_neurons = neurons[neurons['type'] == "excitatory"]["node"]
    in_neurons = neurons[neurons['type'] == "inhibitory"]["node"]
    
    
    #### get graph / synapses
    out_edges = np.array(G.edges) 
        
    if (out_edges.shape[1] == 3): # the case for scale free graphs
        out_edges = np.delete(out_edges, 2, 1) # drop third colum "weights if mutligraph"
        
    synapses = pd.DataFrame(out_edges, columns = ['id_from','id_to'])
    
    synapses['s_type'] = None
    
    ### assign synapse type
    ## assign inhibitory -> inhibitory
    synapses = synapses.assign(
        s_type=np.where(
            np.logical_and(synapses['id_from'].isin(in_neurons),
                              synapses['id_to'].isin(in_neurons)),
            'ii',synapses['s_type']))
    
    ## assign excitatory --> excitatory
    synapses = synapses.assign(
        s_type=np.where(
            np.logical_and(synapses['id_from'].isin(ex_neurons),
                              synapses['id_to'].isin(ex_neurons)),
            'ee',synapses['s_type']))
    
    ## assign inhibitory --> excitatory
    synapses = synapses.assign(
        s_type=np.where(
            np.logical_and(synapses['id_from'].isin(in_neurons),
                              synapses['id_to'].isin(ex_neurons)),
            'ie',synapses['s_type']))
    ## assign excitatory --> inhibitory
    synapses = synapses.assign(
        s_type=np.where(
            np.logical_and(synapses['id_from'].isin(ex_neurons),
                              synapses['id_to'].isin(in_neurons)),
            'ei',synapses['s_type']))
    
    
    ## add sub-synapse group indices for brian2
    # assign simulation id to be used by brain2 later
    neurons['sim_id'] = neurons.groupby(['type']).cumcount()
    
    #drop unnessary data
    neurons =  neurons[["node", "sim_id"]]
    
    # add simulation neuron id for outgoing
    synapses = synapses.merge(neurons, left_on='id_from', right_on='node')
    synapses.drop('node', inplace=True, axis=1) # drop common column
    synapses.rename(columns = {'sim_id': 'sim_id_from'}, inplace = True) #rename 
    
    # add simulation neuron id for incoming
    synapses = synapses.merge(neurons, left_on='id_to', right_on='node')
    synapses.drop('node', inplace=True, axis=1) # drop common column
    synapses.rename(columns = {'sim_id': 'sim_id_to'}, inplace = True) #rename 

    
    ## check if all types of synapses are there
    if (synapses['s_type'].unique()).size < 4:
        synapsesMissing =  set({'ee', 'ei', 'ie', 'ii'}) - set(synapses['s_type']) 
        raise Exception(f"This network does not have all synapse types. Synapses of types: {synapsesMissing} are missing. Choose a differnt seed.")
        
    return synapses



###############################################################################
###############################################################################
if __name__ == '__main__':

    ###########################################################################
    ####################### Graph Generation ##################################

    # no of nodes
    n = 100

    scaleFreeGraph = True
    randomGraph = True
    
    graphType = ""
    
    seed1 = 4712
    

    if scaleFreeGraph: 
        # make a scale free graph of defined parameters
    
        a=0.26 #0.41    # Prob of adding new node with connection --> existing node
        b=0.54          # Prob of adding edge from existing node to existing node
        g=0.20 #0.05     # Prob of adding new node with connection <-- existing node
        
        d_in = 0.2
        d_out = 0
    
        genParams = {'alpha':a, 'beta':b, 'gamma':g, 'delta_in':d_in, 'delta_out':d_out}
        graphType = "scale-free"
   
        G = scale_free_net(n,a,b,g,d_in,d_out,seed1)

        ######################## Vizailize Graphs #################################
        # plot graphs
        ######################### Analyze Graphs ##################################
        # calculate and visualize graph characteristics
        
        GInspector = netInspector(G, graphType, genParams)
        GInspector.eval_all()
        GInspector.analyticsPanel()
            
        
        
    if randomGraph:
        
        p = 0.02 # Probability for edge creation
        
        graphType = "random"
        
        genParams = {'p':p}
    
        G = random_net(n,p,seed1)
        
       

        ######################## Vizailize Graphs #################################
        # plot graphs
        ######################### Analyze Graphs ##################################
        # calculate and visualize graph characteristics
        
        GInspector = netInspector(G, graphType, genParams)
        GInspector.eval_all()
        GInspector.analyticsPanel()
        
        
            
    

    
    
    
    
        
    
    
    
    