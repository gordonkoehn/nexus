#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User-interface to generate nets, run simulations, classify, infer-connecitify.

Created on Mon Aug  1 13:59:48 2022

@author: gordonkoehn
"""
# import common stuff
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx

# import inference method
sys.path.append('tools/spycon/src')
from sci_sccg import Smoothed_CCG
### get conncectivity test 
from spycon_tests import load_test, ConnectivityTest

# custom modules
import netGen.genNet
import simulations.wp2_adex_model_netX
import conInf.output 
import conInf.analyser

sys.path.append('tools')
import adEx_util as  adEx_util



###############################################################################
###############################################################################
if __name__ == '__main__':
    
    ###########################################################################
    ### program start welcome
    print("\n")
    print("==================================================================")
    print("===                  Welcome to Nexus                          ===")
    print("==================================================================\n")
    
    # print("Instructions:\n")
    # print("1) specify Network Generation\n2) specify AdEx Model \n--> Run Simulation\n")
    
    # ###########################################################################
    
    # print("=== Network Generation ===") 
    
    # #### graph type
    # graphType_s = {'s','S','scale', 'scale-free', 'sf'}
    # graphType_r = {'r','R','random'}
    
    # graphType = None
    
    # graphType_in = input("Specify the graph type - scale-free / random (s/r): ")
        
    # if (graphType in graphType_s):
    #     graphType = "scale-free"
    # elif (graphType in graphType_r):
    #     graphType = "random"
    # else:
    #     graphType = "random"
    #     print("Graph type not recognized, choose default: " + graphType)
        
    # #### choose graph parameters
    
    # if (graphType == "scale-free"):
        
    
        
   ############################################################################
   ######### Generate Network
    print("===== Generating Network ====\n")

    # no of nodes
    n = 100

    scaleFreeGraph = False
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
   
        G = netGen.genNet.scale_free_net(n,a,b,g,d_in,d_out,seed1)

        
    if randomGraph:
        
        p = 0.02 # Probability for edge creation
        
        graphType = "random"
        
        genParams = {'p':p}
    
        G = netGen.genNet.random_net(n,p,seed1)
    
    ## printout graph proberties choosen
    print("= Graph Properties: = ")
    print("type: " + graphType)
    print("Generation Parameters: " + str(genParams))

    ##########################################################################
    ######### Run Simulation
    print("\n===== Run AdEx Simulation ====\n")
    
    # generate synapses
    # get synapses    
    S, N = netGen.genNet.classifySynapses(G=G, inhibitoryHubs = True)
    
    
    params = dict()
    params['sim_time'] = float(10)
    params['a'] = float(28)
    params['b'] = float(21)
    params['N'] = int(100)
    #conductances
    params['ge']=float(40)
    params['gi']=float(60)
    #connection probabilities
    params['synapses'] = S
    params['neurons'] = N
   
    ## printout adEx simulation properties choosen
    print("= AdEx neuron/network properties: = ")
    print("simulation time [s]: " + str(params['sim_time']))
    print("no of neurons: N=" + str(params['N']))
    print("adaption: a=" + str(params['a']) + " b="+ str(params['b']))
    print("conductance: ge=" + str(params['ge']) + " gi="+ str(params['gi']))
    
    ## run simulation
    print("\nstarting simulation...")
    result = simulations.wp2_adex_model_netX.run_sim(params)    
    
    print('simulation successfully finished for ' + result['save_name'])
    
    ###########################################################################
    ##### Classify Network Activity
    
    
    # TODO: Calculate mean firing freq
    # TODO: claculate corr
    # TODO: clacualte COV
    
    
    ##########################################################################
    ######### Connectivity Inference
    print("\n=== Connectivity Inference ===")
    print("infer functional connectivity...")
    
    #############################
    ##### Unpack data ###########
    # unpack spikes
    times=np.append( result['in_time'],  result['ex_time']) # [s]
    ids=np.append(result['in_idx'],  result['ex_idx'])  
    nodes=np.arange(0, params['N'], 1)
    
    #############################
    ###### Infer. Fn. Conn ######
    
    #Note: Conncectvity can only be inferred for the neurons that have at least one spike.
    #      Neurons of no single spike are not shown in the graph.
    
    # conversions fom Brain2 --> sPYcon
    times_in_sec = (times) /1000 # convert to unitless times from [ms] -> [s]
    
    
    inference_params = {'binsize': 1e-3,
                           'hf': .6,
                           'gauss_std': 0.01,
                           'syn_window': (0,5e-3),
                           'ccg_tau': 20e-3,
                           'alpha': .01}
    
    print("inference parameters:")
    print(str(inference_params))
    
    # define inference method
    coninf = Smoothed_CCG(params = inference_params) # English2017
    
    # get ground truth graph of network
    marked_edges, nodes = adEx_util.make_marked_edges_TwoGroups(ids,result['conn_ee'], result['conn_ei'], result['conn_ii'], result['conn_ie'])
    
    # define test
    spycon_test = ConnectivityTest("spycon_test",times_in_sec, ids, nodes, marked_edges)
    # run test
    print("Infering functional conncectivity...")#
    spycon_result, test_metrics = spycon_test.run_test(coninf, only_metrics=False, parallel=True,)
    print("succesfully infered the functional connectivity")
    
    ### get theshold    
    print("Threshold: " + str(spycon_result.threshold))
    
    G_infered_nx = conInf.analyser.getInferedNxGraph(spycon_result)
    
    print(f"No of infered edged: {len(G_infered_nx.edges)}")

    netGen.analyzeNet.draw_graph(G_infered_nx, title = "Inferred Graph")

    ## plot ROC
    conInf.output.plot_ROC(test_metrics)    
    
    ## plot all CCGs - for true edges
    conInf.output.plot_all_ccgs(coninf, spycon_test)
    
    ## plot single CCG
    #conInf.output.plot_ccg(coninf, spycon_test, 4)