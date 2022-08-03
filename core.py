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
import os

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
import classifySim
import netGen

sys.path.append('tools')
import adEx_util

#disable warings -  mostly warning about usage of pandas and brian2
import warnings
warnings.filterwarnings("ignore")



def simClasInfer():
    """Run Simulation Classifcation and Connectivity Inference."""
     
    ###########################################################################
    ### program start welcome
    print("\n")
    print("==================================================================")
    print("===                  Welcome to Nexus                          ===")
    print("==================================================================\n")

    plt.close('all') # close all opd plots
        
   ############################################################################
   ######### Generate Network
    print("===== Generating Network ====\n")

    # no of nodes
    n = 100

    scaleFreeGraph = False
    randomGraph = True
    
    graphType = ""
    
    seed1 = 576
    

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
    print("= True Graph Properties: = ")
    # print("type: " + graphType)
    # print("Generation Parameters: " + str(genParams))
    
    ### Stats ###
    GInspector_true = netGen.analyzeNet.netInspector(G, graphType, genParams)
    GInspector_true.eval_all()
    print(GInspector_true)
        
    ### Plot ###
    ## plot infered graph 
    nodePosition_G_true = nx.kamada_kawai_layout(G)
    netGen.analyzeNet.draw_graph( G, title = "Ground Truth Graph", nodePos = nodePosition_G_true)

    #TODO: plot degree distributions of the true graph with fits if available
    GInspector_true.plotDegreeDist(title="Ground Truth Graph")
    
    ##########################################################################
    ######### Run Simulation
    print("\n===== Run AdEx Simulation ====\n")
    
    # generate synapses
    # get synapses    
    S, N = netGen.genNet.classifySynapses(G=G, inhibitoryHubs = False)
    
    
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
    
    # trunicate the result - discard the stimulation period
    result = classifySim.classifySimulations.truncResult(result)
    
    ###########################################################################
    ##### Classify Network Activity
    print("\n=== Classify Network Activity ===\n")
    
    #calculate network stats
    netActivityStats = classifySim.classifySimulations.classifyResult(result)
    print(f"Network dormant: {netActivityStats['dormant']}")
    if netActivityStats['dormant']: 
        raise Exception("Network is dormant - not point to continue")
      
        
    print(f"Network is recurrent: {netActivityStats['recurrent']}")
    if not netActivityStats['recurrent']: 
        raise Exception("Network is not recurrent - not point to continue")
        
    
    print(f"Mean firing freq [Hz]: {netActivityStats['m_freq'] : 2.1f}")
    print(f"Mean pairwise-correlation: {netActivityStats['m_pairwise_corr'] : 2.2f}")
    print(f"Mean coefficient of variation: {netActivityStats['m_cv'] : 2.1f}")
    print(f"Asynchronous: {netActivityStats['asynchronous']}")
    ### Plot ###
    ## Rasterplot
    classifySim.plotClassify.getRasterplot(result)
    ## Mean Firing Freq
    classifySim.plotClassify.getMeanFreqBoxplot(result)
    
    ##########################################################################
    ######### Connectivity Inference
    print("\n=== Connectivity Inference ===\n")
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
                           'alpha': .005} # default is 0.01
    
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
    
    # get infered graph + including thresholding by the significance value
    G_infered_nx = conInf.analyser.getInferedNxGraph(spycon_result)
  
    ### get theshold    
    print(f"Threshold: {spycon_result.threshold : .2f}")
    print(f"No of infered edged: {len(G_infered_nx.edges)}")
    

    #### Plot ####
    ## plot infered graph 
    # position of nodes is fixed to that of the groud true
    netGen.analyzeNet.draw_graph(G_infered_nx, title = "Inferred Graph" , nodePos = nodePosition_G_true)

    ## printout graph proberties choosen
    print("\n= Inferred Graph Properties: =\n ")
    # calculate
    #graphType = "inferred-scale-free"
    genParams = {}
    GInspector_infered = netGen.analyzeNet.netInspector(G_infered_nx, graphType, genParams)
    GInspector_infered.eval_all()
    print(GInspector_infered)
    
    #TODO: plot degree distributions of the infered graph with fits if available
    
      

    ## plot ROC
    conInf.output.plot_ROC(test_metrics)    
    
    ## plot all CCGs - for true edges
    conInf.output.plot_all_ccgs(coninf, spycon_test)
    
    ## plot single CCG
    #conInf.output.plot_ccg(coninf, spycon_test, 4)
    
    
    ###########################################################################
    ##################### CLOSE ALL PLOTS #####################################
    # plt.close('all')

###############################################################################
###############################################################################
if __name__ == '__main__':
    
    simClasInfer()
    
   #  ###########################################################################
   #  ### program start welcome
   #  print("\n")
   #  print("==================================================================")
   #  print("===                  Welcome to Nexus                          ===")
   #  print("==================================================================\n")

   #  plt.close('all') # close all opd plots
        
   # ############################################################################
   # ######### Generate Network
   #  print("===== Generating Network ====\n")

   #  # no of nodes
   #  n = 100

   #  scaleFreeGraph = True
   #  randomGraph = False
    
   #  graphType = ""
    
   #  seed1 = 576
    

   #  if scaleFreeGraph: 
   #      # make a scale free graph of defined parameters
    
   #      a=0.26 #0.41    # Prob of adding new node with connection --> existing node
   #      b=0.54          # Prob of adding edge from existing node to existing node
   #      g=0.20 #0.05     # Prob of adding new node with connection <-- existing node
        
   #      d_in = 0.2
   #      d_out = 0
    
   #      genParams = {'alpha':a, 'beta':b, 'gamma':g, 'delta_in':d_in, 'delta_out':d_out}
   #      graphType = "scale-free"
   
   #      G = netGen.genNet.scale_free_net(n,a,b,g,d_in,d_out,seed1)

        
   #  if randomGraph:
        
   #      p = 0.02 # Probability for edge creation
        
   #      graphType = "random"
        
   #      genParams = {'p':p}
    
   #      G = netGen.genNet.random_net(n,p,seed1)
    
   #  ## printout graph proberties choosen
   #  print("= True Graph Properties: = ")
   #  # print("type: " + graphType)
   #  # print("Generation Parameters: " + str(genParams))
    
   #  ### Stats ###
   #  GInspector_true = netGen.analyzeNet.netInspector(G, graphType, genParams)
   #  GInspector_true.eval_all()
   #  print(GInspector_true)
        
   #  ### Plot ###
   #  ## plot infered graph 
   #  nodePosition_G_true = nx.kamada_kawai_layout(G)
   #  netGen.analyzeNet.draw_graph( G, title = "Ground Truth Graph", nodePos = nodePosition_G_true)

   #  #tODO: plot degree distributions of the true graph with fits if available
    
   #  ##########################################################################
   #  ######### Run Simulation
   #  print("\n===== Run AdEx Simulation ====\n")
    
   #  # generate synapses
   #  # get synapses    
   #  S, N = netGen.genNet.classifySynapses(G=G, inhibitoryHubs = False)
    
    
   #  params = dict()
   #  params['sim_time'] = float(10)
   #  params['a'] = float(28)
   #  params['b'] = float(21)
   #  params['N'] = int(100)
   #  #conductances
   #  params['ge']=float(40)
   #  params['gi']=float(60)
   #  #connection probabilities
   #  params['synapses'] = S
   #  params['neurons'] = N
   
   #  ## printout adEx simulation properties choosen
   #  print("= AdEx neuron/network properties: = ")
   #  print("simulation time [s]: " + str(params['sim_time']))
   #  print("no of neurons: N=" + str(params['N']))
   #  print("adaption: a=" + str(params['a']) + " b="+ str(params['b']))
   #  print("conductance: ge=" + str(params['ge']) + " gi="+ str(params['gi']))
    
   #  ## run simulation
   #  print("\nstarting simulation...")
   #  result = simulations.wp2_adex_model_netX.run_sim(params)    
    
   #  print('simulation successfully finished for ' + result['save_name'])
    
   #  # trunicate the result - discard the stimulation period
   #  result = classifySim.classifySimulations.truncResult(result)
    
   #  ###########################################################################
   #  ##### Classify Network Activity
   #  print("\n=== Classify Network Activity ===\n")
    
   #  #calculate network stats
   #  netActivityStats = classifySim.classifySimulations.classifyResult(result)
   #  print(f"Network dormant: {netActivityStats['dormant']}")
   #  if netActivityStats['dormant']: 
   #      raise Exception("Network is dormant - not point to continue")
      
        
   #  print(f"Network is recurrent: {netActivityStats['recurrent']}")
   #  if not netActivityStats['recurrent']: 
   #      raise Exception("Network is not recurrent - not point to continue")
        
    
   #  print(f"Mean firing freq [Hz]: {netActivityStats['m_freq'] : 2.1f}")
   #  print(f"Mean pairwise-correlation: {netActivityStats['m_pairwise_corr'] : 2.2f}")
   #  print(f"Mean coefficient of variation: {netActivityStats['m_cv'] : 2.1f}")
   #  print(f"Asynchronous: {netActivityStats['asynchronous']}")
   #  ### Plot ###
   #  ## Rasterplot
   #  classifySim.plotClassify.getRasterplot(result)
   #  ## Mean Firing Freq
   #  classifySim.plotClassify.getMeanFreqBoxplot(result)
    
   #  ##########################################################################
   #  ######### Connectivity Inference
   #  print("\n=== Connectivity Inference ===\n")
   #  print("infer functional connectivity...")
    
   #  #############################
   #  ##### Unpack data ###########
   #  # unpack spikes
   #  times=np.append( result['in_time'],  result['ex_time']) # [s]
   #  ids=np.append(result['in_idx'],  result['ex_idx'])  
   #  nodes=np.arange(0, params['N'], 1)
    
   #  #############################
   #  ###### Infer. Fn. Conn ######
    
   #  #Note: Conncectvity can only be inferred for the neurons that have at least one spike.
   #  #      Neurons of no single spike are not shown in the graph.
    
   #  # conversions fom Brain2 --> sPYcon
   #  times_in_sec = (times) /1000 # convert to unitless times from [ms] -> [s]
    
    
   #  inference_params = {'binsize': 1e-3,
   #                         'hf': .6,
   #                         'gauss_std': 0.01,
   #                         'syn_window': (0,5e-3),
   #                         'ccg_tau': 20e-3,
   #                         'alpha': .005} # default is 0.01
    
   #  print("inference parameters:")
   #  print(str(inference_params))
    
   #  # define inference method
   #  coninf = Smoothed_CCG(params = inference_params) # English2017
    
   #  # get ground truth graph of network
   #  marked_edges, nodes = adEx_util.make_marked_edges_TwoGroups(ids,result['conn_ee'], result['conn_ei'], result['conn_ii'], result['conn_ie'])
    
   #  # define test
   #  spycon_test = ConnectivityTest("spycon_test",times_in_sec, ids, nodes, marked_edges)
   #  # run test
   #  print("Infering functional conncectivity...")#
   #  spycon_result, test_metrics = spycon_test.run_test(coninf, only_metrics=False, parallel=True,)
   #  print("succesfully infered the functional connectivity")
    
   #  # get infered graph + including thresholding by the significance value
   #  G_infered_nx = conInf.analyser.getInferedNxGraph(spycon_result)
  
   #  ### get theshold    
   #  print(f"Threshold: {spycon_result.threshold : .2f}")
   #  print(f"No of infered edged: {len(G_infered_nx.edges)}")
    

   #  #### Plot ####
   #  ## plot infered graph 
   #  # position of nodes is fixed to that of the groud true
   #  netGen.analyzeNet.draw_graph(G_infered_nx, title = "Inferred Graph" , nodePos = nodePosition_G_true)

   #  ## printout graph proberties choosen
   #  print("\n= Inferred Graph Properties: =\n ")
   #  # calculate
   #  #graphType = "inferred-scale-free"
   #  genParams = {}
   #  GInspector_infered = netGen.analyzeNet.netInspector(G_infered_nx, graphType, genParams)
   #  GInspector_infered.eval_all()
   #  print(GInspector_infered)
    
   #  #tODO: plot degree distributions of the infered graph with fits if available
    
      

   #  ## plot ROC
   #  conInf.output.plot_ROC(test_metrics)    
    
   #  ## plot all CCGs - for true edges
   #  conInf.output.plot_all_ccgs(coninf, spycon_test)
    
   #  ## plot single CCG
   #  #conInf.output.plot_ccg(coninf, spycon_test, 4)
    
    
   #  ###########################################################################
   #  ##################### CLOSE ALL PLOTS #####################################
   #  # plt.close('all')