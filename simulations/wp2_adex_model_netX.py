#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 19:53:10 2022

@author: th


"""
import os
import sys
from brian2 import *
import numpy as np
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
import random
import pandas as pd


import timeit


import argparse



# custom modules
import netGen.genNet 
import classifySim.plotClassify



"""
This code is adapted version of the original code of :
Susin, Eduarda, and Alain Destexhe. 2021. “Integration, Coincidence Detection and Resonance in Networks of Spiking Neurons Expressing Gamma Oscillations and Asynchronous States.” PLoS Computational Biology 17 (9): e1009416.

Alain Destexhe 2009. "Self-sustained asynchronous irregular states and Up–Downstates in thalamic, cortical and thalamocortical networksof nonlinear integrate-and-fire neurons", J Comput Neurosci (2009) 27:493–506DOI 10.1007/s10827-009-0164-4


"""



    

def run_sim(params):
    #========================================================================

    
    root_dir = 'simData'

    curr_dir = root_dir + '/' 
    curr_dir += 'N_' +str(params['N']) + '/'  

                                          
    if(not(os.path.exists(curr_dir))):
        os.makedirs(curr_dir)


    params['save_fol'] = curr_dir
    
    start_scope() # This is just to make sure that any Brian objects created before the function is called aren't included in the next run of the simulation.
    
    t_simulation = params['sim_time']*second
    
    defaultclock.dt = 0.1*ms
    
    
    ################################################################################
    #Network Structure
    ################################################################################
    
    N=params['N']
    
    NE=int(N*4./5.); NI=int(N/5.)
    
    
    #-----------------------------

    
    # prob_p=0.02 #External, not used now external drive (kick-off drive) uses the same connection probability as above.
    
    
    ################################################################################
    #Reescaling Synaptic Weights based on Synaptic Decay
    ################################################################################
    
    tau_i= 10*ms; tau_e= 5*ms # follows Destexhe 2009. 
    
    #----------------------------
    #Reference synaptic weights
    #----------------------------
    
    # Ge_extE_r=4*nS #(External in RS)
    # Ge_extI_r=4*nS #(External in FS)
    
    Gee_r=params['ge']*nS  #(RS->RS) 6ns for destexhe 2009
    Gei_r=params['ge']*nS  #(RS->FS) 
    
    Gii_r=params['gi']*nS #(FS->FS) #67nS for Destexhe (for small scale simulations << 10000 neurons)
    Gie_r=params['gi']*nS #(FS->RS)
    
    
    #-----------------------------
    #This allows to study the effect of the time scales alone
    
    tauI_r= 10.*ms; tauE_r= 5.*ms #References time scales
    
    # Ge_extE=Ge_extE_r*tauE_r/tau_e 
    # Ge_extI=Ge_extI_r*tauE_r/tau_e
    Gee=Gee_r*tauE_r/tau_e
    Gei=Gei_r*tauE_r/tau_e
    
    Gii=Gii_r*tauI_r/tau_i 
    Gie=Gie_r*tauI_r/tau_i 
    
    
    ################################################################################
    #Neuron Model 
    ################################################################################
    
    #######Parameters#######
    
    V_reset=-60.*mvolt; VT=-50.*mV
    Ei= -80.*mvolt; Ee=0.*mvolt; t_ref=5*ms
    C = 200 * pF; gL = 10 * nS  # C = 1 microF gL = 0.05 mS when S (membrane area) = 20,000 um^2
    
    tauw=600*ms #600 for Destexhe 2009. 500~600
    
    Delay= 1.5*ms
    
    #######Eleaky Heterogenities#######
    
    Eleaky_RS=np.full(NE,-60)*mV
    Eleaky_FS=np.full(NI,-60)*mV
    
    ########Equation#########
    
    eqs= """
    dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + ge*(Ee-v)+ gi*(Ei-v) - w)/C : volt (unless refractory)
    IsynE=ge*(Ee-v) : amp
    IsynI=gi*(Ei-v) : amp
    dge/dt = -ge/tau_e : siemens
    dgi/dt = -gi/tau_i : siemens
    dw/dt = (a*(v - EL) - w)/tauw : amp
    taum= C/gL : second
   
    a : siemens
    b : amp
    DeltaT: volt
    Vcut: volt
    EL : volt
    """
    
    
    ###### Initialize neuron group#############
    
    #FS
    neuronsI = NeuronGroup(NI, eqs, threshold='v>Vcut',reset="v=V_reset; w+=b", refractory=t_ref)
    neuronsI.a=0.*nS; neuronsI.b=0.*pA; neuronsI.DeltaT = 2.5*mV; neuronsI.Vcut = VT 
    neuronsI.EL=Eleaky_FS
    
    
    #RS
    neuronsE = NeuronGroup(NE, eqs, threshold='v>Vcut',reset="v=V_reset; w+=b", refractory=t_ref)
    neuronsE.a=params['a']*nS; neuronsE.b=params['b']*pA; neuronsE.DeltaT = 2.5*mV; neuronsE.Vcut = VT  #a = 1ns, b = 0.04nA for Destexhe 2009. # b = [40, 0] for grid search
    neuronsE.EL=Eleaky_RS
    
    ############################################################################################
    #Initial conditions
    ############################################################################################
    
    #Random Membrane Potentials
    neuronsI.v=np.random.uniform(low=-60,high=-50,size=NI)*mV
    neuronsE.v=np.random.uniform(low=-60,high=-50,size=NE)*mV
    
    #Conductances
    neuronsI.gi = 0.*nS;       neuronsI.ge = 0.*nS
    neuronsE.gi = 0.*nS;       neuronsE.ge = 0.*nS
    
    #Adaptation Current
    neuronsI.w = 0.*amp; neuronsE.w = 0.*amp
    
    
   

    
    #==========================================================================
    #External Current (not being used for now.)
    #==========================================================================
    
    # neuronsI.I = 0.*namp
    # neuronsE.I = 0.*namp
    
    ##########################################################################################
    #Synaptic Connections
    ############################################################################################
    
    #===========================================
    #FS-RS Network (AI Network)
    #===========================================
    #### assign connections
    
    #### get synapse groups
    synapses = params['synapses']
    
    
    graph_ee = synapses[synapses['s_type']=='ee']
    graph_ii = synapses[synapses['s_type']=='ii']
    graph_ie = synapses[synapses['s_type']=='ie']
    graph_ei = synapses[synapses['s_type']=='ei']
    
    # # adjust indixes of inhibitory to be between 0-19
    # with pd.option_context('mode.chained_assignment', None):
        
    #     graph_ii.loc[:, 'id_from'] =   graph_ii['id_from'].apply(lambda x: x - NE)   
    #     graph_ii.loc[:, 'id_to']   =   graph_ii['id_to'].apply(lambda x: x - NE)
    #     graph_ie.loc[:, 'id_from'] =   graph_ie['id_from'].apply(lambda x: x - NE)
    #     graph_ei.loc[:, 'id_to']   =   graph_ei['id_to'].apply(lambda x: x - NE)

    #ä double check that there are synapses of all types
    if  (graph_ee.shape[0] == 0) : # are there any connectiosn to make
        raise Exception("Cannot simulate - not a single  Excitory -> Excitory synapse")
        
    if  (graph_ii.shape[0] == 0) : # are there any connectiosn to make
        raise Exception("Cannot simulate - not a single  Inhibitory --> Inhibitory synapse")
        
    if  (graph_ie.shape[0] == 0) : # are there any connectiosn to make
        raise Exception("Cannot simulate - not a single  Inhibitory --> Excitatory synapse")
        
    if  (graph_ei.shape[0] == 0) : # are there any connectiosn to make
        raise Exception("Cannot simulate - not a single  Excitory --> Inhibitory synapse")
        
    ### Define Synapses     
    
    # Excitory --> Excitory
    con_ee = Synapses(neuronsE, neuronsE, on_pre='ge_post += Gee')
    con_ee.connect(i=graph_ee['sim_id_from'].to_list(), j=graph_ee['sim_id_to'].to_list())
    con_ee.delay='rand()*5*ms'
    
    # Inhibitory --> Inhibitory
    con_ii = Synapses(neuronsI, neuronsI, on_pre='gi_post += Gii')
    con_ii.connect(i=graph_ii['sim_id_from'].to_list(), j=graph_ii['sim_id_to'].to_list())
    con_ii.delay= 'rand()*5*ms'
       
    # Inhibitory --> Excitatory
    con_ie = Synapses(neuronsI, neuronsE, on_pre='gi_post += Gie')
    con_ie.connect(i=graph_ie['sim_id_from'].to_list(), j=graph_ie['sim_id_to'].to_list())
    con_ie.delay='rand()*5*ms' 
    
    # Excitory --> Inhibitory
    con_ei = Synapses(neuronsE, neuronsI, on_pre='ge_post += Gei')
    con_ei.connect(i=graph_ei['sim_id_from'].to_list(), j=graph_ei['sim_id_to'].to_list())
    con_ei.delay='rand()*5*ms'

    
    ##########################################################################################
    #initial excitation
    ############################################################################################
    
    stimulus = TimedArray(np.array([200,0])*Hz, dt=100.*ms)
    P = PoissonGroup(1, rates='stimulus(t)')
    
    con_pe = Synapses(P, neuronsE, on_pre='ge_post += Gee', delay=Delay)
    con_pe.connect(p=1)
    
    spikemonP = SpikeMonitor(P,variables='t')
    
    con_pi = Synapses(P, neuronsI, on_pre='ge_post += Gei', delay=Delay)
    con_pi.connect(p=1)
    
    
    ########################################################################################
    # Simulation
    ########################################################################################
    
    #Recording informations from the groups of neurons
    
    #FS
    # statemonI = StateMonitor(neuronsI, ['v'], record=[0])
    spikemonI = SpikeMonitor(neuronsI, variables='t') 
    
    
    #RS
    # statemonE = StateMonitor(neuronsE, ['v'], record=[0])
    spikemonE = SpikeMonitor(neuronsE, variables='t') 
    
    starttime = timeit.default_timer()
    # print("The start time is :",starttime)
    run(t_simulation) 
    comp_time = timeit.default_timer() - starttime
    print("The time difference is :", comp_time)
    
    print(np.array(spikemonE.t/ms))
    print(np.array(spikemonP.t/ms))
    
    # plt.plot(np.array(statemonE.v).ravel()[:1500])
    
    ####################################################################################################
    #save simulation results
    ####################################################################################################
    
    
    result = dict()
    
    result['comp_time']= comp_time
    
  
    NeuronIDE=np.array(spikemonE.i)
    NeuronIDI=np.array(spikemonI.i)
    
    timeE=np.array(spikemonE.t/ms) #time in ms
    timeI=np.array(spikemonI.t/ms)

    # TODO: get back the network IDs - not simulation ids
  

    result['ex_idx']= NeuronIDE
    result['in_idx']= NeuronIDI + NE
    
    result['ex_time']= timeE
    result['in_time']= timeI
    
    
    #connections
    connected_ee = np.array((con_ee.i, con_ee.j))
    connected_ii = np.array((con_ii.i+NE, con_ii.j+NE))
    connected_ei = np.array((con_ei.i, con_ei.j+NE))
    connected_ie = np.array((con_ie.i+NE, con_ie.j))
    
    #
    
    result['conn_ee'] = connected_ee
    result['conn_ii'] = connected_ii
    result['conn_ei'] = connected_ei
    result['conn_ie'] = connected_ie
    
    result['params'] = params
    
    result['NE'] = NE
    result['NI'] =  NI
    result['sim_time'] = t_simulation


    
    if(not(os.path.exists(params['save_fol']))):
        os.makedirs(params['save_fol'])
        
    save_fol = params['save_fol']
    save_name = '_'.join([str(params['N']), str(params['a']),str(params['b']), str(params['sim_time']) ]) + '_'
    result['save_name']= save_name
    np.save(save_fol + '/' + save_name, result)

    print('simulation successfullly ran for ' + save_name)    
    

    return result



   

if __name__ == '__main__':
    
    # generate graph
    #G = netGen.genNet.random_net() 
    G = netGen.genNet.scale_free_net(a=0.26, b=0.54, g=0.20)
    
    # generate synapses
    # get synapses    
    S = netGen.genNet.classifySynapses(G=G, inhibitoryHubs = True)

    
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
   
    
    result = run_sim(params)    
    
    classifySim.plotClassify.getRasterplot(result)
    
    classifySim.plotClassify.getMeanFreqBoxplot(result)
    
    
    print('simulation successfully finished for ' + result['save_name'])