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

import timeit

import argparse

"""
This code is adapted version of the original code of :
Susin, Eduarda, and Alain Destexhe. 2021. “Integration, Coincidence Detection and Resonance in Networks of Spiking Neurons Expressing Gamma Oscillations and Asynchronous States.” PLoS Computational Biology 17 (9): e1009416.

Alain Destexhe 2009. "Self-sustained asynchronous irregular states and Up–Downstates in thalamic, cortical and thalamocortical networksof nonlinear integrate-and-fire neurons", J Comput Neurosci (2009) 27:493–506DOI 10.1007/s10827-009-0164-4


"""



parser = argparse.ArgumentParser()
parser.add_argument("sim_time") # in seconds
parser.add_argument("N")
parser.add_argument("a")    # in nS
parser.add_argument("b")    # in pA
parser.add_argument("ge")
parser.add_argument("gi")
parser.add_argument("prob_Pee")
parser.add_argument("prob_Pei")
parser.add_argument("prob_Pii")
parser.add_argument("prob_Pie")
parser.add_argument("replica")

args=parser.parse_args()


params = dict()
params['sim_time'] = float(args.sim_time)
params['a'] = float(args.a)
params['b'] = float(args.b)
params['N'] = int(args.N)
#conductances
params['ge']=float(args.ge)
params['gi']=float(args.gi)
#connection probabilities
params['prob_Pee'] = float(args.prob_Pee)
params['prob_Pei'] = float(args.prob_Pei)
params['prob_Pii'] = float(args.prob_Pii)
params['prob_Pie'] = float(args.prob_Pie)
params['replica'] = int(args.replica)



root_dir = 'simData'

curr_dir = root_dir + '/' 
curr_dir += 'N_' +str(params['N']) + '/'  
curr_dir +=  '_'.join(['p', str(params['prob_Pee']),str(params['prob_Pei']), str(params['prob_Pii']), str(params['prob_Pie'])])
curr_dir += '/replica/'
                                      
if(not(os.path.exists(curr_dir))):
    os.makedirs(curr_dir)


params['save_fol'] = curr_dir



    

def run_sim(params):
    #========================================================================
    
    start_scope() # This is just to make sure that any Brian objects created before the function is called aren't included in the next run of the simulation.
    
    t_simulation = params['sim_time']*second
    
    defaultclock.dt = 0.1*ms
    
    
    ################################################################################
    #Network Structure
    ################################################################################
    
    N=params['N']
    
    NE=int(N*4./5.); NI=int(N/5.)
    
    #-----------------------------
    
    
    prob_Pee = params['prob_Pee'] #(RS->RS) NB: originally all were p=0.02
    prob_Pei = params['prob_Pei'] #(RS->FS)
    prob_Pii = params['prob_Pii'] #(FS->FS)
    prob_Pie = params['prob_Pie'] #(FS->RS)
    
    
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
    
    con_ee = Synapses(neuronsE, neuronsE, on_pre='ge_post += Gee')
    con_ee.connect(p=prob_Pee)
    con_ee.delay='rand()*5*ms'
    
    con_ii = Synapses(neuronsI, neuronsI, on_pre='gi_post += Gii')
    con_ii.connect(p=prob_Pii)
    con_ii.delay= 'rand()*5*ms'
       
      
    con_ie = Synapses(neuronsI, neuronsE, on_pre='gi_post += Gie')
    con_ie.connect(p=prob_Pie)
    con_ie.delay='rand()*5*ms' 
    
    con_ei = Synapses(neuronsE, neuronsI, on_pre='ge_post += Gei')
    con_ei.connect(p=prob_Pei)
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

    result['ex_idx']= NeuronIDE
    result['in_idx']= NeuronIDI + NE
    
    result['ex_time']= timeE
    result['in_time']= timeI
    
    
    #connections
    
    connected_ee = np.array((con_ee.i, con_ee.j))
    connected_ii = np.array((con_ii.i+NE, con_ii.j+NE))
    connected_ei = np.array((con_ei.i, con_ei.j+NE))
    connected_ie = np.array((con_ie.i+NE, con_ie.j))
    
    
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
    save_name += '_'.join([str(params['ge']), str(params['gi']),  str(params['prob_Pee']),str(params['prob_Pei']), str(params['prob_Pii']), str(params['prob_Pie']) ])
    save_name += '_' + str(params['replica'])
    result['save_name']= save_name
    np.save(save_fol + '/' + save_name, result)

    print('simulation successfullly ran for ' + save_name)    
    

    return result



   

if __name__ == '__main__':
    
    
    result = run_sim(params)    
    print('simulation successfully finished for ' + result['save_name'])