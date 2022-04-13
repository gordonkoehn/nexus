#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:40:21 2022

@author: gordonkoehn


This scritpt shall contain all generic functions to run adEx neuron net simulaitons.

"""

#### GENERAL IMPORT #####
from brian2 import *



#### GENERAL METHDOS ######


def simpleNetV2(n,p,c,t,ws,tauw,a,b,Vr): # n = no neurons, p = probablitiy of connection
    """
    Run a brian2 simulation of a network of n neurons connected with prob. p under condition c.
    
    Parameters
    ----------
    n : int
        no of neurons in network
    p : float
        probability of connection between neurons
    c : str
        condition of conncetion - see brain2
    t : int
        simulation time [in ms]
    ws : float
        voltage suppiled by a synaptic connection [*mV - brian2 units]
    tauw : float
        adaption time constant [*ms - in brian2 units]
    a : float
        subthreshold adaption constant [*nS - in brian2 units]
    b : float
        spike-triggert adaption constant [*nA - in brian2 units]
    Vr : float
        reset voltage [*mV- in brian2 units]
    
    
    Returns
    -------
    trace : StateMonitor
        contains w,v,I  
    spikes : SpikeMonitor
        records spike-times and ids
    S : Synapses
        holds true network definition  
    G :  Group of Neurons
        holds neuron group definition
    
    """    
    start_scope()
    
    #(1) ===========================Set Parameters======================================
    
    # Parameters
    C = 281 * pF              # membrane capacitance
    gL = 30 * nS              # leak conductance
    #taum = C / gL            # ? NOT USED ? ~  time to total leakage of current
    EL = -70.6 * mV           # leak reversal potential / resting potential
    VT = -50.4 * mV           # threshold potential
    DeltaT = 2 * mV           # slope factor (sharpness of spike)
    Vcut = VT + 5 * DeltaT    # computational voltage cutoff (not biologically important)
    
    
    # (2) =========================Define model and neurons/network=======================
    # define the two core equations of the adEx model
    eqs = """
    dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + I - w)/C : volt
    dw/dt = (a*(v - EL) - w)/tauw : amp
    I : amp
    """
    # define n neuron with voltage cutoff, reset potential and numerical euler integration
    G = NeuronGroup(n, model=eqs, threshold='v>Vcut',
                         reset="v=Vr; w+=b", method='euler')
    # initialize voltage of neuron as resting potential
    G.v = EL
    
    ## Make a connection between the two Synapses
    # let ws be the weight between synapeses
    S = Synapses(G, G,'ws : volt', on_pre='v_post += ws')
    S.connect(condition = c, p=p)
    S.ws = ws # voltage suppiled by a synaptic connection
    S.delay = '0.2*ms'
    
    
    # (3) ===========================Set up Recorders===================================
    # set up recorders
    trace = StateMonitor(G, ['v', 'w', 'I'], record=True) # record voltage of neuron 1 [NB!!:added w]
    spikes = SpikeMonitor(G)                # recorde spikes (times)
    
    
    # (4) ==========================Run Simulation=====================================
    # run simulation with changing driving current on neuron 1
    #run( * ms) 
    G.I = numpy.random.choice([0, 1], size=n, p=[.1, .9])*nA # with a 10% chance this neuron is driven with 1 nA
    run(t * ms)
    #G.I = 0*nA
    #run(20 * ms)
    
    return trace, spikes, S
    
    
