#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:02:22 2022

@author: gordonkoehn

package used by my conductance and adaptance space classify scipts
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import math
from elephant.statistics import cv, isi
import elephant
import neo
import quantities
from elephant.conversion import BinnedSpikeTrain
import pandas as pd


def getFreq(result):
    """
    Get mean firing frequencies of a given file behin the given path.
    
    Includes inactive neurons.
    
    Parameters
    ----------
    path : str
        path to .npy simulation file
    
    
    Returns
    -------
    freqs : np.array
        frequencies in Hz of spikes per excitatory and inhibitory neuron
    
    
    """    

    ##### Unpack data ###########
    # unpack spikes
    ids=np.append( result['in_idx'],  result['ex_idx'])  
    
    # get freq
    index, counts = np.unique(ids, return_counts=True)
    freqs = np.asarray(counts/result['sim_time']) #in Hz
    
    
    return freqs



def getPath(params):
    """
    Get path to file given the parameters of the simulation
    
    Parameters
    ----------
    params : dict
       see definition in wp2_adex_model_script
        
        
    Returns
    -------
    curr_dir : str
        path to .npy simulation file
    
    """   
    root_dir = '../simulations/simData'

    curr_dir = root_dir + '/' 
    curr_dir += 'N_' +str(params['N']) + '/'  
    curr_dir +=  '_'.join(['p', str(params['prob_Pee']),str(params['prob_Pei']), str(params['prob_Pii']), str(params['prob_Pie'])])
    
    if "replica" in params:
        curr_dir += '/replica'
    
    return curr_dir


def getFilename(params):
    """
    Get filename given the parameters of the simulation
    
    Parameters
    ----------
    params : dict
       see definition in wp2_adex_model_script
        
        
    Returns
    -------
    save_name : str
        name of simulation file
    
    """
    if "replica" in params:
        save_name = '_'.join([str(params['N']), str(params['a']),str(params['b']), str(params['sim_time']) ]) + '_'
        save_name += '_'.join([str(params['ge']), str(params['gi']),  str(params['prob_Pee']),str(params['prob_Pei']), str(params['prob_Pii']), str(params['prob_Pie']) ])
        save_name += '_' + str(params['replica'])
        save_name += ".npy"
    else:
        save_name = '_'.join([str(params['N']), str(params['a']),str(params['b']), str(params['sim_time']) ]) + '_'
        save_name += '_'.join([str(params['ge']), str(params['gi']),  str(params['prob_Pee']),str(params['prob_Pei']), str(params['prob_Pii']), str(params['prob_Pie']) ])
        save_name += ".npy"
         

    return save_name


def getResult(curr_dir, save_name, params):
    """
    Get simulation results given path and filename.
    
    Parameters
    ----------
    curr_dir : str
       path to file from current directory
       
    save_name : str
        name file was saved as
        
        
    Returns
    -------
    result : dict
        see wp2_adex_model_script.py
    
    """  
    result = np.load(curr_dir + '/' + save_name, allow_pickle=True)
    result = result.tolist() # convert from array back to dictionary
    return result

def getResult_trun(curr_dir, save_name, params):
    """
    Get simulation results given path and filename and trunicates the first 
    1000 ms to avoid including the driven period.
    
    Parameters
    ----------
    curr_dir : str
       path to file from current directory
       
    save_name : str
        name file was saved as
        
        
    Returns
    -------
    result : dict
        see wp2_adex_model_script.py
    
    """  
    root_dir = 'simData'

    #curr_dir = root_dir + '/' 
    #curr_dir += 'N_' +str(params['N']) + '/'  
    #curr_dir +=  '_'.join(['p', str(params['prob_Pee']),str(params['prob_Pei']), str(params['prob_Pii']), str(params['prob_Pie'])])

    result = np.load(curr_dir + '/' + save_name, allow_pickle=True)
    result = result.tolist() # convert from array back to dictionary
    
    # trunicate the first 1000ms
    y = 1000
    
    result_temp = dict()
    
    result_temp['in_time'] = result['in_time'][(result['in_time'] > y )]
    result_temp['in_idx'] = result['in_idx'][(result['in_time'] > y)]
    
    result_temp['ex_time'] = result['ex_time'][(result['ex_time'] > y)]
    result_temp['ex_idx'] = result['ex_idx'][(result['ex_time'] > y)]
    
    #write back to actual output dict
    result['in_time'] =  result_temp['in_time']
    result['in_idx'] =  result_temp['in_idx']
    
    result['ex_time'] =  result_temp['ex_time']
    result['ex_idx'] =  result_temp['ex_idx']
    
    return result

def truncResult(result):
    """
    Trunicate the first 1000 ms to avoid including the driven period.
    
    Parameters
    ----------
    result: dict
        as defined by adex scripts
        
    Returns
    -------
    result : dict
        see wp2_adex_model_script.py
    
    """      
    # trunicate the first 1000ms
    y = 1000
    
    result_temp = dict()
    
    result_temp['in_time'] = result['in_time'][(result['in_time'] > y )]
    result_temp['in_idx'] = result['in_idx'][(result['in_time'] > y)]
    
    result_temp['ex_time'] = result['ex_time'][(result['ex_time'] > y)]
    result_temp['ex_idx'] = result['ex_idx'][(result['ex_time'] > y)]
    
    #write back to actual output dict
    result['in_time'] =  result_temp['in_time']
    result['in_idx'] =  result_temp['in_idx']
    
    result['ex_time'] =  result_temp['ex_time']
    result['ex_idx'] =  result_temp['ex_idx']
    
    return result

    
def getCVs(result):
    """
    Get the coefficients of variation / standrad errors for all spike trains  / neurons.
    
    Parameters
    ----------
    result : dict
       see wp2_adex_model_script.py
          
        
    Returns
    -------
    cvs : list
        coefficients of variation of all spike trains
    
    """  
    ##### Unpack data ###########
    # unpack spikes
    times=np.append( result['in_time'],  result['ex_time']) # [s]
    ids=np.append(result['in_idx'],  result['ex_idx'])  
    
    # get neuron IDs - persume they jsut iterate upwards
    N = result['NE'] + result['NI']
    nodes=np.arange(0, N, 1)
    
    #results list of coefficients of variation
    cvs = []
    
    # iterate through all neurons
    for node in nodes:
        t_start = 0
        t_stop =  result['sim_time']
        times1 = times[ids == node] / 1000 # in s
        neo_spk_train = neo.SpikeTrain(times1, units=quantities.second, t_start=t_start, t_stop=t_stop)
        #aclaulate and append coefficent of variation
        cvs.append(cv(isi(neo_spk_train), nan_policy='omit'))
    
    #convert to numpy
    cvs = np.array(cvs)
    
    return cvs

def computePariwiseCorr(binned_spiketrains):
    """
    Calculate the NxN matrix of pairwise Pearson’s correlation coefficients between all combinations of N binned spike trains.
    
    Each entry in the matrix is a real number ranging between -1 (perfectly anti-correlated spike trains) and +1 (perfectly correlated spike trains).
    
    Parameters
    ----------
    trains : list 
        of binned_spiketrain      
        
    Returns
    -------
    cc_matrix : (N, N) np.ndarray
        The square matrix of correlation coefficients.
    """  
    cc_matrix =  elephant.spike_train_correlation.correlation_coefficient(binned_spiketrains, binary=False, fast=True)

    return cc_matrix


def getBinnedSpiketrains(result):
    """
    Makes a list of binned spietrains given ids and times.
    
    Note inactive neurons are kicked out here.
    
    Throws: Exception: if no two spiektrains can be found.
    
    Parameters
    ----------
    result : dict
       see wp2_adex_model_script.py
        
    Returns
    -------
    cc_matrix : (N, N) np.ndarray
        The square matrix of correlation coefficients.
    """  
    ##### Unpack data ###########
    # unpack spikes
    times=np.append( result['in_time'],  result['ex_time']) # [s]
    ids=np.append(result['in_idx'],  result['ex_idx'])  
    
    # get neuron IDs - persume they just iterate upwards
    N = result['NE'] + result['NI']
    nodes=np.arange(0, N, 1)
    
    #results list of binned spiketrains
    spiketrains = []
    
    # iterate through all neurons
    for node in nodes:
        t_start = 0
        t_stop =  result['sim_time'] * 1000 # in msget
        times1 = times[ids == node]
        
        # check if spiketrain has zero spikes --> kick out inactive neuron
        if (not (len(times1) == 0)):
            #make spiketrain
            neo_spk_train = neo.SpikeTrain(times1, units='ms', t_start=t_start, t_stop=t_stop)
            #save spike train
            spiketrains.append(neo_spk_train)
    
    if (len(spiketrains) < 2):
        raise Exception(f"No two spiketrains for a: {result['a']}, b: {result['a']}, ge : {result['ge']}, gi: {result['a']} - no pairwise correlation can be computed.")
   
    else:
        binned_spiketrains = BinnedSpikeTrain(spiketrains, bin_size=5 * quantities.ms)
    
    return binned_spiketrains

def getNameClassifyData(params,space):
    """Make save name for the classifyData resulst based on the choosen search space.
    
    Parameters
    ----------
    params : dict
       imcomplete parameters of simulations (excluding a,b,gi,ge, replica)
       
    space : 
        constins max and min parameters in a,b,gi,ge,replica, named aMin and alike
    Returns
    -------
    save_name : str
        recommended save name
    """  
    save_name = '_'.join(["N",str(params['N']), "t",str(params['sim_time']) ]) + '_'
    save_name += '_'.join(["probs",str(params['prob_Pee']),str(params['prob_Pei']), str(params['prob_Pii']), str(params['prob_Pie']) ])
    save_name += '_' + '_'.join(["a", str(space['aMin']), str(space['aMax']), "b", str(space['bMin']), str(space['bMax']), "gi", str(space['giMin']),str(space['giMax']), "ge", str(space['geMin']),str(space['geMax']) ,"rep",str(space['replica'])])
    save_name += ".pkl"           
    
    
    return save_name

def classifyResult(result):
    """Calculate the mean coeffieiencet of varriation, pairwirse correlation and more given a result of a of an adex simulation.
    
    Parameters
    ----------
    result : dict
       adEx simulation results
       
    Returns
    -------
    stats : dict
        stats['m_freq'] as mean firing freq
        stats['m_cv'] as mean coeff. of varriation
        stats['m_pairwise_corr'] as the mean pariwise correlation
        stats['dormant'] if simulation stricly dormant (no activity of ex/inhibitory)
        stats['recurrent'] if there is still ex. and in. activityafter 9900ms#
        stats['asynchronous'] classification of simulation
    """
    ## analyze
    # get mean firing frequencies
    freqs = getFreq(result)
    #skip NaN frequencies - inactive neurons
    freqs_noNaN =  freqs[np.logical_not(np.isnan(freqs))]
    #get mean
    m_freq = np.mean(freqs_noNaN)
    
    # get coefficient of variation
    cvs = getCVs(result)
    #skip NaN - inactive neurons
    cvs_noNaN =  cvs[np.logical_not(np.isnan(cvs))]
    #get mean
    m_cv = np.mean(cvs_noNaN)
    
    # results dictionary to return
    stats = dict()
    # save the data
    stats['m_freq'] = m_freq
    stats['m_cv'] = (m_cv)
    
    try:
        # get pairwise correlations
        binned_spiketrains = getBinnedSpiketrains(result)
       
        cc_matrix = computePariwiseCorr(binned_spiketrains)
        total_pairwise_corr =cc_matrix.sum() - cc_matrix.trace() # get sum of correlation, excluing self correlation (i.e. diagonal)
        m_pairwise_corr = total_pairwise_corr / (cc_matrix.shape[0]*cc_matrix.shape[1] - cc_matrix.shape[1]) #  note the lengeth of the diagonal of a NxN matrix is always N
            
        #save data
        stats['m_pairwise_corr'] = (m_pairwise_corr)
    
    except Exception as e:
        stats['m_pairwise_corr'] = None
        print(e)
    
    try:
        stats['dormant'] =  ((len(result['ex_idx']) == 0) and  (len(result['in_idx'])==0))
    except TypeError as e:
        print(e)
        print("Could not determine if dormant or not.")
        
    # Recurrent if there is activity after 9900 ms
    y = 9900 
    stats['recurrent'] = (len(result['in_time'][(result['in_time'] > y )]) > 0) & (len(result['ex_time'][(result['ex_time'] > y )]) > 0)
    try:
        stats['asynchronous'] =  ((stats['m_cv'] > 1) & (stats['m_pairwise_corr'] < 0.1))
    except TypeError as e:
        print(e)
        print("Possibly mean CV or Corr could not be calculated.")
        
    return stats
    
    



    
    