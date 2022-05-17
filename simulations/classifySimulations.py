#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 21:32:13 2022

@author: gordonkoehn
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


def getFreq(result):
    """
    Get mean firing frequencies of a given file behin the given path
    
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
    root_dir = 'simData'

    curr_dir = root_dir + '/' 
    curr_dir += 'N_' +str(params['N']) + '/'  
    curr_dir +=  '_'.join(['p', str(params['prob_Pee']),str(params['prob_Pei']), str(params['prob_Pii']), str(params['prob_Pie'])])
    
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
    save_name = '_'.join([str(params['N']), str(params['a']),str(params['b']), str(params['sim_time']) ]) + '_'
    save_name += '_'.join([str(params['ge']), str(params['gi']),  str(params['prob_Pee']),str(params['prob_Pei']), str(params['prob_Pii']), str(params['prob_Pie']) ])
    save_name += ".npy"

    return save_name


def getResult(curr_dir, save_name):
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
    root_dir = 'simData'

    curr_dir = root_dir + '/' 
    curr_dir += 'N_' +str(params['N']) + '/'  
    curr_dir +=  '_'.join(['p', str(params['prob_Pee']),str(params['prob_Pei']), str(params['prob_Pii']), str(params['prob_Pie'])])

    result = np.load(curr_dir + '/' + save_name, allow_pickle=True)
    result = result.tolist() # convert from array back to dictionary
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
        cvs.append(cv(isi(neo_spk_train)))
    
    return cvs

def computePariwiseCorr(binned_spiketrains):
    """
    Calculate the NxN matrix of pairwise Pearson’s correlation coefficients between all combinations of N binned spike trains.
    
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
    
    # get neuron IDs - persume they jsut iterate upwards
    N = result['NE'] + result['NI']
    nodes=np.arange(0, N, 1)
    
    #results list of binned spiketrains
    spiketrains = []
    
    # iterate through all neurons
    for node in nodes:
        t_start = 0
        t_stop =  result['sim_time'] * 1000 # in ms
        times1 = times[ids == node]
        neo_spk_train = neo.SpikeTrain(times1, units='ms', t_start=t_start, t_stop=t_stop)
        #aclaulate and append coefficent of variation
        
        spiketrains.append(neo_spk_train)
    
    binned_spiketrains = BinnedSpikeTrain(spiketrains, bin_size=5 * quantities.ms)
    
    return binned_spiketrains


###############################################################################
###############################################################################
if __name__ == '__main__':
    
    ######################################################
    ########### specify simulation parameters ############
    params = dict()
    params['sim_time'] = float(10)
    params['a'] = float(1)
    params['b'] = float(5)
    params['N'] = int(100)
    
    #connection probabilities
    params['prob_Pee'] = float(0.02)
    params['prob_Pei'] = float(0.02)
    params['prob_Pii'] = float(0.02)
    params['prob_Pie'] = float(0.02)
    #params['ge']=float(0)
    #params['gi']=float(0)
    
    
    # specify conductance space
    gi_min = 0
    gi_max = 100 # should be 100
    ge_min = 0
    ge_max = 10 # should be 100
    
    step = 5
    
    #######################################################
    
    
    ############## loop through all files ################
    ## define output lists
    # conductance values
    ges = [] 
    gis = []
    #mean firing frequencies
    m_freqs = []
    # coefficeit of varriation
    m_cvs = []
    # mean pairwise correlation
    m_pairwise_corrs = []   
    # loop through all conditions and analyze files
    for gi in np.arange(gi_min,gi_max+step,step):
        for ge in np.arange(gi_min,gi_max+step,step):
            # define parameters
            params['ge']=float(ge)
            params['gi']=float(gi)
            # get file
            save_name = getFilename(params)
    
            curr_dir = getPath(params)
            
            # get data
            result = getResult(curr_dir, save_name)
            
            ## analyze
            # get mean firing frequencies
            freqs = getFreq(result)
    
            m_freq = np.mean(freqs)
            
            # get coefficient of variation
            cvs = getCVs(result)
            m_cv = np.mean(cvs)
            m_cvs.append(m_cv)
            
            #get mean pairwise correlation
            binned_spiketrains = getBinnedSpiketrains(result)
            cc_matrix = computePariwiseCorr(binned_spiketrains)
            total_pairwise_corr =cc_matrix.sum() - cc_matrix.trace() # get summ of correlation, excluing self correlation
            m_pairwise_corr = total_pairwise_corr / (cc_matrix.shape[0]*cc_matrix.shape[1] - cc_matrix.shape[1]) #  note the lengeth of the diagonal of a NxN matrix is always N
            
            
            #save data
            ges.append(ge)
            gis.append(gi)
            m_freqs.append(m_freq)
            m_pairwise_corrs.append(m_pairwise_corr)
       
            
    # create output dictonary
    condFreqSpace =  dict()
    condFreqSpace['ge'] = ges
    condFreqSpace['gi'] = gis
    # save frequency values but replace NAs with 0
    condFreqSpace['m_freqs'] = [0 if x != x else x for x in m_freqs] 

    
    #######################################################
    
    
    
    ################# Make Figures #######################
    
    
    ########## mean firing frequencies
    
    ##### make excitatory
    
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
 
    # Creating plot
    ax.scatter3D(condFreqSpace['ge'],condFreqSpace['gi'] , condFreqSpace['m_freqs'], color = "green")
    plt.title("conductance-frequency space")
    
    ax.set_xlabel('ge [nS]')
    ax.set_ylabel('gi [nS]')
    ax.set_zlabel('mean freq. [Hz]')
 
    # show plot
    plt.show()
    
    ######### show CV 
    
    # Creating figure
    fig2 = plt.figure(figsize = (10, 7))
 
    # Creating plot
    #plt.scatter(m_cvs,m_pairwise_corrs)
    
    #groupSyncronous = [] # contains classification
    #i=0
    #while i < len(m_cvs):
    #    if (m_cvs[i] > 1) and (m_pariswise_corrs[i] < 0.1):
    #        groupSyncronous.append(False)
    #    else:
    #        groupSyncronous.append(True)
    #    i += 1 
    
    

    plt.plot(m_cvs, m_pairwise_corrs, marker='o', linestyle='', label="", color = "blue")
    
    
    plt.title("coefficient of variation to pairwise correlation" )
    
    plt.xlabel('mean CV ')
    plt.ylabel('mean pairwise correlaion')

    #plt.legend()
    # show plot
    plt.show()
   
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    