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


def getFreq(path):
    
    result = np.load('simData/100/'+ save_name, allow_pickle=True)
    result = result.tolist() # convert from array back to dictionary

    index, counts = np.unique( result['ex_idx'], return_counts=True)
    rates_Hz_ex = np.asarray(counts/1000)


    index, counts = np.unique(result['in_idx'], return_counts=True)
    rates_Hz_in = np.asarray(counts/1000)
    
    
    #fig7, ax7 = plt.subplots()

 
    #green_diamond = dict(markerfacecolor='g', marker='D')
    #data = [rates_Hz_ex, rates_Hz_in]
    #ax7.set_title('Mean Firing Frequency')
    #bp = ax7.boxplot(data, flierprops=green_diamond)
    #ax7.set_xticklabels(("excitatory", "inhibitory"), size=10)
    #ax7.set_xlabel('Neuron Type')
    #ax7.set_ylabel('Hz')
    #ax7.grid()
    #fig7.show()
    

    return rates_Hz_ex, rates_Hz_in



def getFilename(params):
    
    save_name = '_'.join([str(params['N']), str(params['a']),str(params['b']), str(params['sim_time']), str(params['ge']), str(params['gi']) + ".npy"])
  
    return save_name


if __name__ == '__main__':
    
    # specify simulation parameters
    params = dict()
    params['sim_time'] = float(10)
    params['a'] = float(1)
    params['b'] = float(5)
    params['N'] = int(100)
    #params['ge']=float(0)
    #params['gi']=float(0)
    
    
    # specify conductance space
    gi_min = 0
    gi_max = 100 # should be 100
    ge_min = 0
    ge_max = 100 # should be 100
    
    step = 5
    
    
    ges = []
    gis = []
    m_freqs_ex = []
    m_freqs_in = []
    
    for gi in np.arange(gi_min,gi_max+step,step):
        for ge in np.arange(gi_min,gi_max+step,step):
            params['ge']=float(ge)
            params['gi']=float(gi)
    
            save_name = getFilename(params)
    
            rates_Hz_ex, rates_Hz_in = getFreq("save_name")
    
            m_rates_Hz_ex = np.mean(rates_Hz_ex)
            m_rates_Hz_in = np.mean(rates_Hz_in)
            
            ges.append(ge)
            gis.append(gi)
            m_freqs_ex.append(m_rates_Hz_ex)
            m_freqs_in.append(m_rates_Hz_in)
            
    # create output dictonary
    condFreqSpace =  dict()
    condFreqSpace['ge'] = ges
    condFreqSpace['gi'] = gis
    # save frequency values but replace NAs with 0
    condFreqSpace['m_freq_ex'] = [0 if x != x else x for x in m_freqs_ex] 
    condFreqSpace['m_freq_in'] = [0 if x != x else x for x in m_freqs_in] 
    
    print(condFreqSpace)
    
    
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
 
    # Creating plot
    ax.scatter3D(condFreqSpace['ge'],condFreqSpace['gi'] , condFreqSpace['m_freq_ex'], color = "green")
    plt.title("conductance-frequency space for excitatory")
    
    ax.set_xlabel('ge [nS]')
    ax.set_ylabel('gi [nS]')
    ax.set_zlabel('mean freq. [Hz]')
 
    # show plot
    plt.show()
    
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
 
    # Creating plot
    ax.scatter3D(condFreqSpace['ge'],condFreqSpace['gi'] , condFreqSpace['m_freq_in'], color = "green")
    plt.title("conductance-frequency space for inhibitory")
    
    ax.set_xlabel('ge [nS]')
    ax.set_ylabel('gi [nS]')
    ax.set_zlabel('mean freq. [Hz]')
 
    # show plot
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    