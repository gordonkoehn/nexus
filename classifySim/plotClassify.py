#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:29:29 2022

@author: gordonkoehn

package containting various plotting functions for the classifyer scripts.
"""  

from classifySim.classifySimulations import *
from brian2 import *

#import matplotlib.pyplot as plt

from matplotlib import pyplot

def getRasterplot(result):

    ##### raster plot
    # filter for timeframe x-y ms
    x = 0
    y = 10000
    
    in_time_sub = result['in_time'][((result['in_time'] > x) & (result['in_time'] < y) )]
    in_idx_sub = result['in_idx'][((result['in_time'] > x) & (result['in_time'] < y) )]
    
    ex_time_sub = result['ex_time'][((result['ex_time'] > x) & (result['ex_time'] < y) )]
    ex_idx_sub = result['ex_idx'][((result['ex_time'] > x) & (result['ex_time'] < y) )]
    
    #plot
    Fig=plt.figure(figsize=(9,7))
    plt.subplots_adjust(hspace=0.7, wspace=0.4)
    
   # matplotlib.rc('xtick', labelsize=15) 
    #matplotlib.rc('ytick', labelsize=15) 
    
    
    figa=Fig.add_subplot()
    plt.title('Raster Plot', fontsize=15)
    plt.scatter( in_time_sub, in_idx_sub, color='red',s=5,label="FS")
    plt.scatter(  ex_time_sub, ex_idx_sub, color='green',s=5,label="RS")
    plt.legend(loc='best', fontsize=15)
    plt.xlabel('Time [ms]', fontsize=15)
    plt.ylabel('Neuron Index', fontsize=15) 
    
    #plt.savefig("rasterplot.svg", format = 'svg', dpi=300)
    
    
def getMeanFreqBoxplot(result):
    ##### mean fireing freq.
    
    ## individual neurons
    #fig = pyplot.figure(figsize=(9,4))
    
    sim_time = result['sim_time']/second
    
    index, counts = np.unique( result['ex_idx'], return_counts=True)
    rates_Hz_ex = np.asarray(counts/sim_time )
    #pyplot.scatter(rates_Hz_ex, index , s=15, c=[[.4,.4,.4]], label="RS")
    
    index, counts = np.unique(result['in_idx'], return_counts=True)
    rates_Hz_in = np.asarray(counts/sim_time)
    #pyplot.scatter(rates_Hz_in, index , s=15, c=[[.4,.4,.4]], label="FS")
    
    
    #pyplot.yticks(np.arange(0,result['NI']+result['NE'], 1)) # tick every 1 neuron(s)
    #pyplot.xlim([0,200])
    #pyplot.ylabel('IDs')
    #pyplot.xlabel('Mean firing freq. [Hz]')
    #pyplot.title('Mean firing frequency')
    #pyplot.grid()
    #pyplot.show()
    
    #### per neuron group
    fig7, ax7 = plt.subplots()
    
    #data = np.concatenate((rates_Hz_ex, rates_Hz_in))
    green_diamond = dict(markerfacecolor='g', marker='D')
    data = [rates_Hz_ex, rates_Hz_in]
    ax7.set_title('Mean Firing Frequency')
    bp = ax7.boxplot(data, flierprops=green_diamond)
    ax7.set_xticklabels(("excitatory", "inhibitory"), size=10)
    ax7.set_xlabel('Neuron Type')
    ax7.set_ylabel('mean firing frequency [Hz]')
    ax7.grid()
    
    #plt.savefig("meanFreq.svg", format = 'svg', dpi=300)
    
    
###############################################################################
###############################################################################
if __name__ == '__main__':    
    
    
    ######################################################
    ########### specify simulation parameters ############
    params = dict()
    params['sim_time'] = float(10)
    params['N'] = int(100)
    #conductance fixed -> to favorite
    params['ge'] = float(40) #40
    params['gi'] = float(80) #80
    
    #connection probabilities
    params['prob_Pee'] = float(0.02)
    params['prob_Pei'] = float(0.02)
    params['prob_Pii'] = float(0.02)
    params['prob_Pie'] = float(0.02)
    
    # adpation
    params['a'] = float(28)
    params['b'] = float(21)
    
    # replica
    params['replica']  = 3
    
  
    # get data
    save_name = getFilename(params)

    curr_dir = getPath(params)
    
    result = getResult_trun(curr_dir, save_name, params)
    
    getRasterplot(result)
    
    getMeanFreqBoxplot(result)
    
  