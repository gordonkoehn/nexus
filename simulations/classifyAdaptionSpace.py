#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:56:23 2022

@author: gordonkoehn
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import math
import pandas as pd

# module with all the functions and logic
from classifySimulations import *
# for progress bar
from tqdm import tqdm

from multiprocessing import Pool

from datetime import datetime

###############################################################################
###############################################################################
if __name__ == '__main__':
    
    ######################################################
    ########### specify simulation parameters ############
    params = dict()
    params['sim_time'] = float(10)
    params['N'] = int(100)
    #conductance fixed -> to favorite
    params['ge'] = float(85)
    params['gi'] = float(75)
    
    #connection probabilities
    params['prob_Pee'] = float(0.02)
    params['prob_Pei'] = float(0.02)
    params['prob_Pii'] = float(0.02)
    params['prob_Pie'] = float(0.02)
    #params['ge']=float(0)
    #params['gi']=float(0)
    
    
    # specify conductance space
    a_min = 55
    a_max = 56 # should be 85
    b_min = 55
    b_max = 56 # should be 85
    
    step = 1
    
    #######################################################
    ########## specify output #############################
    doCorrelation = True
    
    
    #######################################################
    
    
    
    ############## loop through all files ################
    ## define output lists
    # conductance values
    a_s = [] 
    b_s = []
    #mean firing frequencies
    m_freqs = []
    # coefficeit of varriation
    m_cvs = []
    # mean pairwise correlation
    m_pairwise_corrs = []   
    
    
    # no of states
    no_of_states = int(((a_max-a_min)/step) *((b_max-b_min)/step))
    
    with tqdm(total=no_of_states) as pbar:  # default setting
        # loop through all conditions and analyze files
        for a in np.arange(a_min,a_max+step,step):
            for b in np.arange(b_min,b_max+step,step):
                
                
                # define parameters
                params['a']=float(a)
                params['b']=float(b)
                # get file
                save_name = getFilename(params)
        
                curr_dir = getPath(params)
                
                try:
                    
                    # get data
                    result = getResult(curr_dir, save_name, params)
                    
                    ## analyze
                    # get mean firing frequencies
                    freqs = getFreq(result)
            
                    m_freq = np.mean(freqs)
                    
                    # get coefficient of variation
                    cvs = getCVs(result)
                    m_cv = np.mean(cvs)
                    m_cvs.append(m_cv)
                    
                  
                    
                    #get mean pairwise correlation
                    if doCorrelation:
                        binned_spiketrains = getBinnedSpiketrains(result)
                        cc_matrix = computePariwiseCorr(binned_spiketrains)
                        total_pairwise_corr =cc_matrix.sum() - cc_matrix.trace() # get sum of correlation, excluing self correlation (i.e. diagonal)
                        m_pairwise_corr = total_pairwise_corr / (cc_matrix.shape[0]*cc_matrix.shape[1] - cc_matrix.shape[1]) #  note the lengeth of the diagonal of a NxN matrix is always N
                        #save data
                        m_pairwise_corrs.append(m_pairwise_corr)
                        
                    
                    #save data
                    a_s.append(a)
                    b_s.append(b)
                    m_freqs.append(m_freq)
                except FileNotFoundError as fnf_error:
                    print(fnf_error)
                    print("File was skipped and condition excluded from analysis.")
                
            pbar.update(int(a_max-a_min)) # update progress bar
          
       
    # create output dictonary
    adapFreqSpace =  dict()
    adapFreqSpace['a'] = a_s
    adapFreqSpace['b'] = b_s
    # save frequency values but replace NAs with 0
    adapFreqSpace['m_freqs'] = [0 if x != x else x for x in m_freqs] 

    
    #######################################################
    
    
    
    ################# Make Figures #######################
    
    
    ########## mean firing frequencies
    
    ##### make excitatory
    
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
 
    # Creating plot
    ax.scatter3D(adapFreqSpace['a'],adapFreqSpace['b'] , adapFreqSpace['m_freqs'], color = "green")
    plt.title("adaptance-frequency space")
    
    ax.set_xlabel('a [nS]')
    ax.set_ylabel('b [pA]')
    ax.set_zlabel('mean freq. [Hz]')
    
    
    # # results dictionary for points that are physical
    # pointsPhysical = pd.DataFrame.from_dict(adapFreqSpace)
    # # 1 Hz < mean freq < 20 Hz 
    # pointsPhysical = pointsPhysical[(pointsPhysical['m_freqs'] > 1) & (pointsPhysical['m_freqs'] < 20 )]
    # pointsPhysical = pointsPhysical.sort_values(by = ['ge', 'gi'], ascending = [False, False], na_position = 'last')
    
    
    
    # #zipZip = zip(adapFreqSpace['ge'],adapFreqSpace['gi'] , adapFreqSpace['m_freqs'],zip(adapFreqSpace['ge'],adapFreqSpace['gi'],np.around(adapFreqSpace['m_freqs'], decimals=1)))
    
    # for row in pointsPhysical.head(5).itertuples(index=False):
    #     row = row._asdict()
    #     ax.text(row['ge'], row['gi'], row['m_freqs'], "("+ str(row['ge']) +", " +  str(row['gi']) +", " + str( np.around(row['m_freqs'],1)) + " )", size=8)
    
 
    # show plot
    plt.show()
    
    ######### show CV and Corr
    
    if doCorrelation : 
        #######################################################################
        #### do CV and corr plot
        # Creating figure
        fig2 = plt.figure(figsize = (10, 7))
     
        # Creating plot
        #plt.scatter(m_cvs,m_pairwise_corrs)
        
        groupSyncronous = [] # contains classification
        i=0
        while i < len(m_cvs):
            if (m_cvs[i] > 1) and (m_pairwise_corrs[i] < 0.1):
                groupSyncronous.append(False)
            else:
                groupSyncronous.append(True)
            i += 1 
        
        
        m_cvs = np.asarray(m_cvs)
        m_pairwise_corrs = np.asarray(m_pairwise_corrs)
        
        # make complete results array!!!
        classifyResults = dict()
        classifyResults['a'] = a_s
        classifyResults['b'] = b_s
        classifyResults['m_freqs'] = m_freqs
        classifyResults['m_cvs'] = m_cvs
        classifyResults['m_pairwise_corrs'] = m_pairwise_corrs
        classifyResults['syncronous']  = groupSyncronous
        
        
        # results dictionary for points that are physical
        classifyResults= pd.DataFrame.from_dict(classifyResults)
    
        # plot syncronous
        
        
        m_cvs_s = np.ma.masked_array(m_cvs, mask = not(groupSyncronous)).compressed()
        m_pairwise_corrs_s = np.ma.masked_array(m_pairwise_corrs, mask = not(groupSyncronous)).compressed()
    
        plt.plot(m_cvs_s, m_pairwise_corrs_s, marker='o', linestyle='', label="synchronous", color = "blue")
        
        
        #plot asyncronous
        m_cvs_a = np.ma.masked_array(m_cvs, mask = groupSyncronous).compressed()
        m_pairwise_corrs_a = np.ma.masked_array(m_pairwise_corrs, mask = groupSyncronous).compressed()
        
        plt.plot(m_cvs_a, m_pairwise_corrs_a, marker='o', linestyle='', label="asynchronous", color = "orange")
        
        
        
        plt.title("coefficient of variation to pairwise correlation" )
        
        plt.xlabel('mean CV ')
        plt.ylabel('mean pairwise correlaion')
    
        plt.legend()
        # show plot
        plt.show()
        
        
        
        ########################################################################
        #### show (a,b) grid with colored syncronous any asyncornous colored
        # Creating figure
        fig3 = plt.figure(figsize = (10, 7))
        
        maxFreq = 200 # Hz

        # syncronous
        plt.plot(classifyResults[(classifyResults['syncronous']==True) & (classifyResults['m_freqs']<=maxFreq)]['a'], classifyResults[(classifyResults['syncronous']==True) & (classifyResults['m_freqs']<=maxFreq)]['b'], marker='o', linestyle='', label="syncronous", color = "blue")
        
        #asyncornous
        plt.plot(classifyResults[(classifyResults['syncronous']==False) & (classifyResults['m_freqs']<=maxFreq)]['a'], classifyResults[(classifyResults['syncronous'] ==False) & (classifyResults['m_freqs']<=maxFreq)]['b'], marker='o', linestyle='', label="asyncronous", color = "orange")
        
        
        
        plt.title("adaptance (a,b) space by synchrony - mean freq <200Hz" )
        
        plt.xlabel('a [nS]')
        plt.ylabel('b [pA]')
    
        plt.legend()
        # show plot
        plt.show()
        
        #######################################################################
        #### show (a,b) grid with colored syncronous any asyncornous colored - 20 Hz
        # Creating figure
        fig3 = plt.figure(figsize = (10, 7))
        
        maxFreq = 20 # Hz

        # syncronous
        plt.plot(classifyResults[(classifyResults['syncronous']==True) & (classifyResults['m_freqs']<=maxFreq)]['a'], classifyResults[(classifyResults['syncronous']==True) & (classifyResults['m_freqs']<=maxFreq)]['b'], marker='o', linestyle='', label="syncronous", color = "blue")
        
        #asyncornous
        plt.plot(classifyResults[(classifyResults['syncronous']==False) & (classifyResults['m_freqs']<=maxFreq)]['a'], classifyResults[(classifyResults['syncronous'] ==False) & (classifyResults['m_freqs']<=maxFreq)]['b'], marker='o', linestyle='', label="asyncronous", color = "orange")
        
        
        
        plt.title("adaptance (a,b) space by synchrony - mean freq <20Hz" )
        
        plt.xlabel('a [nS]')
        plt.ylabel('b [pA]')
    
        plt.legend()
        # show plot
        plt.show()
  
        
        ########################################################################
        ### output the top 10 asyncronous (a,b) by lowest lowest corr and highest CV
        favouriteAdaptance = classifyResults[(classifyResults['syncronous'] == False)] #  only asyncronous
        favouriteAdaptance = favouriteAdaptance.sort_values(by = ['m_pairwise_corrs', 'm_cvs'], ascending = [True, False], na_position = 'last')
        if (favouriteAdaptance.empty):
            print(f"No asyncronous simulations were found with a mean network spiking frequency below {maxFreq} Hz")
            
        else:
            print("===== favourite simulation sorted my most asynchronous =====")    
            print(favouriteAdaptance.head(10))
    
    

    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    