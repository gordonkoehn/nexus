#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 12:16:30 2022

@author: gordonkoehn

Loads specified data.
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
import time


###############################################################################
###############################################################################
if __name__ == '__main__':
    
    replicaSpace = pd.read_pickle("classData/avg_N_100_t_10.0_probs_0.02_0.02_0.02_0.02_a_0_85_b_0_85_gi_85.0_85.0_ge_20.0_20.0_rep_3.pkl")
    
    classifyResults = pd.read_pickle("classData/N_100_t_10.0_probs_0.02_0.02_0.02_0.02_a_0_85_b_0_85_gi_85.0_85.0_ge_20.0_20.0_rep_3.pkl")
    
    #######################################################
    
    ################# Make Figures #######################
    
    
    ########## mean firing frequencies
    
    ##### make excitatory
    
        # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
       
    # Creating plot
    ax.scatter3D(classifyResults['a'],classifyResults['b'] , classifyResults['m_freq'], color = "green")
    plt.title("adaptance-frequency space")
      
    ax.set_xlabel('a [nS]')
    ax.set_ylabel('b [pA]')
    ax.set_zlabel('mean freq. [Hz]')
         
    # show plot
    plt.show()
      
    # ######### show CV and Corr
      
      
    # ######################################################################
    # ### do CV and corr plot
    # #Creating figure
    fig2 = plt.figure(figsize = (10, 7))
       
    #add asynchronous and synchronous classification
    classifyResults['asynchronous'] =  ((classifyResults['m_cvs'] > 1) & (classifyResults['m_pairwise_corr'] < 0.1))
      
        
    #plot synchronous
    asynPoints = classifyResults[classifyResults['asynchronous']==False]
      
    plt.plot(asynPoints['m_cvs'],asynPoints['m_pairwise_corr'],  marker='o', linestyle='', label="synchronous", color = "blue")
          
    #plot synchronous
    synPoints = classifyResults[classifyResults['asynchronous']==True]
      
    plt.plot(synPoints['m_cvs'],synPoints['m_pairwise_corr'],  marker='o', linestyle='', label="asynchronous", color = "orange")
         
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
    plt.plot(classifyResults[(classifyResults['asynchronous']==False) & (classifyResults['m_freq']<=maxFreq)]['a'], classifyResults[(classifyResults['asynchronous']==False) & (classifyResults['m_freq']<=maxFreq)]['b'], marker='o', linestyle='', label="synchronous", color = "blue")
          
    #asyncornous
    plt.plot(classifyResults[(classifyResults['asynchronous']==True) & (classifyResults['m_freq']<=maxFreq)]['a'], classifyResults[(classifyResults['asynchronous'] ==True) & (classifyResults['m_freq']<=maxFreq)]['b'], marker='o', linestyle='', label="asynchronous", color = "orange")
          
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
    plt.plot(classifyResults[(classifyResults['asynchronous']==False) & (classifyResults['m_freq']<=maxFreq)]['a'], classifyResults[(classifyResults['asynchronous']==False) & (classifyResults['m_freq']<=maxFreq)]['b'], marker='o', linestyle='', label="synchronous", color = "blue")
          
    #asyncornous
    plt.plot(classifyResults[(classifyResults['asynchronous']==True) & (classifyResults['m_freq']<=maxFreq)]['a'], classifyResults[(classifyResults['asynchronous'] ==True) & (classifyResults['m_freq']<=maxFreq)]['b'], marker='o', linestyle='', label="asynchronous", color = "orange")
          
    plt.title("adaptance (a,b) space by synchrony - mean freq <20Hz" )
          
    plt.xlabel('a [nS]')
    plt.ylabel('b [pA]')
      
    plt.legend()
    # show plot
    plt.show()
    
          
    ########################################################################
    ### output the top 10 asyncronous (a,b) by lowest lowest corr and highest CV
    favouriteAdaptance = classifyResults[(classifyResults['asynchronous'] == True)] #  only asyncronous
    favouriteAdaptance = favouriteAdaptance.sort_values(by = ['m_pairwise_corr', 'm_cvs'], ascending = [True, False], na_position = 'last')
    if (favouriteAdaptance.empty):
        print(f"No asyncronous simulations were found with a mean network spiking frequency below {maxFreq} Hz")
              
    else:
        print("===== favourite simulation sorted my most asynchronous =====")    
        print(favouriteAdaptance.head(10))
      
      
      