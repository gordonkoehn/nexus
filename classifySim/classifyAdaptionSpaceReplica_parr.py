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
import time


import multiprocessing

from multiprocessing import Process, Manager

import itertools


def processAdaptancePoint(a,b,replica, params):
    """
    Access and process a given simulation file for 'a' and 'b' given, 
    and return the simulaiton statistics. 
    
    Parameters
    ----------
    a : float
        
    b : float
    
    (params : dict - access via manager)
    
    Returns
    -------
    stats : dict
         a as 'a';
         b as 'b';
         replicaNo as 'replicaNo'
         mean firing frequency as 'm_freq';
         mean pairwise correlation as 'm_pairwise_corr';
         mean coefficient of variation as 'm_'
    
    
    """   
    #make local params dictiroary
    paramsFull = params.copy()
    paramsFull['a'] = float(a)
    paramsFull['b'] = float(b)
    paramsFull['replica'] = int(replica)
    
    
    
    # results dictionary to return
    stats = dict()
    stats['a'] = a
    stats['b'] = b
    stats['replica'] = replica
    
    
    
    try:
        # get data
        save_name = getFilename(paramsFull)

        curr_dir = getPath(paramsFull)
        
        result = getResult_trun(curr_dir, save_name, paramsFull)
        
        ## analyze
        # get mean firing frequencies
        freqs = getFreq(result)

        m_freq = np.mean(freqs)
        
        # get coefficient of variation
        cvs = getCVs(result)
        m_cv = np.mean(cvs)
        
        # save the data
        stats['m_freq'] = m_freq
        stats['m_cvs'] = (m_cv)
        
        # get pairwise correlations
        binned_spiketrains = getBinnedSpiketrains(result)
        cc_matrix = computePariwiseCorr(binned_spiketrains)
        total_pairwise_corr =cc_matrix.sum() - cc_matrix.trace() # get sum of correlation, excluing self correlation (i.e. diagonal)
        m_pairwise_corr = total_pairwise_corr / (cc_matrix.shape[0]*cc_matrix.shape[1] - cc_matrix.shape[1]) #  note the lengeth of the diagonal of a NxN matrix is always N
            
        #save data
        stats['m_pairwise_corr'] = (m_pairwise_corr)

        
    except FileNotFoundError as fnf_error:
        stats['m_pairwise_corr'] = None
        stats['m_freq'] = None
        stats['m_cvs'] = None
        
        print(fnf_error)
        print("File was skipped and condition excluded from analysis.")
        
    return stats
    
    



###############################################################################
###############################################################################
if __name__ == '__main__':
    
    ######################################################
    ########### specify simulation parameters ############
    params = dict()
    params['sim_time'] = float(10)
    params['N'] = int(100)
    #conductance fixed -> to favorite
    params['ge'] = float(40)
    params['gi'] = float(80)
    
    #connection probabilities
    params['prob_Pee'] = float(0.02)
    params['prob_Pei'] = float(0.02)
    params['prob_Pii'] = float(0.02)
    params['prob_Pie'] = float(0.02)
    #params['ge']=float(0)
    #params['gi']=float(0)
    
    
    # specify conductance space
    a_min = 0
    a_max = 85 # should be 85
    b_min = 0
    b_max = 85 # should be 85
    
    step = 1
    
    # specify replica number
    replicaNo = 3
    
    #######################################################
    ########## specify output #############################
    doCorrelation = True
    
    
    #######################################################
    
    
    
    ############## loop through all files ################
    # make conditions
    
    
    ## define output lists
    # conductance values
    a_s = np.arange(a_min, a_max+1, step)
    b_s = np.arange(b_min, b_max+1, step)
    
    inputdata = [a_s, b_s]
    
    #adaptanceSpace = list(itertools.product(*inputdata))
    
    adaptanceSpace = []
    
    # loop through all conditions and analyze files
    for a in np.arange(a_min,a_max+step,step):
        for b in np.arange(b_min,b_max+step,step):
            for rep in np.arange(1,replicaNo+1,1):
                simParams = [a,b,rep, params]
                adaptanceSpace.append(simParams)
        
  

    with Manager() as manager:
        p = manager.dict()
        
        p['sim_time'] = params['sim_time']
        p['N'] = params['N']
        #conductance fixed -> to favorite
        p['ge'] = params['ge']
        p['gi'] = params['gi']
        
        #connection probabilities
        p['prob_Pee'] = params['prob_Pee']
        p['prob_Pei'] = params['prob_Pei']
        p['prob_Pii'] = params['prob_Pii']
        p['prob_Pie'] = params['prob_Pie']
    
        num_cores = 14
    

        job_arguments = adaptanceSpace
   
        pool = multiprocessing.Pool(processes=num_cores)
        
        results = pool.starmap(processAdaptancePoint, job_arguments)
        
        #results = pool.starmap(processAdaptancePoint, tqdm(job_arguments))
        
        
        pool.close()
        

        classifyResults = pd.DataFrame(results)
        
        # print how many files skipped / not analyzed
        conditionsNotAnalyzed =classifyResults['m_freq'].isnull().sum()
        
        ##save processed data to disk
        # get space
        space = dict()
        space["aMin"] = a_min
        space["bMin"] = b_min
        space["aMax"] = a_max
        space["bMax"] = b_max
            #unchanged as only adaptions space
        space["giMin"] = params["gi"]
        space["giMax"] = params["gi"]
        space["geMin"] = params["ge"]
        space["geMax"] = params["ge"]
        space["replica"] = replicaNo
        
        # get save name
        save_name = getNameClassifyData(params, space)
        #save
        classifyResults.to_pickle("classData/" + save_name)  
        
        print ("Saved full data as classData/" + save_name)
        
        print (f"A total of {conditionsNotAnalyzed} of {len(adaptanceSpace)} conditions where not analzyed/files not found. (no m_freq calulated).")
        
        
    ########################################
    #####  get variance and mean per replica
    

    #TODO: consider makeing this next block parralle.
    
    a_s = []
    b_s = []
    mm_freq = []
    mm_freq_stderr = []
    mm_cv = []
    mm_cv_stderr = []
    mm_corr = []
    mm_corr_stderr = []
    
    
    for a in np.arange(a_min,a_max+step,step):
        for b in np.arange(b_min,b_max+step,step):
            
                replicas = classifyResults[(classifyResults['a'] == a) & (classifyResults['b']==b)]
                
                mm_freq.append(replicas['m_freq'].mean())
                mm_freq_stderr.append(replicas['m_freq'].std()/replicas.shape[0])
                mm_cv.append(replicas['m_cvs'].mean())
                mm_cv_stderr.append(replicas['m_cvs'].std()/replicas.shape[0])
                mm_corr.append(replicas['m_pairwise_corr'].mean())
                mm_corr_stderr.append(replicas['m_pairwise_corr'].std()/replicas.shape[0])
                
                a_s.append(a)
                b_s.append(b)
                
    
    # dictionary of lists 
    dict = {'a_s': a_s,
            'b_s': b_s,
            'mm_freq': mm_freq,
            'mm_freq_stderr':mm_freq_stderr,
            'mm_cv':mm_cv,
            'mm_cv_stderr:':mm_cv_stderr,
            'mm_corr:':mm_corr,
            'mm_corr_stderr':mm_corr_stderr} 
    
    replicaSpace = pd.DataFrame(dict)
    
    # get save name
    save_name = "avg_" + getNameClassifyData(params, space)
    
    replicaSpace.to_pickle("classData/" + save_name)
    
    print ("Averaged data as classData/" + save_name)

  # #######################################################
  
  # ################# Make Figures #######################
  
  
  # ########## mean firing frequencies
  
  # ##### make excitatory
  
  #       # Creating figure
  #   fig = plt.figure(figsize = (10, 7))
  #   ax = plt.axes(projection ="3d")
       
  #   # Creating plot
  #   ax.scatter3D(classifyResults['a'],classifyResults['b'] , classifyResults['m_freq'], color = "green")
  #   plt.title("adaptance-frequency space")
      
  #   ax.set_xlabel('a [nS]')
  #   ax.set_ylabel('b [pA]')
  #   ax.set_zlabel('mean freq. [Hz]')
         
  #   # show plot
  #   plt.show()
      
  #   # ######### show CV and Corr
      
      
  #   # ######################################################################
  #   # ### do CV and corr plot
  #   # #Creating figure
  #   fig2 = plt.figure(figsize = (10, 7))
       
  #   #add asynchronous and synchronous classification
  #   classifyResults['asynchronous'] =  ((classifyResults['m_cvs'] > 1) & (classifyResults['m_pairwise_corr'] < 0.1))
      
        
  #   #plot synchronous
  #   asynPoints = classifyResults[classifyResults['asynchronous']==False]
      
  #   plt.plot(asynPoints['m_cvs'],asynPoints['m_pairwise_corr'],  marker='o', linestyle='', label="synchronous", color = "blue")
          
  #   #plot synchronous
  #   synPoints = classifyResults[classifyResults['asynchronous']==True]
      
  #   plt.plot(synPoints['m_cvs'],synPoints['m_pairwise_corr'],  marker='o', linestyle='', label="asynchronous", color = "orange")
         
  #   plt.title("coefficient of variation to pairwise correlation" )
          
  #   plt.xlabel('mean CV ')
  #   plt.ylabel('mean pairwise correlaion')
      
  #   plt.legend()
  #   # show plot
  #   plt.show()
          
          
          
  #   ########################################################################
  #   #### show (a,b) grid with colored syncronous any asyncornous colored
  #   # Creating figure
  #   fig3 = plt.figure(figsize = (10, 7))
          
  #   maxFreq = 200 # Hz
      
  #   # syncronous
  #   plt.plot(classifyResults[(classifyResults['asynchronous']==False) & (classifyResults['m_freq']<=maxFreq)]['a'], classifyResults[(classifyResults['asynchronous']==False) & (classifyResults['m_freq']<=maxFreq)]['b'], marker='o', linestyle='', label="synchronous", color = "blue")
          
  #   #asyncornous
  #   plt.plot(classifyResults[(classifyResults['asynchronous']==True) & (classifyResults['m_freq']<=maxFreq)]['a'], classifyResults[(classifyResults['asynchronous'] ==True) & (classifyResults['m_freq']<=maxFreq)]['b'], marker='o', linestyle='', label="asynchronous", color = "orange")
          
  #   plt.title("adaptance (a,b) space by synchrony - mean freq <200Hz" )
          
  #   plt.xlabel('a [nS]')
  #   plt.ylabel('b [pA]')
      
  #   plt.legend()
  #   # show plot
  #   plt.show()
          
      
  #   #######################################################################
  #   #### show (a,b) grid with colored syncronous any asyncornous colored - 20 Hz
  #   # Creating figure
  #   fig3 = plt.figure(figsize = (10, 7))
          
  #   maxFreq = 20 # Hz
      
  #   # syncronous
  #   plt.plot(classifyResults[(classifyResults['asynchronous']==False) & (classifyResults['m_freq']<=maxFreq)]['a'], classifyResults[(classifyResults['asynchronous']==False) & (classifyResults['m_freq']<=maxFreq)]['b'], marker='o', linestyle='', label="synchronous", color = "blue")
          
  #   #asyncornous
  #   plt.plot(classifyResults[(classifyResults['asynchronous']==True) & (classifyResults['m_freq']<=maxFreq)]['a'], classifyResults[(classifyResults['asynchronous'] ==True) & (classifyResults['m_freq']<=maxFreq)]['b'], marker='o', linestyle='', label="asynchronous", color = "orange")
          
  #   plt.title("adaptance (a,b) space by synchrony - mean freq <20Hz" )
          
  #   plt.xlabel('a [nS]')
  #   plt.ylabel('b [pA]')
      
  #   plt.legend()
  #   # show plot
  #   plt.show()
    
          
  #   ########################################################################
  #   ### output the top 10 asyncronous (a,b) by lowest lowest corr and highest CV
  #   favouriteAdaptance = classifyResults[(classifyResults['asynchronous'] == True)] #  only asyncronous
  #   favouriteAdaptance = favouriteAdaptance.sort_values(by = ['m_pairwise_corr', 'm_cvs'], ascending = [True, False], na_position = 'last')
  #   if (favouriteAdaptance.empty):
  #       print(f"No asyncronous simulations were found with a mean network spiking frequency below {maxFreq} Hz")
              
  #   else:
  #       print("===== favourite simulation sorted my most asynchronous =====")    
  #       print(favouriteAdaptance.head(10))
      
      
      
      
      
      
      
      
      
      
      
      
          
          
          
          
          
          
          
          
          
          
          
          
          
          