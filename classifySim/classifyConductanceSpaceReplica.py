#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 21:32:13 2022

@author: gordonkoehn
"""
import numpy as np
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt

# module with all the functions and logic
from classifySimulations import *



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
    ge_min = 0
    ge_max = 100 # should be 100
    gi_min = 0
    gi_max = 100 # should be 100
    
    step = 5
    
    # specify no of replica
    replicaNo = 3
    
    #######################################################
    ########## specify output #############################
    doCorrelation = False
    
    
    #######################################################
    
    
    
    ############## loop through all files ################
    ## define output lists
    # conductance values
    ges = [] 
    gis = []
    # replica no
    replicaNos = []
    
    #mean firing frequencies
    m_freqs = []
    # coefficeit of varriation
    m_cvs = []
    # mean pairwise correlation
    m_pairwise_corrs = []   
    
    # conditons count
    conduSpace = 0
    
    # loop through all conditions and analyze files
    for gi in np.arange(gi_min,gi_max+step,step):
        for ge in np.arange(ge_min,ge_max+step,step):
            for replNo in range(1,replicaNo+1):
                # define parameters
                params['ge']=float(ge)
                params['gi']=float(gi)
                params['replica']=int(replNo)
                
                conduSpace =+ 1
                
                # get file
                
                try:
                    save_name = getFilename(params)
                    
                    
                    curr_dir = getPath(params)
                    # get data
                    result = getResult_trun(curr_dir, save_name, params)
                    
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
                    ges.append(ge)
                    gis.append(gi)
                    replicaNos.append(replNo)
                    m_freqs.append(m_freq)
                
                except FileNotFoundError as fnf_error:
                    print(fnf_error)
                    print("File was skipped and condition excluded from analysis.")
       
            
    # create output dictonary
    condFreqSpace =  dict()
    condFreqSpace['ge'] = ges
    condFreqSpace['gi'] = gis
    condFreqSpace['replicaNo'] = replicaNos
    # save frequency values but replace NAs with 0
    condFreqSpace['m_freqs'] = [0 if x != x else x for x in m_freqs] 


    ######## save the analyzed data
    
    classifyResults = pd.DataFrame(condFreqSpace)
    
    # print how many files skipped / not analyzed
    conditionsNotAnalyzed =classifyResults['m_freqs'].isnull().sum()
    
    ##save processed data to disk
    # get space
    space = dict()
    space["aMin"] = params["a"]
    space["bMin"] = params["b"]
    space["aMax"] = params["a"]
    space["bMax"] = params["b"]
        #unchanged as only adaptions space
    space["giMin"] = gi_min
    space["giMax"] = gi_max
    space["geMin"] = ge_min
    space["geMax"] = ge_max
    space["replica"] = replicaNo
    
    # get save name
    save_name = getNameClassifyData(params, space) + "_condu"
    #save
    classifyResults.to_pickle("classData/" + save_name)  
    
    print ("Saved full data as classData/" + save_name)
    
    print (f"A total of {conditionsNotAnalyzed} of {conduSpace} conditions where not analzyed/files not found. (no m_freq calulated).")
    
    
    
    
    #######################################################
    
    
    
    ################# Make Figures #######################
    
    
    ########## mean firing frequencies
    
    ##### make excitatory
    
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
 
    # Creating plot
    ax.scatter3D(condFreqSpace['ge'],condFreqSpace['gi'] , condFreqSpace['m_freqs'], color = "green")
    plt.title("conductance-frequency space -  plot the 3 replica on a scatter together")
    
    ax.set_xlabel('ge [nS]')
    ax.set_ylabel('gi [nS]')
    ax.set_zlabel('mean freq. [Hz]')
    
    
    # results dictionary for points that are physical
    pointsPhysical = pd.DataFrame.from_dict(condFreqSpace)
    # 1 Hz < mean freq < 20 Hz 
    pointsPhysical = pointsPhysical[(pointsPhysical['m_freqs'] > 1) & (pointsPhysical['m_freqs'] < 20 )]
    pointsPhysical = pointsPhysical.sort_values(by = ['ge', 'gi'], ascending = [False, False], na_position = 'last')
    
    
    
    #zipZip = zip(condFreqSpace['ge'],condFreqSpace['gi'] , condFreqSpace['m_freqs'],zip(condFreqSpace['ge'],condFreqSpace['gi'],np.around(condFreqSpace['m_freqs'], decimals=1)))
    
    for row in pointsPhysical.head(5).itertuples(index=False):
        row = row._asdict()
        ax.text(row['ge'], row['gi'], row['m_freqs'], "("+ str(row['ge']) +", " +  str(row['gi']) +", " + str( np.around(row['m_freqs'],1)) + " )", size=8)
    
 
    # show plot
    #plt.show()
    plt.savefig("condu-freq_condu.svg", format = 'svg', dpi=300)
    
    
    ######### show CV 
    
    if doCorrelation : 
    
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
    
        # plot syncronous
        
        m_cvs_s = np.ma.masked_array(m_cvs, mask = not(groupSyncronous)).compressed()
        m_pairwise_corrs_s = np.ma.masked_array(m_pairwise_corrs, mask = not(groupSyncronous)).compressed()
    
        plt.plot(m_cvs_s, m_pairwise_corrs_s, marker='o', linestyle='', label="", color = "blue")
        
        
        #plot asyncronous
        m_cvs_a = np.ma.masked_array(m_cvs, mask = groupSyncronous).compressed()
        m_pairwise_corrs_a = np.ma.masked_array(m_pairwise_corrs, mask = groupSyncronous).compressed()
        
        plt.plot(m_cvs_a, m_pairwise_corrs_a, marker='o', linestyle='', label="", color = "orange")
        
        
        
        plt.title("coefficient of variation to pairwise correlation" )
        
        plt.xlabel('mean CV ')
        plt.ylabel('mean pairwise correlaion')
    
        #plt.legend()
        # show plot
        #plt.show()
        plt.savefig("CV-Corr_condu.svg", format = 'svg', dpi=300)
    
   
    ########################################
    #####  get variance per gi, ge pair
    
    condFreqSpace = pd.DataFrame.from_dict(condFreqSpace)
    
    
    ge_s = []
    gi_s = []
    mm_freqs = []
    mm_freq_stderr = []
    
    
    for gi in np.arange(gi_min,gi_max+step,step):
        for ge in np.arange(ge_min,ge_max+step,step):
            try:
                replicas = condFreqSpace[(condFreqSpace['ge'] == ge) & (condFreqSpace['gi']==gi)]
                mm_freq_stderr.append(replicas['m_freqs'].std()/replicas.shape[0])
                mm_freqs.append(replicas['m_freqs'].mean())
                ge_s.append(ge)
                gi_s.append(gi)
            except ZeroDivisionError as zero_error:
                print(zero_error)
                print(f"Condition ({gi}, {ge}) was skipped from analysis.")
   
                
    
    # dictionary of lists 
    dict = {'ge': ge_s, 'gi': gi_s, 'mm_freq': mm_freqs, 'mm_freq_stderr':mm_freq_stderr} 
    
    replicaFreqSpace = pd.DataFrame(dict)
    
    
    ###################
    ### plot mean mean freq
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
 
    # Creating plot
    ax.scatter3D(replicaFreqSpace['ge'],replicaFreqSpace['gi'] , replicaFreqSpace['mm_freq'], color = "green")
    plt.title("conductance-mean-mean frequency space -  mean replica")
    
    ax.set_xlabel('ge [nS]')
    ax.set_ylabel('gi [nS]')
    ax.set_zlabel('mean mean freq. [Hz]')
    

    
    # results dictionary for points that are physical
    pointsPhysical = pd.DataFrame.from_dict(replicaFreqSpace)
    # 1 Hz < mean freq < 20 Hz 
    pointsPhysical = pointsPhysical[(pointsPhysical['mm_freq'] > 1) & (pointsPhysical['mm_freq'] < 20 )]
    pointsPhysical = pointsPhysical.sort_values(by = ['ge', 'gi'], ascending = [False, False], na_position = 'last')
    
    
    
    #zipZip = zip(condFreqSpace['ge'],condFreqSpace['gi'] , condFreqSpace['m_freqs'],zip(condFreqSpace['ge'],condFreqSpace['gi'],np.around(condFreqSpace['m_freqs'], decimals=1)))
    
    for row in pointsPhysical.head(5).itertuples(index=False):
        row = row._asdict()
        ax.text(row['ge'], row['gi'], row['mm_freq'], "("+ str(row['ge']) +", " +  str(row['gi']) +", " + str( np.around(row['mm_freq'],1)) + " )", size=8)
    
 
    # show plot
    #plt.show()
    plt.savefig("Cond-m-m-Freq_condu.svg", format = 'svg', dpi=300)
    
    
    
    #################
    ##### plot the 
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
 
    bottom = np.zeros_like(replicaFreqSpace['mm_freq'])
    width = depth = step
        

    # Creating plot
    ax.bar3d(replicaFreqSpace['ge'],replicaFreqSpace['gi'] ,bottom, width, depth,replicaFreqSpace['mm_freq_stderr'],shade=True)

    plt.title("conductance-mean-mean frequency standard error (of 3 replica)")
    
    ax.set_xlabel('ge [nS]')
    ax.set_ylabel('gi [nS]')
    ax.set_zlabel('std (mean freq.) [Hz]')
    

    # show plot
    #plt.show()
    plt.savefig("Cond-m-m-Freq-stderr_condu.svg", format = 'svg', dpi=300)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    