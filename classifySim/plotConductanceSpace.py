#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 18:27:32 2022

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
    
    
    # specify conductance space
    ge_min = 0
    ge_max = 100 # should be 100
    gi_min = 0
    gi_max = 100 # should be 100
    
    step = 5
    

    
    
    condFreqSpace = pd.read_pickle("classData/N_100_t_10.0_probs_0.02_0.02_0.02_0.02_a_1.0_1.0_b_5.0_5.0_gi_0_100_ge_0_100_rep_3.pkl")  
    
    
    #######################################################
    
    
    
    ################# Make Figures #######################
    
    
    # ########## mean firing frequencies
    
    # ##### make excitatory
    
    # # Creating figure
    # fig = plt.figure(figsize = (10, 7))
    # ax = plt.axes(projection ="3d")
 
    # # Creating plot
    # ax.scatter3D(condFreqSpace['ge'],condFreqSpace['gi'] , condFreqSpace['m_freqs'], color = "green")
    # plt.title("conductance-frequency space -  plot the 3 replica on a scatter together")
    
    # ax.set_xlabel('ge [nS]')
    # ax.set_ylabel('gi [nS]')
    # ax.set_zlabel('mean freq. [Hz]')
    
    
    # # results dictionary for points that are physical
    # pointsPhysical = pd.DataFrame.from_dict(condFreqSpace)
    # # 1 Hz < mean freq < 20 Hz 
    # pointsPhysical = pointsPhysical[(pointsPhysical['m_freqs'] > 1) & (pointsPhysical['m_freqs'] < 20 )]
    # pointsPhysical = pointsPhysical.sort_values(by = ['ge', 'gi'], ascending = [False, False], na_position = 'last')
    
    
    
    # #zipZip = zip(condFreqSpace['ge'],condFreqSpace['gi'] , condFreqSpace['m_freqs'],zip(condFreqSpace['ge'],condFreqSpace['gi'],np.around(condFreqSpace['m_freqs'], decimals=1)))
    
    # for row in pointsPhysical.head(5).itertuples(index=False):
    #     row = row._asdict()
    #     ax.text(row['ge'], row['gi'], row['m_freqs'], "("+ str(row['ge']) +", " +  str(row['gi']) +", " + str( np.around(row['m_freqs'],1)) + " )", size=8)
    
 
    # # show plot
    # plt.show()
    # #plt.savefig("condu-freq_condu.svg", format = 'svg', dpi=300)
     
   
    ########################################
    #####  get variance per gi, ge pair
    
    #condFreqSpace = pd.DataFrame.from_dict(condFreqSpace)
    
    
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
    ax.scatter3D(replicaFreqSpace['ge'],replicaFreqSpace['gi'] , replicaFreqSpace['mm_freq'], color = "dimgray")
    #plt.title("conductance-mean-mean frequency space -  mean replica")
    
    ax.set_xlabel('ge [nS]')
    ax.set_ylabel('gi [nS]')
    ax.set_zlabel('replica mean freq. [Hz]')
    

    
    # results dictionary for points that are physical
    pointsPhysical = pd.DataFrame.from_dict(replicaFreqSpace)
    # 1 Hz < mean freq < 20 Hz 
    pointsPhysical = pointsPhysical[(pointsPhysical['mm_freq'] > 1) & (pointsPhysical['mm_freq'] < 20 )]
    pointsPhysical = pointsPhysical.sort_values(by = ['ge', 'gi'], ascending = [False, False], na_position = 'last')
    
    ax.scatter3D(pointsPhysical['ge'],pointsPhysical['gi'] , pointsPhysical['mm_freq'], color = "green", label = "physical")
    
    
    
    #zipZip = zip(condFreqSpace['ge'],condFreqSpace['gi'] , condFreqSpace['m_freqs'],zip(condFreqSpace['ge'],condFreqSpace['gi'],np.around(condFreqSpace['m_freqs'], decimals=1)))
    
    # for row in pointsPhysical.head(3).itertuples(index=False):
    #     row = row._asdict()
    #     #ax.text(row['ge'], row['gi'], row['mm_freq'], "("+ str(row['ge']) +", " +  str(row['gi']) +", " + str( np.around(row['mm_freq'],1)) + " )", size=7,horizontalalignment='right',verticalalignment='center')
    #     ax.scatter3D(row['ge'], row['gi'], row['mm_freq'],s=40, color = "red", alpha =1)
    #     #plot errorbars
    #     xval, yval, zval, zerr = row['ge'], row['gi'], row['mm_freq'], row['mm_freq_stderr']
    #     ax.plot([xval, xval], [yval, yval], [zval+zerr, zval-zerr], marker="_", color='k')
    
    
    #plt.legend()
    # show plot
    plt.show()
    #plt.savefig("Cond-m-m-Freq_condu.svg", format = 'svg', dpi=300)
    
    
    
    # #################
    # ##### plot the 
    # # Creating figure
    # fig = plt.figure(figsize = (10, 7))
    # ax = plt.axes(projection ="3d")
 
    # bottom = np.zeros_like(replicaFreqSpace['mm_freq'])
    # width = depth = step
        

    # # Creating plot
    # ax.bar3d(replicaFreqSpace['ge'],replicaFreqSpace['gi'] ,bottom, width, depth,replicaFreqSpace['mm_freq_stderr'],shade=True)

    # plt.title("conductance-mean-mean frequency standard error (of 3 replica)")
    
    # ax.set_xlabel('ge [nS]')
    # ax.set_ylabel('gi [nS]')
    # ax.set_zlabel('std (mean freq.) [Hz]')
    

    # # show plot
    # plt.show()
    # #plt.savefig("Cond-m-m-Freq-stderr_condu.svg", format = 'svg', dpi=300)
    
    
    ########################################################################
    ### output the top 10  (gi,ge) by highest conductances
    
    if (pointsPhysical.empty):
      print(f"No physical simulations were found with a mean network spiking frequency below {maxFreq} Hz")
    
    else:
        print("===== favourite simulation sorted by highest conductances =====")    
        print(pointsPhysical.head(10))

        
    