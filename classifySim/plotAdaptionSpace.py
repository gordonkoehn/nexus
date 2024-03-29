#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 12:16:30 2022

@author: gordonkoehn

Script to load adaption grid search files and plot them to find asynchronous 
(a,b) combinations. 

Note that the replica data loaded is already averaged for the three replica.

At the start of the main the user may select to analyze single or reploca runs.
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

    
    ######### TOGGLE MODE:
    # plot single
    doSingle = False
    # plot replica
    doReplica = True
   
    ########################################################
     
    ############### Import data ############################
    
    if doSingle :
        classifyResults = pd.read_pickle("classData/N_100_t_10.0_probs_0.02_0.02_0.02_0.02_a_20_50_b_20_50_gi_80.0_80.0_ge_40.0_40.0_rep_3.pkl")
        
    if doReplica:   
        replicaSpace = pd.read_pickle("classData/avg_N_100_t_10.0_probs_0.02_0.02_0.02_0.02_a_0_85_b_0_85_gi_80.0_80.0_ge_40.0_40.0_rep_3.pkl")
        
    #######################################################
    
    ################# Make Figures #######################

    #close all old figures
    #plt.figure().close('all') 

    ################### Single ###########################
    if doSingle : 
    
        ########## mean firing frequencies (single simulations)
        
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
        plt.plot(classifyResults[(classifyResults['asynchronous']==False) & (classifyResults['m_freq']<=maxFreq & classifyResults['m_freq']>1)]['a'], classifyResults[(classifyResults['asynchronous']==False) & (classifyResults['m_freq']<=maxFreq & classifyResults['m_freq']>1)]['b'], marker='o', linestyle='', label="synchronous", color = "blue")
              
        #asyncornous
        plt.plot(classifyResults[(classifyResults['asynchronous']==True) & (classifyResults['m_freq']<=maxFreq & classifyResults['m_freq']>1)]['a'], classifyResults[(classifyResults['asynchronous'] ==True) & (classifyResults['m_freq']<=maxFreq & classifyResults['m_freq']>1)]['b'], marker='o', linestyle='', label="asynchronous", color = "orange")
              
        plt.title("adaptance (a,b) space by synchrony - mean freq <20Hz and above 1 Hz" )
              
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
    
    if doReplica:
        
        
      #add asynchronous and synchronous classification
      replicaSpace['asynchronous'] =  ((replicaSpace['mm_cv'] > 1) & (replicaSpace['mm_corr'] < 0.1))
         
         
            
      ###############################################
      ######## plot adaption - mean mean freq #####
      # Creating figure
      fig5 = plt.figure(figsize = (10, 7))
      ax = plt.axes(projection ="3d")
      plt.title("adaptance-mean-mean-frequency space -  mean replica")
      
     
      # Add data
      ## all points
      ax.scatter3D(replicaSpace['a_s'],replicaSpace['b_s'] ,
                   replicaSpace['mm_freq'], color = "dimgray",alpha=0.3)
     
      ## mark (overplot) physical
      # 1 Hz < mean freq < 30 Hz 
      pointsPhysical = replicaSpace[(replicaSpace['mm_freq'] > 1) & (replicaSpace['mm_freq'] < 30 )]
      # ax.scatter3D(pointsPhysical['a_s'],pointsPhysical['b_s'] ,
      #              pointsPhysical['mm_freq'], color = "green", label = "physical" )
      
      ## mark (overplot) asynchronous
      pointsAsynchronous =  replicaSpace[replicaSpace['asynchronous'] == True]
      
      ax.scatter3D(pointsAsynchronous['a_s'],pointsAsynchronous['b_s'] ,
                   pointsAsynchronous['mm_freq'], color = "blue", label = "asynchronous", alpha=0.4 )
      
      
      pointsPhysicalAsync = pointsPhysical[pointsPhysical['asynchronous'] == True]
      
      ax.scatter3D(pointsPhysicalAsync['a_s'],pointsPhysicalAsync['b_s'] ,
                   pointsPhysicalAsync['mm_freq'], color = "green", label = "asynchronous & physical (<30 Hz)", alpha=1 , s=45)
      
      

      ax.legend()
      ax.set_xlabel('a [nS]')
      ax.set_ylabel('b [pA]')
      ax.set_zlabel('mean mean freq. [Hz]')

      plt.show()
      
      
      
      ## print out favorites
      
      favorites  = pointsPhysicalAsync[pointsPhysicalAsync['all_dormant'] ==False]
      
      print("========== Physical (<30Hz) & Asynchronous & Active Points ======")
      
      print(favorites[['a_s','b_s', 'mm_freq', 'mm_freq_stderr', 'mm_cv', 'mm_cv_stderr', 'mm_corr','mm_corr_stderr']])
      
      
      #######################################################################
      #### show (a,b) grid with colored syncronous any asyncornous colored - 20 Hz
      # Creating figure
      fig6 = plt.figure(figsize = (7, 7))
            
      maxFreq = 200 # Hz
        
      # syncronous
      plt.plot(replicaSpace[(replicaSpace['asynchronous']==False) & (replicaSpace['mm_freq']<=maxFreq)]['a_s'],
               replicaSpace[(replicaSpace['asynchronous']==False) & (replicaSpace['mm_freq']<=maxFreq)]['b_s'],
               marker='o', linestyle='', label="synchronous", color = "blue")
            
      #asyncornous
      plt.plot(replicaSpace[(replicaSpace['asynchronous']==True) & (replicaSpace['mm_freq']<=maxFreq)]['a_s'],
               replicaSpace[(replicaSpace['asynchronous'] ==True) & (replicaSpace['mm_freq']<=maxFreq)]['b_s'],
               marker='o', linestyle='', label="asynchronous", color = "orange")
            
      plt.title("adaptance (a,b) space by synchrony - mean freq <20Hz - replica" )
            
      plt.xlabel('a [nS]')
      plt.ylabel('b [pA]')
        
      plt.legend()
      # show plot
      plt.show()
      
      
      # ######################################################################
      # ### do CV and corr plot
      # #Creating figure
      fig7 = plt.figure(figsize = (10, 7))
         
     
      #plot synchronous
      asynPoints = replicaSpace[(replicaSpace['asynchronous']==False)]
        
      plt.plot(asynPoints['mm_cv'],asynPoints['mm_corr'],  marker='o', linestyle='',
               label="synchronous", color = "blue")
            
      #plot synchronous
      synPoints = replicaSpace[replicaSpace['asynchronous']==True]
        
      plt.plot(synPoints['mm_cv'],synPoints['mm_corr'],  marker='o', linestyle='',
               label="asynchronous", color = "orange")
           
      plt.title("coefficient of variation to pairwise correlation - mean replica" )
            
      plt.xlabel('mean CV ')
      plt.ylabel('mean pairwise correlaion')
        
      plt.legend()
      # show plot
      plt.show()
     
      
      ############################################
      # add CV and CORR a.b heatmaps to see gradients
      
      fig8, axs = plt.subplots(1,2, figsize = (10, 5))
      fig8.suptitle('Heatmaps of CV and corr for (a,b) adaptance space')

      #### subplot 0
      
      cm = plt.cm.get_cmap('RdYlBu_r')
      x = replicaSpace['a_s']
      y = replicaSpace['b_s']
      
      z = replicaSpace['mm_cv']
      sc = axs[0].scatter(x, y, c=z, vmin=replicaSpace['mm_cv'].min(), vmax=replicaSpace['mm_cv'].max(), s=35, cmap=cm)
      
      fig8.colorbar(sc, label= "mean CV", ax= axs[0])
      
      axs[0].set(xlabel='a [nS]', ylabel='b [pA]')
      
      # subplot 1
      
      cm = plt.cm.get_cmap('RdYlBu')
      x = replicaSpace['a_s']
      y = replicaSpace['b_s']
      
      z = replicaSpace['mm_corr']
      sc = axs[1].scatter(x, y, c=z, vmin=replicaSpace['mm_corr'].min(), vmax=replicaSpace['mm_corr'].max(), s=35, cmap=cm)
      
      #axs[1].colorbar(sc, label= "CV")
      
      fig8.colorbar(sc, label= "mean pairwise correlation", ax= axs[1])
      
      axs[1].set(xlabel='a [nS]', ylabel='b [pA]')
   
    
      #######################################################################
    ### show (a,b) grid if network is dormant or not
     # Creating figure
    fig6 = plt.figure(figsize = (7, 7))
      
     # dormant
    plt.plot(replicaSpace[(replicaSpace['all_dormant']==True)]['a_s'],
              replicaSpace[(replicaSpace['all_dormant']==True)]['b_s'],
              marker='o', linestyle='', label="dormant", color = "grey")
          
     # active
    plt.plot(replicaSpace[(replicaSpace['all_dormant']==False)]['a_s'],
              replicaSpace[(replicaSpace['all_dormant']==False)]['b_s'],
              marker='o', linestyle='', label="active", color = "orange")
    
    # file not found
    plt.plot(replicaSpace[(replicaSpace['fileNotFound']==True)]['a_s'],
              replicaSpace[(replicaSpace['fileNotFound']==True)]['b_s'],
              marker='o', linestyle='', label="file not found", color = "red")
    
    
          
    plt.title("Is the network spiking at all after simulation (a,b) space - replica" )
          
    plt.xlabel('a [nS]')
    plt.ylabel('b [pA]')
      
    plt.legend()
     # show plot
    plt.show()
    
    