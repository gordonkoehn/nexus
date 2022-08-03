#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core executer - rerun if network dormant, not recurrent.

Allows to force asyncrhonous behaviour.

Created on Wed Aug  3 10:37:27 2022

@author: gordonkoehn
"""

import core

###############################################################################
###############################################################################
if __name__ == '__main__':
    
    
    validSimulationFound = False
    maxNoTries = 100
    
    
    ## counter
    trialNo = 0
    
    while not validSimulationFound:
        print("Attempt: " + str(trialNo+1))
        
        if (maxNoTries < trialNo):
            raise Exception("Maximal number of trials reached - no valid simulation was found.")
        try:
            core.simClasInfer(forceAsync=True, forcePhysical = True)
            validSimulationFound = True
        except Exception as e:
            print(e)
           
        trialNo += 1  
        
        