#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 19:26:31 2022

@author: gordonkoehn
"""



for (( ge=0; ge<=100; ge=ge+5 )); do
    echo  "========================\n========================"
    echo  "ge = $ge \n"
    
    for (( gi=0; gi<=100; gi=gi+5 )); do
        echo  "***********************"
        echo  "gi = $gi"
        
        python wp2_adex_model_script.py 10 100 1 5 $ge $gi
    done
    
done


#python wp2_adex_model_script.py 10 100 1 5 30 67