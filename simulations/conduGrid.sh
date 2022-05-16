#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Fri May 13 19:26:31 2022
#
#@author: gordonkoehn
#"""

 probPee=0.05 #(RS->RS) NB: originally all were p=0.02
 probPei=0.05 #(RS->FS)
 probPii=0.05 #(FS->FS)
 probPie=0.05 #(FS->RS)
 


for (( ge=0; ge<=100; ge=ge+5 )); do
    echo  "========================\n========================"
    echo  "ge = $ge \n"
    
    for (( gi=0; gi<=100; gi=gi+5 )); do
        echo  "***********************"
        echo  "gi = $gi"
        
        python wp2_adex_model_script.py 10 100 1 5 $ge $gi $probPee $probPei $probPii $probPie
    done
    
done



#python wp2_adex_model_script.py 10 100 1 5 30 67