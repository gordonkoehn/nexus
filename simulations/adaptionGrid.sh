#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Tue May 24 18:10:33 2022
#
#@author: gordonkoehn
# creates simulation files for the adaption parameters a and b
#"""

# connection probabilities
 probPee=0.02 #(RS->RS) NB: originally all were p=0.02
 probPei=0.02 #(RS->FS)
 probPii=0.02 #(FS->FS)
 probPie=0.02 #(FS->RS)
 
# conductances fixed
ge=85
gi=75
 
# adaption space to test
amin=0
bmin=0
amax=85
bmax=85
astep=1
bstep=1

totalSim=$(( (amax-amin)/(astep)*(bmax-bmin)/(bstep) ))
curSim=0

for (( a=amin; a<=amax; a=a+astep )); do
    echo  "========================\n========================"
    echo  "a = $a \n"
    
    for (( b=bmin; b<=bmax; b=b+bstep )); do
        echo  "***********************"
        echo  "b = $b"
        
        python wp2_adex_model_script.py 10 100 $a $b $ge $gi $probPee $probPei $probPii $probPie
        
        curSim=$(( curSim+1 ))
        percProgress=$(( 100 *(curSim)/(totalSim) )) 
        echo "$percProgress % of simulations done..."
    done
    
done

echo "========== DONE =============="
echo "=============================="

