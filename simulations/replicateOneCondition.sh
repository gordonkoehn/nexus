#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Fri May 27 09:16:54 2022

#@author: gordonkoehn

#Script written to run serval simulations of the same condition. 
#"""

# connection probabilities
 probPee=0.02 #(RS->RS) NB: originally all were p=0.02
 probPei=0.02 #(RS->FS)
 probPii=0.02 #(FS->FS)
 probPie=0.02 #(FS->RS)
 
# conductances fixed
gemin=20
gemax=100
gimin=60
gimax=85

gistep=5
gestep=5
 
# adaptance values
a=1
b=5

# no of replica
replicaNo=6


totalSim=$(((((gimax-gimin)/(gistep))*((gemax-gemin)/(gestep))*replicaNo)))
curSim=0
echo "A total of $totalSim simulation to do... let's go"


for (( ge=gemin; ge<=gemax; ge=ge+5 )); do
    echo  "========================\n========================"
    echo  "ge = $ge \n"
    
    for (( gi=gimin; gi<=gimax; gi=gi+5 )); do
        echo  "***********************"
        echo  "gi = $gi"
        
        for ((replica=4; replica<=replicaNo; replica=replica+1)); do
            python wp2_adex_model_script_replica.py 10 100 $a $b $ge $gi $probPee $probPei $probPii $probPie $replica
        
            curSim=$(( curSim+1 ))
            percProgress=$(( 100 *(curSim)/(totalSim) )) 
            echo "$percProgress % of simulations done..."
        done
    done
    
done

echo "========== DONE =============="
echo "=============================="