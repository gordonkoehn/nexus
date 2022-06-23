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
gemin=40
gemax=40
gimin=80
gimax=80

gistep=1
gestep=1
 
# adaptance values
amin=0
amax=85
bmin=32
bmax=39

astep=1
bstep=1

# no of replica
replicaNo=3

conduSpace=$((((gimax-gimin+1)/(gistep))*((gemax-gemin+1)/(gestep))))
adapSpace=$((((amax-amin)/astep)*((bmax-bmin)/bstep)))
totalSim=$((conduSpace*adapSpace*replicaNo))
curSim=0

echo "A total conduSpace of $conduSpace "
echo "A total adapSpace of $adapSpace "
echo "A total of $totalSim simulation to do... let's go"

echo "A total of $totalSim simulation to do... let's go"


for (( ge=gemin; ge<=gemax; ge=ge+gistep )); do
    echo  "========================\n========================"
    echo  "ge = $ge \n"
    
    for (( gi=gimin; gi<=gimax; gi=gi+gestep )); do
        echo  "***********************"
        echo  "gi = $gi"
        
        for (( a=amin; a<=amax; a=a+astep )); do
            echo  "----------------------"
            echo  "a = $a"
             
            for (( b=bmin; b<=bmax; b=b+bstep )); do
                echo  "............................"
                echo  "b = $b (a=$a, ge=$gi, gi=$gi)" 
                
                for ((replica=1; replica<=replicaNo; replica=replica+1)); do
                    python ../../wp2_adex_model_script_replica.py 10 100 $a $b $ge $gi $probPee $probPei $probPii $probPie $replica
                
                    curSim=$((curSim+1))
                    percProgress=$(( 100 *(curSim)/(totalSim) )) 
                
                done
                echo "current simulation $curSim out of $totalSim"
                echo "$percProgress % of simulations done..."
            
            done
        
        done
        
    done
    
done

echo "========== DONE =============="
echo "=============================="