#!/bin/bash

#fldr = sys.argv[1] #should be either lr or poiss
#dnm = sys.argv[2] #if above is lr, should be synth / phishing / ds1; if above is poiss, should be synth, biketrips, or airportdelays
#alg = sys.argv[3] #should be hilbert / hilbert_corr / riemann / riemann_corr / uniform 
#ID = sys.argv[4] #just a number to denote trial #, any nonnegative integer


for ID in {1..10}
do
    for alg in "RND" "FW" "GIGA" 
    do
        for dnm in "synth_lr" "phishing" "ds1" "synth_poiss" "biketrips" "airportdelays"
	do
		python3 main.py $dnm $alg $ID
	done
    done
done
