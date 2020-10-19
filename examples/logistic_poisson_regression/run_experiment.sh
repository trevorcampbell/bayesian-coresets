#!/bin/bash

for ID in {1..3}
do
    for alg in "US" "GIGA-OPT" "GIGA-REAL" "SVI" 
    do
        for dnm in "synth_lr" "phishing" "ds1"
	do
		python3 main.py --model lr --dataset $dnm --alg $alg --trial $ID run
	done
    done
done

#for ID in {1..3}
#do
#    for alg in "US" "FW" "GIGA" 
#    do
#        for dnm in "synth_poiss" "biketrips" "airportdelays"
#	do
#		python3 main.py --model poiss --dataset $dnm --alg $alg --trial $ID run 
#	done
#    done
#done
#


 
