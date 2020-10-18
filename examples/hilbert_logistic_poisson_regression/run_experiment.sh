#!/bin/bash

for ID in {1..10}
do
    for alg in "US" "FW" "GIGA" 
    do
        for dnm in "synth_lr" "phishing" "ds1"
	do
		python3 main.py lr $dnm --alg_nm $alg --trial $ID
	done
    done
done

for ID in {1..10}
do
    for alg in "US" "FW" "GIGA" 
    do
        for dnm in "synth_poiss" "biketrips" "airportdelays"
	do
		python3 main.py lr $dnm --alg_nm $alg --trial $ID
	done
    done
done



 
