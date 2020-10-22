#!/bin/bash

for dnm in "synth_lr" "phishing" "ds1"
do
    for alg in "US" "GIGA-OPT" "GIGA-REAL" "SVI" 
    do
        for ID in {1..3}
        do
		python3 main.py --model lr --dataset $dnm --alg $alg --trial $ID run
	done
    done
done


