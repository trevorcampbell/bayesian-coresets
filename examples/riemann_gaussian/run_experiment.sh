#!/bin/bash


for alg in "US" "GIGA-REAL" "GIGA-REAL-EXACT" "GIGA-OPT" "GIGA-OPT-EXACT" "SVI-EXACT" "SVI"
do 
    for ID in {1..3}
    do

        python3 main.py --alg $alg --trial $ID run
    done
done
