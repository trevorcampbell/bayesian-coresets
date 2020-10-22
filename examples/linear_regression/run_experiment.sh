#!/bin/bash


for alg in "US" "GIGA-REAL" "GIGA-REAL-EXACT" "GIGA-OPT" "GIGA-OPT-EXACT" "SVI"
do 
    for ID in {1..3}
    do
        python3 main.py --alg $alg --trial $ID run
    done
done

for alg in "SVI-EXACT" 
do 
    for ID in {1..3}
    do
        python3 main.py --alg $alg --trial $ID --proj_dim 30 run
    done
done
