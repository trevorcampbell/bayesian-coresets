#!/bin/bash

for ID in {1..10}
do
    #for alg in "SVIF" "GIGAT" "GIGAN" "RAND" 
    for alg in "SVIF"
    do
	python3 main.py $alg $ID
    done
done
