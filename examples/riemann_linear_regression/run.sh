#!/bin/bash

for ID in {3..10}
do
    #for alg in "SVI" "GIGAO" "GIGAOE" "GIGAR" "GIGARE" "RAND" 
    for alg in "GIGAO" "GIGAOE" "GIGAR" "GIGARE" "RAND" 
    do
	python3 main.py $alg $ID
    done
done
