#!/bin/bash

for ID in {1..10}
do
    #for alg in "BPSVI" "SVI" "GIGAO" "GIGAOE" "GIGAR" "GIGARE" "RAND"
    for alg in "BPSVI"
    do
	python3 main.py $alg $ID
    done
done
