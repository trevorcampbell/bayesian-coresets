#!/bin/bash

#Algs: "SVI1" "SVIF" "GIGAT" "GIGAN" "RAND"

for ID in {1..10}
do
    for alg in "SVIF" "GIGAT" "GIGAN" "RAND" 
    do
	python3 main.py $alg $ID
    done
done
