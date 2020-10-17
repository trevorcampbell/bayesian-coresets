#!/bin/bash

for ID in {1..10}
do
    for alg in "GIGA" "FW" "OMP" "IS" "US"
    do
	python3 main.py $alg --trial $ID 
    done
done
