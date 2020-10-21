#!/bin/bash

for ID in {1..5}
do
    for alg in "GIGA" "FW" "OMP" "US"
    do
	python3 main.py --alg $alg --trial $ID --data_type normal run 
    done
done

for ID in {1..5}
do
    for alg in "GIGA" "FW" "OMP" "US"
    do
	python3 main.py --alg $alg --trial $ID --data_type axis --data_num 100 --coreset_size_max 100 --coreset_num_sizes 10 run
    done
done


