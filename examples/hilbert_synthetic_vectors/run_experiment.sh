#!/bin/bash

for ID in {1..10}
do
    for alg in "GIGA" "FW" "OMP" "IS" "US"
    do
	python3 main.py run --alg_nm $alg --trial $ID --data_type normal
    done
done

for ID in {1..10}
do
    for alg in "GIGA" "FW" "OMP" "IS" "US"
    do
	python3 main.py run --alg_nm $alg --trial $ID --data_type axis --data_num 100 --coreset_size_max 100 --coreset_num_sizes 10
    done
done


