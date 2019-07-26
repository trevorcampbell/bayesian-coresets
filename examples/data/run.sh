#!/bin/bash

#for bound in {0..1017357..50000}

for bound in {0..100..20}
do
	python3 process_housing_prices.py $bound $((bound+20)) > logfile-$bound-$((bound+20)).log 2>&1 &
	#python3 process_housing_prices.py $bound $((bound+50000))
done
