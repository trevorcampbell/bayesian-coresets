#!/bin/bash

for bound in {500000..1017357..50000}
do
	python3 process_housing_prices.py $bound $((bound+50000)) > logfile-$bound-$((bound+50000)).log 2>&1 &
done
