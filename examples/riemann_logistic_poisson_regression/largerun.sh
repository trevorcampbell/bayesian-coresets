#!/bin/bash

#fldr = sys.argv[1] #should be either lr or poiss
#dnm = sys.argv[2] #if above is lr, should be synth / phishing / ds1; if above is poiss, should be synth, biketrips, or airportdelays
#alg = sys.argv[3] #should be hilbert / hilbert_corr / riemann / riemann_corr / uniform 
#ID = sys.argv[4] #just a number to denote trial #, any nonnegative integer


for ID in {1..1}
do
    for alg in "PRIOR"  #"DPBPSVI" #"PRIOR" #"BPSVI" #"SVI" "GIGAO" "GIGAR" "RAND" "PRIOR"
    do
        for dnm in  "ds1.100" #"fma" #"mnist2class_test" #"ds1.100" #"mnist2class_test" "fma" "ds1.100" # "mnist2class_test"
	do
		python3 largemain.py $dnm $alg $ID 
	done
    done
done
