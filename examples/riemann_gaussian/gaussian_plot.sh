#!/bin/bash

n_trials=10
plot_every=5
d=200
N=1000
fldr_res='results'
fldr_res_prfx='results'
no_pcst='False'
python3 plot_kl.py $n_trials $plot_every $d $N $fldr_res $fldr_res_prfx $no_pcst
