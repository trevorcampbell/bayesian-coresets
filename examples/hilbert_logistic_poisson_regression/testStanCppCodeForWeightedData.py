from __future__ import print_function
import numpy as np
import time
import sys, os
import argparse

#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import mcmc

parser = argparse.ArgumentParser(description="Runs MCMC sampling using stan on a randomly subsampled and randomly (integer) weighted portion of the specified dataset. Does this with both the default stan model and with the weighted-coreset-version of the stan model - both should return the same results if the weighted version is written correctly.")
parser.add_argument('model', type=str, choices=["lr","poiss"], help="The regression model to use. lr refers to logistic regression, and poiss refers to poisson regression.")
parser.add_argument('dnm', type=str, help="The name of the dataset on which to run regression")
parser.add_argument('ID', type=int, help="The trial number - used to initialize random number generation (for replicability)")
parser.add_argument('numIdcs', type=int, help="number of indices to use in the weighted coreset")
#TODO: make mcmc params command-line args

arguments = parser.parse_args()
dnm = arguments.dnm
ID = arguments.ID
model = arguments.model
numIdcs = arguments.numIdcs

#lrdnms = ['synth_lr', 'phishing', 'ds1', 'synth_lr_large', 'phishing_large', 'ds1_large']
#prdnms = ['synth_poiss', 'biketrips', 'airportdelays', 'synth_poiss_large', 'biketrips_large', 'airportdelays_large']
if model=="lr":
  from model_lr import *
elif model=="poiss":
  from model_poiss import *

np.random.seed(ID)

#should be command line arguments 
mcmc_steps = 10000#10000 #total number of MH steps

print('Loading dataset '+dnm)
X,Y,Z, Zt, D = load_data('../data/'+dnm+'.npz')

idcs = np.random.choice(X.shape[0],numIdcs, replace=True) #random.choice is a bit slower without replacement, but may still be worth trying
weights = np.random.randint(1,10,numIdcs)
curX = X[idcs, :]
curY = Y[idcs]
#run the weighted version of sampler
samples_using_code_for_weights = mcmc.sampler(dnm, curX, curY, mcmc_steps, stan_representation, weights=weights, seed = ID) 
#build the modified dataset so that we can run the unweighted version of sampler
for i in range(len(idcs)):
    toAdd = np.ones(weights[i]-1,dtype=int) * (i) #list i a number of times equal to the weight of the ith index
    curX = np.append(curX, curX[toAdd], axis=0)
    curY = np.append(curY, curY[toAdd])
samples_using_standard_stan_code = mcmc.sampler(dnm, curX, curY, mcmc_steps, stan_representation, seed = ID)
#samples_using_code_for_weights = mcmc.sampler(dnm, curX, curY, mcmc_steps, stan_representation, weights=np.ones(curX.shape[0], dtype=int), seed=ID)
print("largest difference between samples using code for weights and samples using standard stan code (should be 0): ")
print(np.max(samples_using_code_for_weights - samples_using_standard_stan_code))
print("largest sample using standard stan code (if the previous value was not 0, our implementation may still be valid - just with minor numerical differences - if the value here is much bigger): ")
print(np.max(samples_using_standard_stan_code))

