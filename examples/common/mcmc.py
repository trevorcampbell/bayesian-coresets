import numpy as np
import pystan
import os
import pickle as pk
import time

def build_model(cache_folder, modelName, model_code):
  if cache_folder and os.path.exists(os.path.join(cache_folder, modelName)):
    f = open(os.path.join(cache_folder, modelName),'rb')
    sm = pk.load(f)
    f.close()
  else: 
    print('STAN: building model')
    sm = pystan.StanModel(model_code=model_code)

    if cache_folder: 
      f = open(os.path.join(cache_folder, modelName),'wb')
      pk.dump(sm, f)
      f.close()

  return sm

def sampler(dnm, X, Y, N_samples, stan_representation, weights = None, cache_folder = None, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True):

  if cache_folder and not os.path.exists(cache_folder):
    os.mkdir(cache_folder)

  if cache_folder and os.path.exists(cache_folder+dnm+'_samples.npy'):
    print("Using cached samples for " + dnm)
    return np.load(cache_folder+dnm+'_samples.npy')
  else:
    print('No MCMC samples found -- running STAN')
    print('STAN: loading data')
    Y[Y == -1] = 0 #convert to Stan LR label style if necessary

    sampler_data = {'x': X, 'y': Y.astype(int), 'w': weights if weights != None else [], 'd': X.shape[1], 'n': X.shape[0]}

    print('STAN: building/loading model')
    name, code = stan_representation
    sm = build_model(cache_folder, name, code)

    if weights != None: #presumably weights is only ever None in the case where we're using the full dataset - this code may need to be adjusted to handle the case of a coreset of size 0
      print('Altering cpp code used by stan to allow weighted data')
      sm.model_cppcode = load_modified_cpp_code(cache_folder, modelName, model_code)

    print('STAN: sampling posterior: ' + dnm)
    t0 = time.process_time()
    thd = sampler_data['d']+1
    #call sampling with N_samples actual iterations, and some number of burn iterations
    fit = sm.sampling(data=sampler_data, iter=N_samples*2, chains=chains, control=control, verbose=verbose)
    samples = fit.extract(permuted=False)[:, 0, :thd]
    if cache_folder:
      np.save(os.path.join(cache_folder, dnm+'_samples.npy'), samples) 
      tf = time.process_time()
      np.save(os.path.join(cache_folder, dnm+'_mcmc_time.npy'), tf-t0)
    return samples

#Takes in the name of and code for a statistical model that allows for stan to run MCMC, and returns cpp code that stan can use to 
#perform MCMC on a coreset.  
#TODO: throw error if cach_folder is none, or find another way to handle this case
#TODO: modify this to use hashes
def load_modified_cpp_code(cache_folder, modelName, model_code):
  codeHash = 'cppCode'#this will eventually refer to the actual hash of the model code, but for now I just want to make sure this framework is valid
  if os.path.exists(os.path.join(cache_folder, modelName+ '_cppCode')):
    f = open(os.path.join(cache_folder, modelName+'_cppCode'),'rb')
    modified_code = pk.load(f)
    return modified_code
  else: 
    return EnvironmentError("No modified code to handle weighted data present - unable to use stan for MCMC sampling")