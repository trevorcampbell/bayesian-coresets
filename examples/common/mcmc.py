import numpy as np
import pystan
import os
import pickle as pk
import time

def build_model(resfldr, modelName, model_code):
  if not os.path.exists(os.path.join(resfldr, modelName)): 
      print('STAN: building model')
      sm = pystan.StanModel(model_code=model_code)
      f = open(os.path.join(resfldr, modelName),'wb')
      pk.dump(sm, f)
      f.close()
  else:
      print("do we ever call this? like ever?")
      f = open(os.path.join(resfldr, modelName),'rb')
      sm = pk.load(f)
      f.close()
  return sm

def sampler(dnm, X, Y, resfldr, N_samples, stan_representation):

  if not os.path.exists('caching/'):
    os.mkdir('caching')

  if os.path.exists('caching/'+dnm+'_samples.npy'):
    print("Using cached samples for " + dnm)
    return np.load('caching/'+dnm+'_samples.npy')
  else:
    print('No MCMC samples found -- running STAN')
    print('STAN: loading data')
    Y[Y == -1] = 0 #convert to Stan LR label style if necessary

    sampler_data = {'x': X, 'y': Y.astype(int), 'd': X.shape[1], 'n': X.shape[0]}

    print('STAN: building/loading model')
    name, code = stan_representation
    sm = build_model(resfldr, name, code)

    print('STAN: sampling posterior: ' + dnm)
    t0 = time.process_time()
    thd = sampler_data['d']+1
    fit = sm.sampling(data=sampler_data, iter=N_samples*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
    samples = fit.extract(permuted=False)[:, 0, :thd]
    np.save(os.path.join(resfldr, dnm+'_samples.npy'), samples) 
    tf = time.process_time()
    np.save(os.path.join(resfldr, dnm+'_mcmc_time.npy'), tf-t0)
    return samples

  #if not cached, run sampler.
  #else return with message "using cached data" - this is definitely something mcmc should do, not individual examples
