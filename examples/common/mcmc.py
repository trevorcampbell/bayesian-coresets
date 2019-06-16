import numpy as np
import pystan
from stan_code import logistic_code, poisson_code
import os
import pickle as pk
import time

def load_data(dnm):
  data = np.load(dnm)
  X = data['X']
  Y = data['y']
  Xt = data['Xt']
  #standardize the covariates; last col is intercept, so no stdization there
  m = X[:, :-1].mean(axis=0)
  V = np.cov(X[:, :-1], rowvar=False)+1e-12*np.eye(X.shape[1]-1)
  X[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (X[:, :-1] - m).T).T
  Xt[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (Xt[:, :-1] - m).T).T
  data.close()
  return X[:, :-1], Y

def sampler(dnm, lr, datafldr, resfldr, N_samples):
  print('STAN: loading data')
  X, Y = load_data(os.path.join(datafldr,dnm+'.npz'))
  Y[Y == -1] = 0 #convert to Stan LR label style if necessary

  sampler_data = {'x': X, 'y': Y.astype(int), 'd': X.shape[1], 'n': X.shape[0]}

  print('STAN: building/loading model')
  if lr:
    if not os.path.exists(os.path.join(resfldr,'pystan_model_logistic.pk')): 
      print('STAN: building LR model')
      sm = pystan.StanModel(model_code=logistic_code)
      f = open(os.path.join(resfldr,'pystan_model_logistic.pk'),'wb')
      pk.dump(sm, f)
      f.close()
    else:
      f = open(os.path.join(resfldr,'pystan_model_logistic.pk'),'rb')
      sm = pk.load(f)
      f.close()
  else:
    if not os.path.exists(os.path.join(resfldr,'pystan_model_poisson.pk')): 
      print('STAN: building PR model')
      sm = pystan.StanModel(model_code=poisson_code)
      f = open(os.path.join(resfldr,'pystan_model_poisson.pk'),'wb')
      pk.dump(sm, f)
      f.close()
    else:
      f = open(os.path.join(resfldr,'pystan_model_poisson.pk'),'rb')
      sm = pk.load(f)
      f.close()

  print('STAN: sampling posterior: ' + dnm)
  t0 = time.process_time()
  thd = sampler_data['d']+1
  fit = sm.sampling(data=sampler_data, iter=N_samples*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
  samples = fit.extract(permuted=False)[:, 0, :thd]
  np.save(os.path.join(resfldr, dnm+'_samples.npy'), samples) 
  tf = time.process_time()
  np.save(os.path.join(resfldr, dnm+'_mcmc_time.npy'), tf-t0)

  #t0 = time.process_time()
  #thd = sampler_data['d']+1
  #samples = np.zeros((0, thd))
  #for i in range(int(N_samples/N_per)):
  #  if i == 0: 
  #    fit = sm.sampling(data=sampler_data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
  #  else:
  #    try:
  #      fit = sm.sampling(data=sampler_data, init=[dict(theta=samples[-1,:-1], theta0=samples[-1,-1])], iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
  #    except:
  #      print('STAN: initialization failed, trying again')
  #      fit = sm.sampling(data=sampler_data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
  #  samples = np.vstack((samples, fit.extract(permuted=False)[:, 0, :thd]))
  #np.save(os.path.join(resfldr, dnm+'_samples.npy'), samples) 
  #tf = time.process_time()
  #np.save(os.path.join(resfldr, dnm+'_mcmc_time.npy'), tf-t0)
