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
  Z = data['y'][:, np.newaxis]*X
  Zt = data['yt'][:, np.newaxis]*Xt
  data.close()
  return Z, X[:, :-1], Y

N_samples = 10000
N_per = 2000


dnms = [(logistic_data_synth, 'lr', 'synth', X_synth.shape[1]+1, sml), (logistic_data_ds1, 'lr', 'ds1', X_ds1.shape[1]+1, sml), (logistic_data_phish, 'lr', 'phishing', X_phish.shape[1]+1, sml), (poisson_data_synth, 'poiss', 'synth', X_synthp.shape[1]+1, smp), (poisson_data_bike, 'poiss', 'biketrips', X_bike.shape[1]+1, smp), (poisson_data_air, 'poiss', 'airportdelays', X_air.shape[1]+1, smp)]


def sampler(dnm, datafldr, resfldr, N_samples, N_per):
  Z, X, Y = load_data(os.path.join(datafldr,dnm))
  Y[Y == -1] = 0 #convert to Stan LR label style if necessary

  sampler_data = {'x': X, 'y':Y.astype(int), 'd': X.shape[1], 'n': X.shape[0]}

  if not os.path.exists(os.path.join(resfldr,'pystan_model_logistic.pk')): 
    sml = pystan.StanModel(model_code=logistic_code)
    f = open(os.path.join(resfldr,'pystan_model_logistic.pk'),'wb')
    pk.dump(sml, f)
    f.close()
  else:
    f = open(os.path.join(resfldr,'pystan_model_logistic.pk'),'rb')
    sml = pk.load(f)
    f.close()

  if not os.path.exists(os.path.join(resfldr,'pystan_model_poisson.pk')): 
    sml = pystan.StanModel(model_code=poisson_code)
    f = open(os.path.join(resfldr,'pystan_model_poisson.pk'),'wb')
    pk.dump(sml, f)
    f.close()
  else:
    f = open(os.path.join(resfldr,'pystan_model_poisson.pk'),'rb')
    sml = pk.load(f)
    f.close()

  print('sampling posterior: ' + dfnm)
  t0 = time.process_time()
  samples = np.zeros((0, sampler_data['d']))
  for i in range(int(N_samples/N_per)):
    if i == 0: 
      fit = sm.sampling(data=data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
    else:
      try:
        fit = sm.sampling(data=sampler_data, init=[dict(theta=samples[-1,:-1], theta0=samples[-1,-1])], iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
      except:
        print('initialization failed, trying again')
        fit = sm.sampling(data=sampler_data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
    samples = np.vstack((samples, fit.extract(permuted=False)[:, 0, :sampler_data['d']]))
  np.save(os.path.join(resfldr, dnm+'_samples.npy'), samples) 
  tf = time.process_time()
  np.save(os.path.join(resfldr, dnm+'_mcmc_time.npy', tf-t0)


