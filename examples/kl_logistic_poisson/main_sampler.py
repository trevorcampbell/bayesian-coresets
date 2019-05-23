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

print('loading logistic regression datasets')
Z_synth, X_synth, Y_synth = load_data('./lr/synth.npz')
Y_synth[Y_synth == -1] = 0
Z_ds1, X_ds1, Y_ds1 = load_data('./lr/ds1.npz')
Y_ds1[Y_ds1 == -1] = 0
Z_phish, X_phish, Y_phish = load_data('./lr/phishing.npz')
Y_phish[Y_phish == -1] = 0
print('loading poisson regression datasets')
Z_synthp, X_synthp, Y_synthp = load_data('./poiss/synth.npz')
Y_synthp[Y_synthp == -1] = 0
Z_bike, X_bike, Y_bike = load_data('./poiss/biketrips.npz')
Y_bike[Y_bike == -1] = 0
Z_air, X_air, Y_air = load_data('./poiss/airportdelays.npz')
Y_air[Y_air == -1] = 0


logistic_data_synth = {'x': X_synth, 'y':Y_synth.astype(int), 'd': X_synth.shape[1], 'n': X_synth.shape[0]}
logistic_data_ds1 = {'x': X_ds1, 'y':Y_ds1.astype(int), 'd': X_ds1.shape[1], 'n': X_ds1.shape[0]}
logistic_data_phish = {'x': X_phish, 'y':Y_phish.astype(int), 'd': X_phish.shape[1], 'n': X_phish.shape[0]}
poisson_data_synth = {'x': X_synthp, 'y':Y_synthp.astype(int), 'd': X_synthp.shape[1], 'n': X_synthp.shape[0]}
poisson_data_bike = {'x': X_bike, 'y':Y_bike.astype(int), 'd': X_bike.shape[1], 'n': X_bike.shape[0]}
poisson_data_air = {'x': X_air, 'y':Y_air.astype(int), 'd': X_air.shape[1], 'n': X_air.shape[0]}

if not os.path.exists('pystan_model_logistic.pk'): 
  sml = pystan.StanModel(model_code=logistic_code)
  f = open('pystan_model_logistic.pk','wb')
  pk.dump(sml, f)
  f.close()
else:
  f = open('pystan_model_logistic.pk','rb')
  sml = pk.load(f)
  f.close()

if not os.path.exists('pystan_model_poisson.pk'): 
  smp = pystan.StanModel(model_code=poisson_code)
  f = open('pystan_model_poisson.pk','wb')
  pk.dump(smp, f)
  f.close()
else:
  f = open('pystan_model_poisson.pk','rb')
  smp = pk.load(f)
  f.close()

N_samples = 10000
N_per = 2000
#dnms = [(logistic_data_synth, 'lr', 'synth', X_synth.shape[1]+1, sml), (logistic_data_ds1, 'lr', 'ds1', X_ds1.shape[1]+1, sml), (logistic_data_phish, 'lr', 'phishing', X_phish.shape[1]+1, sml), (poisson_data_synth, 'poiss', 'synth', X_synthp.shape[1]+1, smp), (poisson_data_bike, 'poiss', 'biketrips', X_bike.shape[1]+1, smp), (poisson_data_air, 'poiss', 'airportdelays', X_air.shape[1]+1, smp)]
dnms = [(logistic_data_synth, 'lr', 'synth', X_synth.shape[1]+1, sml), (poisson_data_synth, 'poiss', 'synth', X_synthp.shape[1]+1, smp)]
for data, fldr, nm, d, sm in dnms:
  print('sampling posterior: ' + nm)
  t0 = time.process_time()
  samples = np.zeros((0, d))
  for i in range(int(N_samples/N_per)):
    if i == 0: 
      fit = sm.sampling(data=data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
    else:
      try:
        fit = sm.sampling(data=data, init=[dict(theta=samples[-1,:-1], theta0=samples[-1,-1])], iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
      except:
        print('initialization failed, trying again')
        fit = sm.sampling(data=data, iter=N_per*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
    samples = np.vstack((samples, fit.extract(permuted=False)[:, 0, :d]))
  np.save(fldr+'_'+nm+'_samples.npy', samples) 
  tf = time.process_time()
  np.save(fldr+'_'+nm+'_mcmc_time.npy', tf-t0)


