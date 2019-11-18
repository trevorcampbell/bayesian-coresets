from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
import time

np.random.seed(3)

bc.util.set_verbosity('error')

n_trials = 5
Ms = np.unique(np.logspace(0., 4., 100, dtype=np.int32))

anms = ['FW', 'GIGA', 'OMP', 'IS', 'US']
algs = [bc.snnls.FrankWolfe, bc.snnls.GIGA, bc.snnls.OrthoPursuit, bc.snnls.ImportanceSampling, bc.snnls.UniformSampling]

##########################################
## Test 1: gaussian data
##########################################
N = 10000
D = 100

err = np.zeros((len(anms), n_trials, Ms.shape[0]))
scaled_err = np.zeros((len(anms), n_trials, Ms.shape[0]))
csize = np.zeros((len(anms), n_trials, Ms.shape[0]))
cput = np.zeros((len(anms), n_trials, Ms.shape[0]))
for tr in range(n_trials):
  X = np.random.randn(N, D)
  
  def loglike(idcs, samps):
    return X[idcs, :samps]

  for aidx, anm in enumerate(anms):
    print('data: gauss, trial ' + str(tr+1) + '/' + str(n_trials) + ', alg: ' + anm)
    alg = bc.HilbertCoreset(loglike, X.shape[0], X.shape[1], snnls = algs[aidx])

    for m, M in enumerate(Ms):
      t0 = time.time()
      alg.build(Ms[m] if m == 0 else Ms[m] - Ms[m-1]) #build to size M 
      tf = time.time()
      cput[aidx, tr, m] = tf-t0 + cput[aidx, tr, m-1] if m > 0 else tf-t0
      wts, idcs = alg.weights()
      csize[aidx, tr, m] = (wts > 0).sum()
      err[aidx, tr, m] = alg.error()
      wts = wts.copy() 
      idcs = idcs.copy()
      alg.optimize()
      opt_err[aidx, tr, m] = alg.error()
      alg._overwrite(wts, idcs)

np.savez_compressed('gauss_results.npz', err=err, csize=csize, cput=cput, opt_err=opt_err, Ms = Ms, anms=anms)

##########################################
## Test 2: axis-aligned data
##########################################
 
N = 5000
N = 100

X = np.eye(N)
def loglike(idcs, samps):
    return X[idcs, :samps]

err = np.zeros((len(anms), n_trials, Ms.shape[0]))
scaled_err = np.zeros((len(anms), n_trials, Ms.shape[0]))
csize = np.zeros((len(anms), n_trials, Ms.shape[0]))
cput = np.zeros((len(anms), n_trials, Ms.shape[0]))
for tr in range(n_trials):
  for aidx, anm in enumerate(anms):
    print('data: axis, trial ' + str(tr+1) + '/' + str(n_trials) + ', alg: ' + anm)
    alg = bc.HilbertCoreset(loglike, X.shape[0], X.shape[1], snnls = algs[aidx])

    for m, M in enumerate(Ms):
      t0 = time.time()
      alg.build(Ms[m] if m == 0 else Ms[m] - Ms[m-1]) #build to size M 
      tf = time.time()
      cput[aidx, tr, m] = tf-t0 + cput[aidx, tr, m-1] if m > 0 else tf-t0
      wts, idcs = alg.weights()
      csize[aidx, tr, m] = (wts > 0).sum()
      err[aidx, tr, m] = alg.error()
      wts = wts.copy() 
      idcs = idcs.copy()
      alg.optimize()
      opt_err[aidx, tr, m] = alg.error()
      alg._overwrite(wts, idcs)

np.savez_compressed('axis_results.npz', err=err, csize=csize, cput=cput, opt_err=opt_err, Ms = Ms, anms=anms)

