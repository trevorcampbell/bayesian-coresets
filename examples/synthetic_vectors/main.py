from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate lots of warnings re: reaching numeric limits; ignore those here


n_trials = 5
Ms = np.unique(np.logspace(0., 4., 100, dtype=np.int32))
#anms = ['GIGA', 'FW', 'MP', 'FSW', 'OMP', 'LAR', 'IS', 'RND']
#algs = [bc.GIGACoreset, bc.FrankWolfeCoreset, bc.MatchingPursuitCoreset, bc.ForwardStagewiseCoreset, bc.OrthoPursuitCoreset, bc.LARCoreset, bc.VectorImportanceSamplingCoreset, bc.VectorUniformSamplingCoreset]


anms = ['FW']
algs = [bc.FrankWolfeCoreset]

#print(Ms)


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
  T = bc.FixedFiniteTangentSpace(X)
  XS = X.sum(axis=0)
  for aidx, anm in enumerate(anms):
    print('data: gauss, trial ' + str(tr+1) + '/' + str(n_trials) + ', alg: ' + anm)
    alg = algs[aidx](T)

    for m, M in enumerate(Ms):
      #print('iteration ' + str(m) + ' at M = ' + str(M))
      t0 = time.time()
      alg.build(M)
      tf = time.time()
      cput[aidx, tr, m] = tf-t0 + cput[aidx, tr, m-1] if m > 0 else tf-t0
      wts, idcs = alg.weights()
      err[aidx, tr, m] = np.sqrt((((wts[:, np.newaxis]*X[idcs,:]).sum(axis=0) - XS)**2).sum())
      alpha = T.optimal_scaling(wts, idcs)
      scaled_err[aidx, tr, m] = np.sqrt((((alpha*wts[:, np.newaxis]*X[idcs,:]).sum(axis=0) - XS)**2).sum())
      csize[aidx, tr, m] = (wts > 0).sum()

np.savez_compressed('gauss_results.npz', err=err, csize=csize, cput=cput, scaled_err=scaled_err, Ms = Ms, anms=anms)

##########################################
## Test 2: axis-aligned data
##########################################
 
N = 5000
N = 100

X = np.eye(N)
XS = np.ones(N)
T = bc.FixedFiniteTangentSpace(X)

err = np.zeros((len(anms), n_trials, Ms.shape[0]))
scaled_err = np.zeros((len(anms), n_trials, Ms.shape[0]))
csize = np.zeros((len(anms), n_trials, Ms.shape[0]))
cput = np.zeros((len(anms), n_trials, Ms.shape[0]))
for tr in range(n_trials):
  for aidx, anm in enumerate(anms):
    print('data: axis, trial ' + str(tr+1) + '/' + str(n_trials) + ', alg: ' + anm)
    alg = algs[aidx](T)

    for m, M in enumerate(Ms):
      t0 = time.time()
      alg.build(M)
      tf = time.time()
      cput[aidx, tr, m] = tf-t0 + cput[aidx, tr, m-1] if m > 0 else tf-t0
      wts, idcs = alg.weights()
      err[aidx, tr, m] = np.sqrt((((wts[:, np.newaxis]*X[idcs, :]).sum(axis=0) - XS)**2).sum())
      alpha = T.optimal_scaling(wts, idcs)
      scaled_err[aidx, tr, m] = np.sqrt((((alpha*wts[:, np.newaxis]*X[idcs, :]).sum(axis=0) - XS)**2).sum())
      csize[aidx, tr, m] = (wts>0).sum()

np.savez_compressed('axis_results.npz', err=err, csize=csize, cput=cput, scaled_err=scaled_err, Ms = Ms, anms=anms)

