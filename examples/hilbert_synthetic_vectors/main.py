from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
import time
import os
import pickle as pk

class IDProjector(bc.Projector):

  def update(self, wts, pts):
    pass

  def project(self, pts, grad=False):
    return pts

bc.util.set_verbosity('error')
n_trials = 5
Ms = np.unique(np.logspace(0., 4., 100, dtype=np.int32))
anms = ['FW', 'GIGA', 'OMP', 'IS', 'US']
algs = [bc.snnls.FrankWolfe, bc.snnls.GIGA, bc.snnls.OrthoPursuit, bc.snnls.ImportanceSampling, bc.snnls.UniformSampling]

if not os.path.exists('results/'):
  os.mkdir('results')

def snapshot(coreset, elapsed_time, trace):
  snapshot.num_calls += 1
  if snapshot.num_calls in Ms:
    #take a snapshot
    pass
snapshot.num_calls = 0

##########################################
## Test 1: gaussian data
##########################################
N = 10000
D = 100

for tr in range(n_trials):
  np.random.seed(tr)
  X = np.random.randn(N, D)

  for aidx, anm in enumerate(anms):
    print('data: gauss, trial ' + str(tr+1) + '/' + str(n_trials) + ', alg: ' + anm)
    alg = bc.HilbertCoreset(X, IDProjector(), snnls_alg = algs[aidx])

    trace = []
    alg.build(M, trace = trace)
    with open('results/gauss_'+anm+'_'+str(tr)+'.pk', 'wb') as f:
      pk.dump(trace, f)
    
    #for m, M in enumerate(Ms):
    #  t0 = time.time()
    #  alg.build(Ms[m] if m == 0 else Ms[m] - Ms[m-1]) #no explicit bound on size, just run correct # iterations (size will be upper bounded by # itrs)
    #  tf = time.time()
    #  cput[aidx, tr, m] = tf-t0 + cput[aidx, tr, m-1] if m > 0 else tf-t0
    #  wts, pts, idcs = alg.get()
    #  csize[aidx, tr, m] = (wts > 0).sum()
    #  err[aidx, tr, m] = alg.error()

#np.savez_compressed('gauss_results.npz', err=err, csize=csize, cput=cput, Ms = Ms, anms=anms)

##########################################
## Test 2: axis-aligned data
##########################################
 
N = 5000
N = 100

X = np.eye(N)

for tr in range(n_trials):
  np.random.seed(tr)
  for aidx, anm in enumerate(anms):
    print('data: axis, trial ' + str(tr+1) + '/' + str(n_trials) + ', alg: ' + anm)
    alg = bc.HilbertCoreset(X, IDProjector(), snnls_alg = algs[aidx])

    trace = []
    alg.build(M, trace = trace)
    with open('results/axis_'+anm+'_'+str(tr)+'.pk', 'wb') as f:
      pk.dump(trace, f)
    
    #for m, M in enumerate(Ms):
    #  t0 = time.time()
    #  alg.build(Ms[m] if m == 0 else Ms[m] - Ms[m-1], np.inf) #no explicit bound on size, just run correct # iterations (size will be upper bounded by # itrs)

    #  tf = time.time()
    #  cput[aidx, tr, m] = tf-t0 + cput[aidx, tr, m-1] if m > 0 else tf-t0
    #  wts, pts, idcs = alg.get()
    #  csize[aidx, tr, m] = (wts > 0).sum()
    #  err[aidx, tr, m] = alg.error()

#np.savez_compressed('axis_results.npz', err=err, csize=csize, cput=cput, Ms = Ms, anms=anms)

