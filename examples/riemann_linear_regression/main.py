import numpy as np
import bayesiancoresets as bc
import os, sys
from scipy.stats import multivariate_normal
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import model_linreg


np.random.seed(1)

#load data and compute true posterior
#each row of x is [lat, lon, price]
print('Loading data')
x = np.load('../data/prices2018.npy')

N_subsample = 10000

print('taking a random subsample')
#get a random subsample of it
idcs = np.arange(x.shape[0])
np.random.shuffle(idcs)
x = x[idcs[:N_subsample], :]

#log transform
x[:, 2] = np.log10(x[:, 2])

#get empirical mean/std
datastd = x[:,2].std()
datamn = x[:,2].mean()


#bases of increasing size; the last one is effectively a constant
basis_unique_scales = np.array([.2, .4, .8, 1.2, 1.6, 2., 100])
basis_unique_counts = np.hstack((50*np.ones(6, dtype=np.int64), 1))

#the dimension of the scaling vector for the above bases
d = basis_unique_counts.sum()
print('Basis dimension: ' + str(d))

#model params
mu0 = datamn*np.ones(d)
Sig0 = (datastd**2+datamn**2)*np.eye(d)
#Sig = datastd**2*np.eye(d)
#SigL = np.linalg.cholesky(Sig)
Sig0inv = np.linalg.inv(Sig0)
#Siginv = np.linalg.inv(Sig)
#SigLInv = np.linalg.inv(SigL)

#experiment params
M = 500
trials = np.arange(1)
opt_itrs = 50
proj_dim = 20
pihat_noise =0.75


trnum = 0
for t in trials:
  trnum += 1
  #generate basis functions by uniformly randomly picking locations in the dataset
  print('Trial ' + str(trnum) + '/' + str(trials.shape[0]))
  print('Creating bases')
  basis_scales = np.array([])
  basis_locs = np.zeros((0,2))
  for i in range(basis_unique_scales.shape[0]):
    basis_scales = np.hstack((basis_scales, basis_unique_scales[i]*np.ones(basis_unique_counts[i])))
    idcs = np.random.choice(np.arange(x.shape[0]), replace=False, size=basis_unique_counts[i])
    basis_locs = np.vstack((basis_locs, x[idcs, :2]))

  print('Converting bases and observations into X/Y matrices')
  #convert basis functions + observed data locations into a big X matrix
  X = np.zeros((x.shape[0], basis_scales.shape[0]))
  for i in range(basis_scales.shape[0]):
    X[:, i] = np.exp( -((x[:, :2] - basis_locs[i, :])**2).sum(axis=1) / (2*basis_scales[i]**2) )
  Y = x[:, 2]

  #get true posterior
  print('Computing true posterior')
  mup, Sigp = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, X, Y, np.ones(X.shape[0]))
  Sigpinv = np.linalg.inv(Sigp)

  #create function to output log_likelihood given param samples
  print('Creating log-likelihood function')
  log_likelihood = lambda samples : model_linreg.potentials(datastd**2, X, Y, samples)

  #create tangent space for well-tuned Hilbert coreset alg
  print('Creating tuned tangent space for Hilbert coreset construction')
  T_true = bc.MonteCarloFiniteTangentSpace(log_likelihood, lambda sz : np.random.multivariate_normal(mup, 9*Sigp, sz), proj_dim)

  #create tangent space for poorly-tuned Hilbert coreset alg
  print('Creating untuned tangent space for Hilbert coreset construction')
  U = np.random.rand()
  muhat = U*mup + (1.-U)*mu0
  Sighat = U*Sigp + (1.-U)*Sig0
  #now corrupt the smoothed pihat
  muhat += pihat_noise*np.sqrt((muhat**2).sum())*np.random.randn(muhat.shape[0])
  Sighat *= np.exp(-2*pihat_noise*np.fabs(np.random.randn()))
  T_noisy = bc.MonteCarloFiniteTangentSpace(log_likelihood, lambda sz : np.random.multivariate_normal(muhat, Sighat, sz), proj_dim)

  #create exact tangent space factory for Riemann coresets
  def tangent_space_factory(wts, idcs):
    w = np.zeros(X.shape[0])
    w[idcs] = wts
    muw, Sigw = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, X, Y, w)

    evs = np.linalg.eigvalsh(Sigw)
    if np.any(evs < 0):
      Sigw -= 2*evs.min()*np.eye(Sigw.shape[0])

    nu = X.dot(np.linalg.cholesky(Sigw))/datastd**2
   
    dnu = (Y - X.dot(muw))[:,np.newaxis]*nu

    #correct but quadratic cost...
    #nu = np.hstack((dnu, 1./np.sqrt(2.)*( nu[:, :, np.newaxis]*nu[:, np.newaxis, :]).reshape( (nu.shape[0], nu.shape[1]**2))))
    #only diagonal terms
    nu = np.hstack((dnu, 1./np.sqrt(2.)*nu**2))
    #none of the quartic terms
    #nu = dnu
    
    return bc.FixedFiniteTangentSpace(nu, wts, idcs)

  def nulltsf(wts, idcs):
    return bc.FixedFiniteTangentSpace(np.zeros((X.shape[0], 2)), wts, idcs)
    
   
  #create coreset construction objects
  print('Creating coreset construction objects')
  riemann_one = bc.SparseVICoreset(x.shape[0], tangent_space_factory, opt_itrs=opt_itrs, update_single=True)
  riemann_full = bc.SparseVICoreset(x.shape[0], tangent_space_factory, opt_itrs=opt_itrs, update_single=False, step_sched=lambda i : 20*np.sqrt(1/(1.+i)))
  giga_true = bc.GIGACoreset(T_true)
  giga_noisy = bc.GIGACoreset(T_noisy)
  unif = bc.UniformSamplingKLCoreset(x.shape[0], nulltsf)
 
  #algs = [riemann_one, riemann_full, giga_true, giga_noisy, unif]
  algs = [riemann_full, giga_true, giga_noisy]
  #nms = ['SVI1', 'SVIF', 'GIGAT', 'GIGAN', 'RAND']
  nms = ['SVIF', 'GIGAT', 'GIGAN']


  print('Building coresets')
  #build coresets
  for nm, alg in zip(nms, algs):
    #create coreset construction objects
    w = np.zeros((M+1, x.shape[0]))
    w_opt = np.zeros((M+1, x.shape[0]))
    for m in range(1, M+1):
      print('trial: ' + str(trnum)+'/'+str(trials.shape[0])+' alg: ' + nm + ' ' + str(m) +'/'+str(M))

      alg.build(m)
      #store weights
      wts, idcs = alg.weights()
      w[m, idcs] = wts
      #store optimized weights
      alg.optimize()
      wts_opt, idcs_opt = alg.weights()
      w_opt[m, idcs_opt] = wts_opt
      #restore pre-opt weights
      alg._overwrite(idcs, wts)
      #printouts for debugging purposes
      #print('reverse KL: ' + str(weighted_post_KL(mu0, Sig0inv, Siginv, x, w_opt[m, :], reverse=True)))
      #print('reverse KL opt: ' + str(weighted_post_KL(mu0, Sig0inv, Siginv, x, w_opt[m, :], reverse=True)))

    muw = np.zeros((M+1, mu0.shape[0]))
    Sigw = np.zeros((M+1,mu0.shape[0], mu0.shape[0]))
    rklw = np.zeros(M+1)
    fklw = np.zeros(M+1)
    for m in range(M+1):
      print('KL divergence computation for trial: ' + str(trnum)+'/'+str(trials.shape[0])+' alg: ' + nm + ' ' + str(m) +'/'+str(M))
      muw[m, :], Sigw[m, :, :] = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, X, Y, w[m, :])
      rklw[m] = model_linreg.weighted_post_KL(mu0, Sig0inv, datastd**2, X, Y, w[m,:], reverse=True)
      fklw[m] = model_linreg.weighted_post_KL(mu0, Sig0inv, datastd**2, X, Y, w[m,:], reverse=False)
    muw_opt = np.zeros((M+1, mu0.shape[0]))
    Sigw_opt = np.zeros((M+1,mu0.shape[0], mu0.shape[0]))
    rklw_opt = np.zeros(M+1)
    fklw_opt = np.zeros(M+1)
    for m in range(M+1):
      print('Optimized KL divergence computation for trial: ' + str(trnum)+'/'+str(trials.shape[0])+' alg: ' + nm + ' ' + str(m) +'/'+str(M))
      muw_opt[m, :], Sigw_opt[m, :, :] = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, X, Y, w_opt[m, :])
      rklw_opt[m] = model_linreg.weighted_post_KL(mu0, Sig0inv, datastd**2, X, Y, w_opt[m,:], reverse=True)
      fklw_opt[m] = model_linreg.weighted_post_KL(mu0, Sig0inv, datastd**2, X, Y, w_opt[m,:], reverse=False)


    if not os.path.exists('results/'):
      os.mkdir('results')
    print('saving result for trial: ' + str(trnum)+'/'+str(trials.shape[0])+' alg: ' + nm)
    np.savez('results/results_'+nm+'_' + str(t)+'.npz', x=x, mu0=mu0, Sig0=Sig0, mup=mup, Sigp=Sigp, w=w, w_opt=w_opt,
                                   muw=muw, Sigw=Sigw, rklw=rklw, fklw=fklw,
                                   muw_opt=muw_opt, Sigw_opt=Sigw_opt, rklw_opt=rklw_opt, fklw_opt=fklw_opt,
                                   basis_scales=basis_scales, basis_locs=basis_locs, datastd=datastd)
  
