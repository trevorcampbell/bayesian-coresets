"""
Original code for No-U-Turn Sampler (NUTS). (c) 2015 by Mat Leonard.
"""

from __future__ import division, print_function
import sys
import time
import numpy as np

def update_progress(current, total, width=30, end=False):
    bar_width = width
    block = int(round(bar_width * current/total))
    text = "\rProgress: [{0}] {1} of {2} steps".\
             format("#"*block + "-"*(bar_width-block), current, total)
    if end:
        text = text +'\n'
    sys.stdout.write(text)
    sys.stdout.flush()

def bern(p):
    return np.random.uniform() < p

def buildtree(x, r, u, v, j, e, x0, r0, H, dH, Emax):
    if j == 0:
        x1, r1 = leapfrog(x, r, v*e, dH)
        E = energy(H, x1, r1)
        E0 = energy(H, x0, r0)
        dE = E - E0

        n1 = (np.log(u) - dE <= 0)
        s1 = (np.log(u) - dE < Emax)
        return x1, r1, x1, r1, x1, n1, s1, np.min(np.array([1, np.exp(dE)])), 1
    else:
        xn, rn, xp, rp, x1, n1, s1, a1, na1 = \
            buildtree(x, r, u, v, j-1, e, x0, r0, H, dH, Emax)
        if s1 == 1:
            if v == -1:
                xn, rn, _, _, x2, n2, s2, a2, na2 = \
                    buildtree(xn, rn, u, v, j-1, e, x0, r0, H, dH, Emax)
            else:
                _, _, xp, rp, x2, n2, s2, a2, na2 = \
                    buildtree(xp, rp, u, v, j-1, e, x0, r0, H, dH, Emax)
            if bern(n2/max(n1 + n2, 1.)):
                x1 = x2

            a1 = a1 + a2
            na1 = na1 + na2

            dx = xp - xn
            s1 = s2 * (np.dot(dx, rn) >= 0) * \
                      (np.dot(dx, rp) >= 0)
            n1 = n1 + n2
        return xn, rn, xp, rp, x1, n1, s1, a1, na1

def leapfrog(x, r, step_size, grad):
    r1 = r + step_size/2*grad(x)
    x1 = x + step_size*r1
    r2 = r1 + step_size/2*grad(x1)
    return x1, r2

def accept(x, y, r_0, r, logp):
    E_new = energy(logp, y, r)
    E = energy(logp, x, r_0)
    A = np.min(np.array([0, E_new - E]))
    return (np.log(np.random.rand()) < A)

def energy(logp, x, r):
    return logp(x) - 0.5*np.dot(r, r)


def nuts(logp, gradlogp, x0, sample_steps = 1000, adapt=True, adapt_steps=100, burn_steps=100, step_size=0.25, Emax=1000, target_accept=0.65, gamma=0.05, k=0.75, t0=10., scale=None, progress_bar=True, seed=None):
  step_size = step_size / x0.shape[0]**(1/4.)
  if scale is None:
    scale = np.ones(x0.shape[0])
  if seed is not None:
    np.random.seed(seed)

  burn_steps = max(burn_steps, adapt_steps)
  
  Hbar = 0.
  ebar = 1.
  mu = np.log(step_size*10)

  samples = np.zeros((sample_steps, x0.shape[0]))
  x = x0.copy()

  start_time = time.time() # For progress bar
  for i in range(sample_steps+burn_steps):

    #do NUTS step
    r0 = np.random.multivariate_normal(np.zeros(x0.shape[0]), np.diagflat(scale))
    u = np.random.uniform()
    e = step_size
    xn, xp, rn, rp, y = x, x, r0, r0, x
    j, n, s = 0, 1, 1
    while s == 1:
        v = bern(0.5)*2 - 1
        if v == -1:
            xn, rn, _, _, x1, n1, s1, a, na = buildtree(xn, rn, u, v, j, e, x, r0,
                                                        logp, gradlogp, Emax)
        else:
            _, _, xp, rp, x1, n1, s1, a, na = buildtree(xp, rp, u, v, j, e, x, r0,
                                                        logp, gradlogp, Emax)

        if s1 == 1 and bern(np.min(np.array([1, n1/n]))):
            y = x1

        dx = xp - xn
        s = s1 * (np.dot(dx, rn) >= 0) * \
                 (np.dot(dx, rp) >= 0)
        n = n + n1
        j = j + 1
  
    if adapt:
      if i >= adapt_steps:
          step_size = ebar
      else:
          # Adapt step size
          w = 1./(i+1. + t0)
          Hbar = (1 - w)*Hbar + w*(target_accept - a/na)
          log_e = mu - ((i+1.)**.5/gamma)*Hbar
          step_size = np.exp(log_e)
          z = (i+1.)**(-k)
          ebar = np.exp(z*log_e + (1 - z)*np.log(ebar))

    x[:] = y

    if i >= burn_steps:
      samples[i-burn_steps, :] = x

    if progress_bar and time.time() - start_time > 1:
      update_progress(i+1, sample_steps+burn_steps)
      start_time = time.time()

  if progress_bar:
    update_progress(i+1, sample_steps+burn_steps, end=True)

  return samples


def rhat(chains):
  ''' 
  samples is an M x N x D np array; M = number of chains, N = number of samples, D = parameter dimension
  '''  
  M = chains.shape[0]
  N = chains.shape[1]
  D = chains.shape[2]
  chain_means = chains.mean(axis=1)
  total_mean = chain_means.mean(axis=0)
  chain_covs = np.zeros((M, D, D))
  for m in range(M):
    chain_covs[m, :, :] = np.cov(chains[m, :, :], rowvar=False)

  W = chain_covs.mean(axis=0)

  B = N/(M-1)* (chain_means - total_mean).T.dot(chain_means - total_mean)

  V = (N-1)/N*W + (M+1)/(M*N)*B

  Rh = np.linalg.eigvals(np.linalg.inv(W).dot(V)).max()
  return Rh
 















