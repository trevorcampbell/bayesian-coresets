import numpy as np
import sys

def nn_opt(x0, grd, nn_idcs=None, opt_itrs=1000, step_sched=lambda i : 1./(i+1), b1=0.9, b2=0.999, eps=1e-8, verbose=False):
    """
    Optimize the optimal optimal optimal optimal in x0.

    Args:
        x0: (array): write your description
        grd: (int): write your description
        nn_idcs: (str): write your description
        opt_itrs: (int): write your description
        step_sched: (todo): write your description
        i: (int): write your description
        i: (int): write your description
        b1: (int): write your description
        b2: (int): write your description
        eps: (float): write your description
        verbose: (bool): write your description
    """
  x = x0.copy()
  m1 = np.zeros(x.shape[0])
  m2 = np.zeros(x.shape[0])
  for i in range(opt_itrs):
    g = grd(x)
    if verbose:
      active_idcs = np.intersect1d(nn_idcs, np.where(x==0)[0])
      inactive_idcs = np.setdiff1d(np.arange(x.shape[0]), active_idcs)
      sys.stdout.write('itr ' + str(i+1) +'/'+str(opt_itrs)+': ||inactive constraint grads|| = ' + str(np.sqrt((g[inactive_idcs]**2).sum())) + '                \r')
      sys.stdout.flush()
    m1 = b1*m1 + (1.-b1)*g
    m2 = b2*m2 + (1.-b2)*g**2
    upd = step_sched(i)*m1/(1.-b1**(i+1))/(eps + np.sqrt(m2/(1.-b2**(i+1))))
    x -= upd
    #project onto x>=0
    if nn_idcs is None:
        x = np.maximum(x, 0.)
    else:
        x[nn_idcs] = np.maximum(x[nn_idcs], 0.)

  if verbose:
    sys.stdout.write('\n')
    sys.stdout.flush()
  return x
