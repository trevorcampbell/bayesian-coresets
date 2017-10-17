import numpy as np
from lapjv import lapjv


def wasserstein1(sample1, sample2):
  #reshape samples to be of the same size 
  if sample1.shape[0] > sample2.shape[0]:
    z1 = sample1[np.random.randint(sample1.shape[0], size=sample2.shape[0]), :]
    z2 = sample2
  elif sample1.shape[0] < sample2.shape[0]:
    z1 = sample1
    z2 = sample2[np.random.randint(sample2.shape[0], size=sample1.shape[0]), :]
  else:
    z1 = sample1
    z2 = sample2

  nrmsq1 = (z1**2).sum(axis=1)
  nrmsq2 = (z2**2).sum(axis=1)
  c = np.sqrt(nrmsq1[:, np.newaxis] + nrmsq2 - 2.*(z1.dot(z2.T)))
  cst, col_ind, row_ind = lapjv(np.floor(c/c.max()*1000000.).astype(int))
  return c[row_ind, range(c.shape[0])].sum()/z1.shape[0]

def _ensure_positive_int(val, name):
    if not isinstance(val, int) or val <= 0:
        raise ValueError("'%s' must be a positive integer")
    return True


def _ensure_callable(val, name):
    if not callable(val):
        raise ValueError("'%s' must be a callable")
    return True


def _adapt_param(value, i, log_accept_prob, target_rate, const=3):
    """
    Adapt the value of a parameter.
    """
    new_val = value + const*(np.exp(log_accept_prob) - target_rate)/np.sqrt(i+1)
    tmpconst = const
    while new_val <= 0:
      tmpconst /= 2.
      new_val = value + tmpconst*(np.exp(log_accept_prob) - target_rate)/np.sqrt(i+1)
      
    # new_val = max(min_val, min(max_val, new_val))
    return new_val


def mh(x0, p, q, sample_q, steps=1, warmup=None, thin=1,
       proposal_param=None, target_rate=0.234):
    """
    (Adaptive) Metropolis Hastings sampling.

    Parameters
    ----------
    x0 : object
        The initial state.

    p : function
        Accepts one argument `x` and outputs the log probability density of the
        target distribution at `x`.

    q : function or None
        Accepts two arguments, `x` and `xf`. Outputs the log proposal density
        of going from `x` to `xf`. None indicates the proposal is symmetric,
        so there is no need to calculate the proposal probability when
        deciding whether to accept the move to `xf`.

    sample_q : function
        Accepts one argument `x` and proposes `xf` given `x`.

    steps : int, optional
        The number of MH steps to take. Default is 1.

    warmup : int, optional
        The number of warmup (aka burnin) iterations. Default is ``steps/2``.

    thin : int, optional
        Period for saving samples. Default is 1.

    proposal_param : numeric, optional
        If proveided then use adaptive MH targeting an accept rate of
        `target_rate`. In this case `sample_q` and `q` should both accept
        `proposal_param` as an additional final argument. Default is None.

    target_rate : float, optional
        Default is 0.234.

    Returns
    -------
    samples : array with length ``(steps - warmup) / thin``

    accept_rate : float
        Calculated from non-warmup iterations.
    """
    # Validate parameters
    _ensure_callable(p, 'p')
    if q is not None:
        _ensure_callable(q, 'q')
    _ensure_callable(sample_q, 'sample_q')
    _ensure_positive_int(steps, 'steps')
    _ensure_positive_int(thin, 'thin')
    if warmup is None:
        warmup = steps / 2
    else:
        _ensure_positive_int(warmup, 'warmup')
        if warmup >= steps:
            raise ValueError("Number of warmup iterations is %d, which is "
                             "greater than the total number of steps, %d" %
                             (warmup, steps))

    # Run (adaptive) MH algorithm
    accepts = 0.0
    xs = []
    x = x0
    for step in range(steps):
        # Make a proposal
        p0 = p(x)
        if proposal_param is None:
            xf = sample_q(x)
        else:
            xf = sample_q(x, proposal_param)
        pf = p(xf)

        # Compute acceptance ratio and accept or reject
        odds = pf - p0
        if q is not None:
            if proposal_param is None:
                qf, qr = q(x, xf), q(xf, x)
            else:
                qf, qr = q(x, xf, proposal_param), q(xf, x, proposal_param)
            odds += qr - qf
        if proposal_param is not None and step < warmup:
                proposal_param = _adapt_param(proposal_param, step,
                                              min(0, odds), target_rate)
        if np.log(npr.rand()) < odds:
            x = xf
            if step >= warmup:
                accepts += 1

        if step >= warmup and (step - warmup) % thin == 0:
            xs.append(x)

    accept_rate = accepts / (steps - warmup)
    if len(xs) > 1:
        return xs, accept_rate
    else:
        return xs[0], accept_rate

