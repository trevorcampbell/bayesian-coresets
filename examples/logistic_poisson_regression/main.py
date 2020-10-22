from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
import time
import sys, os
import argparse
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import mcmc
import results
import plotting
from model_gaussian import KL

def get_laplace(wts, Z, mu0, model, diag = False):
  trials = 10
  Zw = Z[wts>0, :]
  ww = wts[wts>0]
  while True:
    try:
      res = minimize(lambda mu : -model.log_joint(Zw, mu, ww)[0], mu0, jac=lambda mu : -model.grad_th_log_joint(Zw, mu, ww)[0,:])
      mu = res.x
    except Exception as e:
      print(e)
      mu0 = mu0.copy()
      mu0 += np.sqrt((mu0**2).sum())*0.1*np.random.randn(mu0.shape[0])
      trials -= 1
      if trials <= 0:
        print('Tried laplace opt 10 times, failed')
        mu = mu0
        break
      continue
    break
  if diag:
    LSigInv = np.sqrt(-model.diag_hess_th_log_joint(Zw, mu, ww)[0,:])
    LSig = 1./LSigInv
  else:
    LSigInv = np.linalg.cholesky(-model.hess_th_log_joint(Zw, mu, ww)[0,:,:])
    LSig = solve_triangular(LSigInv, np.eye(LSigInv.shape[0]), lower=True, overwrite_b = True, check_finite = False)
  return mu, LSig, LSigInv

def plot(arguments):
    # load only the results that match (avoid high mem usage)
    to_match = vars(arguments)
    #remove any ignored params
    if arguments.summarize is not None:
        for nm in arguments.summarize:
            to_match.pop(nm, None)
    #remove any legend param
    to_match.pop(arguments.plot_legend, None)
    #load cols from results dfs that match remaining keys
    resdf = results.load_matching(to_match)
    #call the generic plot function
    plotting.plot(arguments, resdf)

def run(arguments):

    # check if result already exists for this run, and if so, quit
    if results.check_exists(arguments):
      print('Results already exist for arguments ' + str(arguments))
      print('Quitting.')
      quit()

    #######################################
    #######################################
    ########### Step 0: Setup #############
    #######################################
    #######################################

    np.random.seed(arguments.trial)
    bc.util.set_verbosity(arguments.verbosity) 

    if arguments.coreset_size_spacing == 'log':
        Ms = np.unique(np.logspace(0., np.log10(arguments.coreset_size_max), arguments.coreset_num_sizes, dtype=np.int32))
    else:
        Ms = np.unique(np.linspace(1, arguments.coreset_size_max, arguments.coreset_num_sizes, dtype=np.int32))

    #######################################
    #######################################
    ## Step 1: Define Model
    #######################################
    #######################################
    
    if arguments.model=="lr":
      import model_lr as model
    elif arguments.model=="poiss":
      import model_poiss as model

    #######################################
    #######################################
    ## Step 2: Load Dataset & run full MCMC / Laplace
    #######################################
    #######################################
    
    print('Loading dataset '+arguments.dataset)
    X, Y, Z, Zt, D = model.load_data('../data/'+arguments.dataset+'.npz')
    
    #NOTE: Sig0 is currently coded as identity in model_lr and model_pr (see log_prior).
    #so if you change Sig0 here things might break.
    #TODO: fix that...
    mu0 = np.zeros(Z.shape[1])
    Sig0 = np.eye(Z.shape[1])
    LSig0 = np.eye(Z.shape[1])


    print('Checking for cached full MCMC samples')
    mcmc_cache_filename = 'mcmc_cache/full_samples_'+arguments.model+'_'+arguments.dataset+'.npz'
    if os.path.exists(mcmc_cache_filename):
        print('Cache exists, loading')
        tmp__ = np.load(mcmc_cache_filename)
        full_samples = tmp__['samples']
        full_mcmc_time_per_itr = tmp__['t']
    else:
        print('Cache doesnt exist, running MCMC')
        #convert Y to Stan LR label format
        stanY = np.zeros(Y.shape[0])
        stanY[:] = Y
        stanY[stanY == -1] = 0
        sampler_data = {'x': X, 'y': stanY.astype(int), 'w': np.ones(X.shape[0]), 'd': X.shape[1], 'n': X.shape[0]}
        full_samples, t_full_mcmc = mcmc.run(sampler_data, arguments.mcmc_samples_full, arguments.model, model.stan_representation, arguments.trial)
        full_samples = full_samples['theta']
        #TODO right now *2 to account for burn; but this should all be specified via tunable arguments
        full_mcmc_time_per_itr = t_full_mcmc/(arguments.mcmc_samples_full*2)
        if not os.path.exists('mcmc_cache'):
            os.mkdir('mcmc_cache')
        np.savez(mcmc_cache_filename, samples = full_samples, t = full_mcmc_time_per_itr)
 
    #######################################
    #######################################
    ## Step 3: Calculate Likelihoods/Projectors
    #######################################
    #######################################

    #get Gaussian approximation to the true posterior
    print('Approximating true posterior')
    mup = full_samples.mean(axis=0)
    Sigp = np.cov(full_samples, rowvar=False)
    LSigp = np.linalg.cholesky(Sigp)
    LSigpInv = solve_triangular(LSigp, np.eye(LSigp.shape[0]), lower=True, overwrite_b=True, check_finite=False)

    #create tangent space for well-tuned Hilbert coreset alg
    print('Creating tuned projector for Hilbert coreset construction')
    muHat, LSigHat, LSigHatInv = get_laplace(np.ones(Z.shape[0]), Z, np.zeros(Z.shape[1]), model, diag = False)
    sampler_optimal = lambda n, w, pts : muHat + np.random.randn(n, muHat.shape[0]).dot(LSigHat.T)
    prj_optimal = bc.BlackBoxProjector(sampler_optimal, arguments.proj_dim, model.log_likelihood, model.grad_z_log_likelihood)
    
    #create tangent space for poorly-tuned Hilbert coreset alg
    print('Creating untuned projector for Hilbert coreset construction')
    Zhat = Z[np.random.randint(0, Z.shape[0], int(np.sqrt(Z.shape[0]))), :]
    muHat2, LSigHat2, LSigHat2Inv = get_laplace(np.ones(Zhat.shape[0]), Zhat, np.zeros(Zhat.shape[1]), model, diag = False)
    sampler_realistic = lambda n, w, pts : muHat2 + np.random.randn(n, muHat2.shape[0]).dot(LSigHat2.T)
    prj_realistic = bc.BlackBoxProjector(sampler_realistic, arguments.proj_dim, model.log_likelihood, model.grad_z_log_likelihood)

    print('Creating black box projector')
    def sampler_w(n, wts, pts):
        if wts is None or pts is None or pts.shape[0] == 0:
            muw = mu0
            LSigw = LSig0
        else:
            muw, LSigw, _ = get_laplace(wts, pts, np.zeros(Z.shape[1]), model, diag = False)
        return muw + np.random.randn(n, muw.shape[0]).dot(LSigw.T)
    prj_bb = bc.BlackBoxProjector(sampler_w, arguments.proj_dim, model.log_likelihood, model.grad_z_log_likelihood)
    
       
    #######################################
    #######################################
    ## Step 4: Construct Coreset
    #######################################
    #######################################
    
    print('Creating coreset construction objects')
    #create coreset construction objects
    sparsevi = bc.SparseVICoreset(Z, prj_bb, opt_itrs = arguments.opt_itrs, step_sched = eval(arguments.step_sched))
    giga_optimal = bc.HilbertCoreset(Z, prj_optimal)
    giga_realistic = bc.HilbertCoreset(Z,prj_realistic)
    unif = bc.UniformSamplingCoreset(Z)
    
    algs = {'SVI': sparsevi, 
            'GIGA-OPT': giga_optimal, 
            'GIGA-REAL': giga_realistic, 
            'US': unif}
    alg = algs[arguments.alg]

    cputs = np.zeros(Ms.shape[0])
    mcmc_time_per_itr = np.zeros(Ms.shape[0])
    csizes = np.zeros(Ms.shape[0])
    Fs = np.zeros(Ms.shape[0])
    rklw = np.zeros(Ms.shape[0])
    fklw = np.zeros(Ms.shape[0])
    mu_errs = np.zeros(Ms.shape[0])
    Sig_errs = np.zeros(Ms.shape[0])
    
    print('Running coreset construction / MCMC for ' + arguments.dataset + ' ' + arguments.alg + ' ' + str(arguments.trial))
    t_alg = 0.
    for m in range(Ms.shape[0]):
      print('M = ' + str(Ms[m]) + ': coreset construction, '+ arguments.alg + ' ' + arguments.dataset + ' ' + str(arguments.trial))
      #this runs alg up to a level of M; on the next iteration, it will continue from where it left off
      t0 = time.process_time()
      itrs = (Ms[m] if m == 0 else Ms[m] - Ms[m-1])
      alg.build(itrs)
      t_alg += time.process_time()-t0
      wts, pts, idcs = alg.get()
    
      print('M = ' + str(Ms[m]) + ': MCMC')
      # Use MCMC on the coreset, measure time taken 
      stanY = np.zeros(idcs.shape[0])
      stanY[:] = Y[idcs]
      stanY[stanY == -1] = 0
      sampler_data = {'x': X[idcs, :], 'y': stanY.astype(int), 'w': wts, 'd': X.shape[1], 'n': idcs.shape[0]}
      cst_samples, t_cst_mcmc = mcmc.run(sampler_data, arguments.mcmc_samples_coreset, arguments.model, model.stan_representation, arguments.trial)
      cst_samples = cst_samples['theta']
      #TODO see note above re: full mcmc sampling
      t_cst_mcmc_per_step = t_cst_mcmc/(arguments.mcmc_samples_coreset*2)

      print('M = ' + str(Ms[m]) + ': Approximating posterior with Gaussian')
      muw = cst_samples.mean(axis=0)
      Sigw = np.cov(cst_samples, rowvar=False)
      LSigw = np.linalg.cholesky(Sigw)
      LSigwInv = solve_triangular(LSigw, np.eye(LSigw.shape[0]), lower=True, overwrite_b=True, check_finite=False)
    
      print('M = ' + str(Ms[m]) + ': Computing metrics')
      cputs[m] = t_alg
      mcmc_time_per_itr[m] = t_cst_mcmc_per_step
      csizes[m] = (wts > 0).sum()
      gcs = np.array([ model.grad_th_log_joint(Z[idcs, :], full_samples[i, :], wts) for i in range(full_samples.shape[0]) ])
      gfs = np.array([ model.grad_th_log_joint(Z, full_samples[i, :], np.ones(Z.shape[0])) for i in range(full_samples.shape[0]) ])
      Fs[m] = (((gcs - gfs)**2).sum(axis=1)).mean()
      rklw[m] = KL(muw, Sigw, mup, LSigpInv.T.dot(LSigpInv))
      fklw[m] = KL(mup, Sigp, muw, LSigwInv.T.dot(LSigwInv))
      mu_errs[m] = np.sqrt(((mup - muw)**2).sum()) / np.sqrt((mup**2).sum())
      Sig_errs[m] = np.sqrt(((Sigp - Sigw)**2).sum()) / np.sqrt((Sigp**2).sum())

    results.save(arguments, csizes = csizes, Ms = Ms, cputs = cputs, Fs = Fs, full_mcmc_time_per_itr = full_mcmc_time_per_itr, mcmc_time_per_itr = mcmc_time_per_itr, rklw = rklw, fklw = fklw, mu_errs = mu_errs, Sig_errs = Sig_errs)
    


############################
############################
## Parse arguments
############################
############################
 
parser = argparse.ArgumentParser(description="Runs Hilbert coreset construction on a model and dataset")
subparsers = parser.add_subparsers(help='sub-command help')
run_subparser = subparsers.add_parser('run', help='Runs the main computational code')
run_subparser.set_defaults(func=run)
plot_subparser = subparsers.add_parser('plot', help='Plots the results')
plot_subparser.set_defaults(func=plot)


parser.add_argument('--model', type=str, choices=["lr","poiss"], help="The model to use.") #must be one of linear regression or poisson regression
parser.add_argument('--dataset', type=str, help="The name of the dataset") #examples: synth_lr, phishing, ds1, synth_poiss, biketrips, airportdelays, synth_poiss_large, biketrips_large, airportdelays_large
parser.add_argument('--alg', type=str, default='SVI', choices = ['SVI', 'GIGA-OPT', 'GIGA-REAL', 'US'], help="The algorithm to use for solving sparse non-negative least squares") #TODO: find way to make this help message autoupdate with new methods
parser.add_argument("--mcmc_samples_full", type=int, default=10000, help="number of MCMC samples to take for inference on the full dataset (also take this many warmup steps before sampling)")
parser.add_argument("--mcmc_samples_coreset", type=int, default=10000, help="number of MCMC samples to take for inference on the coreset (also take this many warmup steps before sampling)")
parser.add_argument("--proj_dim", type=int, default=500, help="The number of samples taken when discretizing log likelihoods for these experiments")

parser.add_argument('--coreset_size_max', type=int, default=1000, help="The maximum coreset size to evaluate")
parser.add_argument('--coreset_num_sizes', type=int, default=7, help="The number of coreset sizes to evaluate")
parser.add_argument('--coreset_size_spacing', type=str, choices=['log', 'linear'], default='log', help="The spacing of coreset sizes to test")
parser.add_argument('--opt_itrs', type=str, default = 100, help="Number of optimization iterations (for methods that use iterative weight refinement)")
parser.add_argument('--step_sched', type=str, default = "lambda i : 1./(1+i)", help="Optimization step schedule (for methods that use iterative weight refinement); entered as a python lambda expression surrounded by quotes")

parser.add_argument('--trial', type=int, help="The trial number - used to initialize random number generation (for replicability)")
parser.add_argument('--results_folder', type=str, default="results/", help="This script will save results in this folder")
parser.add_argument('--verbosity', type=str, default="error", choices=['error', 'warning', 'critical', 'info', 'debug'], help="The verbosity level.")

# plotting arguments
plot_subparser.add_argument('plot_x', type = str, help="The X axis of the plot")
plot_subparser.add_argument('plot_y', type = str, help="The Y axis of the plot")
plot_subparser.add_argument('--plot_title', type = str, help="The title of the plot")
plot_subparser.add_argument('--plot_x_label', type = str, help="The X axis label of the plot")
plot_subparser.add_argument('--plot_y_label', type = str, help="The Y axis label of the plot")
plot_subparser.add_argument('--plot_x_type', type=str, choices=["linear","log"], default = "log", help = "Specifies the scale for the X-axis")
plot_subparser.add_argument('--plot_y_type', type=str, choices=["linear","log"], default = "log", help = "Specifies the scale for the Y-axis.")
plot_subparser.add_argument('--plot_legend', type=str, help = "Specifies the variable to create a legend for.")
plot_subparser.add_argument('--plot_height', type=int, default=850, help = "Height of the plot's html canvas")
plot_subparser.add_argument('--plot_width', type=int, default=850, help = "Width of the plot's html canvas")
plot_subparser.add_argument('--plot_type', type=str, choices=['line', 'scatter'], default='scatter', help = "Type of plot to make")
plot_subparser.add_argument('--plot_fontsize', type=str, default='32pt', help = "Font size for the figure, e.g., 32pt")
plot_subparser.add_argument('--plot_toolbar', action='store_true', help = "Show the Bokeh toolbar")
plot_subparser.add_argument('--summarize', type=str, nargs='*', help = 'The command line arguments to ignore value of when matching to plot a subset of data. E.g. --summarize trial data_num will compute result statistics over both trial and number of datapoints')
plot_subparser.add_argument('--groupby', type=str, help = 'The command line argument group rows by before plotting. No groupby means plotting raw data; groupby will do percentile stats for all data with the same groupby value. E.g. --groupby Ms in a scatter plot will compute result statistics for fixed values of M, i.e., there will be one scatter point per value of M')

arguments = parser.parse_args()
arguments.func(arguments)


