import numpy as np
import scipy.linalg as sl
import pickle as pk
import os, sys
import argparse
import time
#make it so we can import models/etc from parent folder
import bayesiancoresets as bc
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import model_gaussian as gaussian
import results
import plotting


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
    ## Step 0: Setup
    #######################################
    #######################################

    np.random.seed(arguments.trial)
    bc.util.set_verbosity(arguments.verbosity)
    
    if arguments.coreset_size_spacing == 'log':
        Ms = np.unique(np.logspace(0., np.log10(arguments.coreset_size_max), arguments.coreset_num_sizes, dtype=np.int32))
    else:
        Ms = np.unique(np.linspace(1, arguments.coreset_size_max, arguments.coreset_num_sizes, dtype=np.int32))

    #make sure the first size to record is 0
    if Ms[0] != 0:
        Ms = np.hstack((0, Ms))

    #######################################
    #######################################
    ## Step 1: Generate a Synthetic Dataset
    #######################################
    #######################################
    
    #change these to change the prior / likelihood
    mu0 = np.zeros(arguments.data_dim)
    Sig0 = np.eye(arguments.data_dim)
    Sig = np.eye(arguments.data_dim)

    #these are computed
    Sig0inv = np.linalg.inv(Sig0)
    Siginv = np.linalg.inv(Sig)
    LSigInv = np.linalg.cholesky(Siginv) #Siginv = LL^T, L Lower tri
    USig = sl.solve_triangular(LSigInv, np.eye(LSigInv.shape[0]), lower=True, overwrite_b=True, check_finite=False).T # Sig = UU^T, U upper tri
    th = np.ones(arguments.data_dim)
    logdetSig = np.linalg.slogdet(Sig)[1]
    
    #######################################
    #######################################
    ## Step 2: Calculate Likelihoods/Projectors
    #######################################
    #######################################

    print('Computing true posterior')
    x = np.random.multivariate_normal(th, Sig, arguments.data_num)
    mup, USigp, LSigpInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
    Sigp = USigp.dot(USigp.T)
    SigpInv = LSigpInv.dot(LSigpInv.T)
    
    #create the log_likelihood function
    print('Creating log-likelihood function')
    log_likelihood = lambda x, th : gaussian.log_likelihood(x, th, Siginv, logdetSig)
    
    print('Creating gradient log-likelihood function')
    grad_log_likelihood = lambda x, th : gaussian.gradx_log_likelihood(x, th, Siginv)
    
    print('Creating tuned projector for Hilbert coreset construction')
    #create the sampler for the "optimally-tuned" Hilbert coreset
    sampler_optimal = lambda n, w, pts : mup + np.random.randn(n, mup.shape[0]).dot(USigp.T)
    prj_optimal = bc.BlackBoxProjector(sampler_optimal, arguments.proj_dim, log_likelihood, grad_log_likelihood)
    
    print('Creating untuned projector for Hilbert coreset construction')
    #create the sampler for the "realistically-tuned" Hilbert coreset
    xhat = x[np.random.randint(0, x.shape[0], int(np.sqrt(x.shape[0]))), :]
    muhat, USigHat, LSigHatInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, xhat, np.ones(xhat.shape[0]))
    sampler_realistic = lambda n, w, pts : muhat + np.random.randn(n, muhat.shape[0]).dot(USigHat.T)
    prj_realistic = bc.BlackBoxProjector(sampler_realistic, arguments.proj_dim, log_likelihood, grad_log_likelihood)

    print('Creating black box projector')
    def sampler_w(n, wts, pts):
        if wts is None or pts is None or pts.shape[0] == 0:
          wts = np.zeros(1)
          pts = np.zeros((1, mu0.shape[0]))
        muw, USigw, _ = gaussian.weighted_post(mu0, Sig0inv, Siginv, pts, wts)
        return muw + np.random.randn(n, muw.shape[0]).dot(USigw.T)
    prj_bb = bc.BlackBoxProjector(sampler_w, arguments.proj_dim, log_likelihood, grad_log_likelihood)
    
    print('Creating exact projectors')
    #TODO need to fix all the transposes in this...
    class GaussianProjector(bc.Projector):
      def project(self, pts, grad=False):
        nu = (pts - self.muw).dot(LSigInv)
        PsiL = LSigInv.T.dot(self.USigw)
        Psi = PsiL.dot(PsiL.T)
        nu = np.hstack((nu.dot(PsiL), np.sqrt(0.5*np.trace(np.dot(Psi.T, Psi)))*np.ones(nu.shape[0])[:,np.newaxis]))
        nu *= np.sqrt(nu.shape[1])
        if not grad:
          return nu
        else:
          gnu = np.hstack((SigLInv.dot(PsiL), np.zeros(pts.shape[1])[:,np.newaxis])).T
          gnu = np.tile(gnu, (pts.shape[0], 1, 1))
          gnu *= np.sqrt(gnu.shape[1])
          return nu, gnu
      def update(self, wts = None, pts = None):
        if wts is None or pts is None or pts.shape[0] == 0:
          wts = np.zeros(1)
          pts = np.zeros((1, mu0.shape[0]))
        self.muw, self.USigw, self.LSigwInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, pts, wts)

    prj_optimal_exact = GaussianProjector()
    prj_optimal_exact.update(np.ones(x.shape[0]), x)
    prj_realistic_exact = GaussianProjector()
    prj_realistic_exact.update(np.ones(xhat.shape[0]), xhat)
       
    #######################################
    #######################################
    ## Step 3: Construct Coreset
    #######################################
    #######################################
    
    ##############################
    print('Creating coreset construction objects')
    #create coreset construction objects
    sparsevi_exact = bc.SparseVICoreset(x, GaussianProjector(), opt_itrs = arguments.opt_itrs, step_sched = eval(arguments.step_sched))
    sparsevi = bc.SparseVICoreset(x, prj_bb, opt_itrs = arguments.opt_itrs, step_sched = eval(arguments.step_sched))
    giga_optimal = bc.HilbertCoreset(x, prj_optimal)
    giga_optimal_exact = bc.HilbertCoreset(x,prj_optimal_exact)
    giga_realistic = bc.HilbertCoreset(x,prj_realistic)
    giga_realistic_exact = bc.HilbertCoreset(x,prj_realistic_exact)
    unif = bc.UniformSamplingCoreset(x)
    
    algs = {'SVI-EXACT': sparsevi_exact,
            'SVI': sparsevi, 
            'GIGA-OPT': giga_optimal, 
            'GIGA-OPT-EXACT': giga_optimal_exact, 
            'GIGA-REAL': giga_realistic, 
            'GIGA-REAL-EXACT': giga_realistic_exact, 
            'US': unif}
    alg = algs[arguments.alg]
    
    print('Building coreset')
    w = []
    p = []
    cputs = np.zeros(Ms.shape[0])
    t_build = 0
    for m in range(Ms.shape[0]):
      print('M = ' + str(Ms[m]) + ': coreset construction, '+ arguments.alg + ' ' + str(arguments.trial))
      t0 = time.process_time()
      itrs = (Ms[m] if m == 0 else Ms[m] - Ms[m-1])
      alg.build(itrs)
      t_build += time.process_time()-t0
      wts, pts, idcs = alg.get()
    
      #store weights/pts/runtime
      w.append(wts)
      p.append(pts)
      cputs[m] = t_build
    
    ##############################
    ##############################
    ## Step 4: Evaluate coreset
    ##############################
    ##############################
    
    # computing kld and saving results
    muw = np.zeros((Ms.shape[0], mu0.shape[0]))
    Sigw = np.zeros((Ms.shape[0], mu0.shape[0], mu0.shape[0]))
    rklw = np.zeros(Ms.shape[0])
    fklw = np.zeros(Ms.shape[0])
    csizes = np.zeros(Ms.shape[0])
    mu_errs = np.zeros(Ms.shape[0])
    Sig_errs = np.zeros(Ms.shape[0])
    for m in range(Ms.shape[0]):
      csizes[m] = (w[m] > 0).sum()
      muw[m, :], USigw, LSigwInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, p[m], w[m])
      Sigw[m, :, :] = USigw.dot(USigw.T)
      rklw[m] = gaussian.KL(muw[m,:], Sigw[m,:,:], mup, SigpInv)
      fklw[m] = gaussian.KL(mup, Sigp, muw[m,:], LSigwInv.dot(LSigwInv.T))
      mu_errs[m] = np.sqrt(((mup - muw[m,:])**2).sum()) / np.sqrt((mup**2).sum())
      Sig_errs[m] = np.sqrt(((Sigp - Sigw[m,:,:])**2).sum()) / np.sqrt((Sigp**2).sum())

    results.save(arguments, csizes = csizes, Ms = Ms, cputs = cputs, rklw = rklw, fklw = fklw, mu_errs = mu_errs, Sig_errs = Sig_errs)

    #also save muw/Sigw/etc for plotting coreset visualizations
    f = open('results/coreset_data.pk', 'wb')
    res = (x, mu0, Sig0, Sig, mup, Sigp, w, p, muw, Sigw)
    pk.dump(res, f)
    f.close()

############################
############################
## Parse arguments
############################
############################
 
parser = argparse.ArgumentParser(description="Runs Riemannian linear regression (employing coreset contruction) on the specified dataset")
subparsers = parser.add_subparsers(help='sub-command help')
run_subparser = subparsers.add_parser('run', help='Runs the main computational code')
run_subparser.set_defaults(func=run)
plot_subparser = subparsers.add_parser('plot', help='Plots the results')
plot_subparser.set_defaults(func=plot)

parser.add_argument('--data_num', type=int, default='1000', help='Dataset size/number of examples')
parser.add_argument('--data_dim', type=int, default = '200', help="The dimension of the multivariate normal distribution to use for this experiment")
parser.add_argument('--alg', type=str, default='SVI', choices = ['SVI', 'SVI-EXACT', 'GIGA-OPT', 'GIGA-OPT-EXACT', 'GIGA-REAL', 'GIGA-REAL-EXACT', 'US'], help="The name of the coreset construction algorithm to use")
parser.add_argument("--proj_dim", type=int, default=100, help="The number of samples taken when discretizing log likelihoods for these experiments")

parser.add_argument('--coreset_size_max', type=int, default=200, help="The maximum coreset size to evaluate")
parser.add_argument('--coreset_num_sizes', type=int, default=7, help="The number of coreset sizes to evaluate")
parser.add_argument('--coreset_size_spacing', type=str, choices=['log', 'linear'], default='log', help="The spacing of coreset sizes to test")

parser.add_argument('--opt_itrs', type=int, default = 100, help="Number of optimization iterations (for methods that use iterative weight refinement)")
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


