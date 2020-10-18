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

    #######################################
    #######################################
    ## Step 1: Generate a Synthetic Dataset
    #######################################
    #######################################
    
    mu0 = np.zeros(d)
    Sig0 = np.eye(d)
    Sig = np.eye(d)
    SigL = np.linalg.cholesky(Sig)
    th = np.ones(d)
    Sig0inv = np.linalg.inv(Sig0)
    Siginv = np.linalg.inv(Sig)
    SigLInv = np.linalg.inv(SigL)
    logdetSig = np.linalg.slogdet(Sig)[1]
    
    print('Computing true posterior')
    x = np.random.multivariate_normal(th, Sig, arguments.data_num)
    mup, LSigp, LSigpInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
    Sigp = LSigp.dot(LSigp.T)
    SigpInv = LSigpInv.T.dot(LSigpInv)
    
    #######################################
    #######################################
    ## Step 2: Calculate Likelihoods/Projectors
    #######################################
    #######################################
    
    #create the log_likelihood function
    print('Creating log-likelihood function')
    log_likelihood = lambda x, th : gaussian.log_likelihood(x, th, Siginv, logdetSig)
    
    print('Creating gradient log-likelihood function')
    grad_log_likelihood = lambda x, th : gaussian.gradx_log_likelihood(x, th, Siginv)
    
    print('Creating tuned projector for Hilbert coreset construction')
    #create the sampler for the "optimally-tuned" Hilbert coreset
    sampler_optimal = lambda n, w, pts : mup + np.random.randn(n, mup.shape[0]).dot(LSigp.T)
    prj_optimal = bc.BlackBoxProjector(sampler_optimal, arguments.proj_dim, log_likelihood, grad_log_likelihood)
    
    print('Creating untuned projector for Hilbert coreset construction')
    #create the sampler for the "realistically-tuned" Hilbert coreset
    U = np.random.rand()
    muhat = U*mup + (1.-U)*mu0
    Sighat = U*Sigp + (1.-U)*Sig0
    #now corrupt the smoothed pihat
    muhat += pihat_noise*np.sqrt((muhat**2).sum())*np.random.randn(muhat.shape[0])
    Sighat *= np.exp(-2*pihat_noise*np.fabs(np.random.randn()))
    LSighat = np.linalg.cholesky(Sighat)
    
    sampler_realistic = lambda n, w, pts : mup + np.random.randn(n, mup.shape[0]).dot(LSighat.T)
    prj_realistic = bc.BlackBoxProjector(sampler_realistic, arguments.proj_dim, log_likelihood, grad_log_likelihood)
    
    print('Creating exact projector')
    #exact (gradient) log likelihood projection
    class GaussianProjector(bc.Projector):
      def project(self, pts, grad=False):
        nu = (pts - self.muw).dot(SigLInv.T)
        PsiL = SigLInv.dot(self.LSigw)
        Psi = PsiL.dot(PsiL.T)
        nu = np.hstack((nu.dot(PsiL), np.sqrt(0.5*np.trace(np.dot(Psi.T, Psi)))*np.ones(nu.shape[0])[:,np.newaxis]))
        nu *= np.sqrt(nu.shape[1])
        if not grad:
          return nu
        else:
          gnu = np.hstack((SigLInv.T.dot(PsiL), np.zeros(pts.shape[1])[:,np.newaxis])).T
          gnu = np.tile(gnu, (pts.shape[0], 1, 1))
          gnu *= np.sqrt(gnu.shape[1])
          return nu, gnu
      def update(self, wts = None, pts = None):
        if wts is None or pts is None or pts.shape[0] == 0:
          wts = np.zeros(1)
          pts = np.zeros((1, mu0.shape[0]))
        self.muw, self.LSigw, self.LSigwInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, pts, wts)
    
    prj_exact_optimal = GaussianProjector()
    prj_exact_optimal.update(np.ones(x.shape[0]), x)
    rlst_idcs = np.arange(x.shape[0])
    np.random.shuffle(rlst_idcs)
    rlst_idcs = rlst_idcs[:int(0.1*rlst_idcs.shape[0])]
    rlst_w = np.zeros(x.shape[0])
    rlst_w[rlst_idcs] = 2.*x.shape[0]/rlst_idcs.shape[0]*np.random.rand(rlst_idcs.shape[0])
    prj_exact_realistic = GaussianProjector()
    prj_exact_realistic.update(2.*np.random.rand(x.shape[0]), x)
    
    print("Creating approximate projector for fairer evaluation of SVI-like approaches")
    #approximate log likelihood projection (TODO: add gradient)
    class ApproximateGaussianProjector(bc.Projector):
      def project(self, pts, grad=False):
        #TODO: find error in this approach
        #take the likelihood of our pts according to our samples, using the log likelihood function established earlier
        ll = log_likelihood(pts, self.samples)
        return ll - ll.mean(axis= 1)[:, np.newaxis]
      def update(self, wts = None, pts = None):
        if wts is None or pts is None or pts.shape[0] == 0:
          wts = np.zeros(1)
          pts = np.zeros((1, mu0.shape[0]))
        # TODO: find error in this reasoning
        #[same as exact case] calculate the mean and (cholesky decomposed) variance of pi hat, based on the weights we have and our original Sig0inv, Siginv
        self.muw, self.LSigw, self.LSigwInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, pts, wts)
        #use the same samples for all projections after a given update (so that we can compare projected coreset point log likelihoods with projected data point log likelihoods across the same set of samples)
        self.samples = np.random.multivariate_normal(self.muw, self.LSigw @ self.LSigw.T, proj_dim) #There may be a significant difference between using sigw.T@sigw vs sigw @ sigw.T, but some empirical tests on small, non-diagonal examples encourage the former for sigw and the latter for sigwinv (and for this case, We may just dealing with diagonal matrices, where both choices are equivalent)
    
    #list of what's been tried already (and documented to be unsuccessful - the KL divergence of approximate projection SVI approaches flattens by ~iteration 30) :
    # sampling using self.samples = np.random.multivariate_normal(self.muw, self.LSigw.T @ self.LSigw, proj_dim), with projection...
    #    1) using the log_likelihood function defined earlier
    #    2) using pi hat's variance for the log liklihood: gaussian.log_likelihood(pts, self.samples, self.LSigwInv@self.LsigWinv.T, self.Ldet) where self.Ldet = np.sum(np.log(np.diag(self.LSigw)))
    
    
    prj_exact_approx = ApproximateGaussianProjector()
    
    #######################################
    #######################################
    ## Step 3: Construct Coreset
    #######################################
    #######################################
    
    ##############################
    print('Creating coreset construction objects')
    #create coreset construction objects
    sparsevi_exact = bc.SparseVICoreset(x, GaussianProjector(), opt_itrs = SVI_opt_itrs, step_sched = SVI_step_sched)
    sparsevi = bc.SparseVICoreset(x, ApproximateGaussianProjector(), opt_itrs = SVI_opt_itrs, step_sched = SVI_step_sched)
    giga_optimal = bc.HilbertCoreset(x, prj_optimal)
    giga_optimal_exact = bc.HilbertCoreset(x,prj_exact_optimal)
    giga_realistic = bc.HilbertCoreset(x,prj_realistic)
    giga_realistic_exact = bc.HilbertCoreset(x,prj_exact_realistic)
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
    w = [np.array([0.])]
    p = [np.zeros((1, x.shape[1]))]
    cputs = [0]
    
    t0 = time.process_time()
    t_build = 0
    
    for m in range(1, M+1):
      print('trial: ' + str(tr) +' alg: ' + nm + ' ' + str(m) +'/'+str(M))
      if nm == "HOPS_full_scaling" or nm == "HOPS_full_scaling_exact":
        alg.reset()
        t_prebuild = time.process_time()
        alg.build(m)
        t_build = time.process_time() - t_prebuild
      else:
        t_prebuild = time.process_time()
        alg.build(1)
        t_build += time.process_time() - t_prebuild
      if optimizing and nm != "SVI" and nm != "SVIEXACT": #(SVI approaches don't behave differently in the optimizing case, because their optimization call is the same as their reweight call)
        #if algorithm is in the HOPS family, we should optimize using the algorithm's own optimize() method (making sure that we don't alter the state of the coreset for the next iteration)
        if (nm=="HOPSEXACT" or nm=="HOPS" or nm == "HOPS_full_scaling" or nm == "HOPS_full_scaling_exact"):
          print("simulating results if we optimize after this iteration")
          algCopy = copy.deepcopy(alg)
          t_pre_opt = time.process_time()
          algCopy.optimize()
          t_opt = time.process_time() - t_pre_opt
          wts, pts, _ = algCopy.get()
        else:
          #otherwise, create a HOPS algorithm, copy over the coreset information, and run optimize with that algorithm
          print("simulating results if the coreset were optimized with the HOPS optimization function")
          polishingAlg = algs["HOPS"]
          polishingAlg.wts = alg.wts
          polishingAlg.pts = alg.pts
          polishingAlg.idcs = alg.idcs
          t_pre_opt = time.process_time()
          polishingAlg.optimize()
          t_opt = time.process_time() - t_pre_opt
          wts,pts, _ = polishingAlg.get()
      else:
        wts, pts, _ = alg.get()
        t_opt = 0
    
      #store weights/pts/runtime
      w.append(wts)
      p.append(pts)
      cputs.append(t_build + t_opt)
    
    ##############################
    ##############################
    ## Step 4: Evaluate coreset
    ##############################
    ##############################
    
    # computing kld and saving results
    muw = np.zeros((M+1, mu0.shape[0]))
    Sigw = np.zeros((M+1,mu0.shape[0], mu0.shape[0]))
    rklw = np.zeros(M+1)
    fklw = np.zeros(M+1)
    for m in range(M+1):
      muw[m, :], LSigw, LSigwInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, p[m], w[m])
      Sigw[m, :, :] = LSigw.dot(LSigw.T)
      rklw[m] = gaussian.KL(muw[m,:], Sigw[m,:,:], mup, SigpInv)
      fklw[m] = gaussian.KL(mup, Sigp, muw[m,:], LSigwInv.T.dot(LSigwInv))
    
    if not os.path.exists('results/'):
      os.mkdir('results')
    #make hash of step schedule so it can be encoded in the file name:
    SVI_step_sched_hash_sha1 = hashlib.sha1(arguments.SVI_step_sched.encode('utf-8')).hexdigest()
    f = open('results/'+nm+'_tr='+str(tr)+'_N='+str(N)+'_d='+str(d)+'_proj_dim='+str(proj_dim)+'_optimizing='+str(optimizing)+'_SVI_opt_itrs='+str(SVI_opt_itrs)+'_'+'SVI_step_sched_hash_sha1='+SVI_step_sched_hash_sha1+'_pihat_noise='+str(pihat_noise)+'.pk', 'wb')
    res = (x, mu0, Sig0, Sig, mup, Sigp, w, p, muw, Sigw, rklw, fklw, cputs, tr, N, d, proj_dim, optimizing, SVI_opt_itrs, arguments.SVI_step_sched, pihat_noise)
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
parser.add_argument('--alg', type=str, default='SVI', choices = ['SVI-EXACT', 'GIGA-OPT', 'GIGA-REAL', 'US'], help="The name of the coreset construction algorithm to use")
parser.add_argument("--proj_dim", type=int, default=500, help="The number of samples taken when discretizing log likelihoods for these experiments")

parser.add_argument('--coreset_size_max', type=int, default=1000, help="The maximum coreset size to evaluate")
parser.add_argument('--coreset_num_sizes', type=int, default=7, help="The number of coreset sizes to evaluate")
parser.add_argument('--coreset_size_spacing', type=str, choices=['log', 'linear'], default='log', help="The spacing of coreset sizes to test")

parser.add_argument('--svi_step_sched', type=str, default = "lambda i : 1./(1+i)", help="Step schedule (tuning rate) for SVI, entered as a lambda expression in quotation marks.")
parser.add_argument('--gigar_pihat_noise', type=float, default=.75, help = "How much noise to introduce to the smoothed pi-hat in GIGAR")

parser.add_argument('--trial', type=int, help="The trial number - used to initialize random number generation (for replicability)")
parser.add_argument('--results_folder', type=str, default="results/", help="This script will save results in this folder")
parser.add_argument('--verbosity', type=str, default="error", choices=['error', 'warning', 'critical', 'info', 'debug'], help="The verbosity level.")

# plotting arguments
plot_subparser.add_argument('plot_x', type = str, help="The X axis of the plot")
plot_subparser.add_argument('plot_y', type = str, help="The Y axis of the plot")
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


