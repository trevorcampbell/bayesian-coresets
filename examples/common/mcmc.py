import numpy as np
import pystan
import os
import pickle as pk
import time
import hashlib

def build_model(model_caching_folder, model_name, model_code, use_weighted_coresets = False, code_caching_folder = 'stanCppCode/', verbose_compiling = False):
  if model_caching_folder:
    weights_tag = "weighted_coreset_version_" if use_weighted_coresets else ""
    cachingSpot = os.path.join(model_caching_folder, model_name + "_" + weights_tag + hashlib.sha1(model_code.encode('utf-8')).hexdigest())
  if model_caching_folder and os.path.exists(cachingSpot):
    f = open(cachingSpot,'rb')
    sm = pk.load(f) 
    f.close()
  else: 
    print('STAN: building model')
    if use_weighted_coresets:
      print('Altering cpp code used by stan to allow weighted data')
      stanc_ret = pystan.stanc(model_code=model_code)
      stanc_ret['cppcode'] = load_modified_cpp_code(code_caching_folder, model_name, model_code)
      sm = pystan.StanModel(stanc_ret=stanc_ret, verbose=verbose_compiling)
    else: 
      sm = pystan.StanModel(model_code=model_code)

    if model_caching_folder: 
      f = open(cachingSpot,'wb')
      pk.dump(sm,f)
      f.close()

  return sm

def sampler(dnm, X, Y, N_samples, model_name, model_code, weights = None, sample_caching_folder = None, model_caching_folder = '../common/models', code_caching_folder = '../common/stanCppCode/', chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True, verbose_compiling = False, seed = None):
  if sample_caching_folder and not os.path.exists(sample_caching_folder):
    os.mkdir(sample_caching_folder)
  if sample_caching_folder: 
    sample_caching_prefix = sample_caching_folder+dnm+("_unseeded" if seed is None else "_seed=" + str(seed))+'model_name='+model_name+'_N_samples='+str(N_samples)+'_weights='+str(weights)+'_chains='+str(chains)+'control_hash_sha1='+hashlib.sha1(str(control).encode('utf-8')).hexdigest()+'_model_code_hash_sha1='+hashlib.sha1(model_code.encode('utf-8')).hexdigest()  
  if sample_caching_folder and os.path.exists(sample_caching_prefix + '_samples.npy'):
    print("Using cached samples for " + dnm)
    return np.load(sample_caching_prefix + '_samples.npy')
  else:
    if model_caching_folder and not os.path.exists(model_caching_folder):
      os.mkdir(model_caching_folder)
    print('No MCMC samples found -- running STAN')
    print('STAN: loading data')
    Y[Y == -1] = 0 #convert to Stan LR label style if necessary

    sampler_data = {'x': X, 'y': Y.astype(int), 'w': weights if weights is not None else [], 'd': X.shape[1], 'n': X.shape[0]}

    print('STAN: building/loading model')
    sm = build_model(model_caching_folder, model_name, model_code, use_weighted_coresets = weights is not None, code_caching_folder=code_caching_folder, verbose_compiling=verbose_compiling)

    print('STAN: sampling posterior: ' + dnm)
    t0 = time.process_time()
    thd = sampler_data['d']+1
    #call sampling with N_samples actual iterations, and some number of burn iterations
    try:
      if seed is not None:
        fit = sm.sampling(data=sampler_data, iter=N_samples*2, chains=chains, control=control, verbose=verbose, seed=seed)
      else:
        fit = sm.sampling(data=sampler_data, iter=N_samples*2, chains=chains, control=control, verbose=verbose)
    except:
      print("error encountered in sampling - likely the specified dataset is not compatible with the specified model")
      raise EnvironmentError("error encountered in sampling - likely the specified dataset is not compatible with the specified model")
    samples = fit.extract(permuted=False)[:, 0, :thd]
    if sample_caching_folder:
      np.save(os.path.join(sample_caching_prefix+'_samples.npy'), samples) 
      tf = time.process_time()
      np.save(os.path.join(sample_caching_prefix+'_mcmc_time.npy'), tf-t0)
    return samples

#Takes in the name of and code for a statistical model that allows stan to run MCMC, and returns cpp code that stan can use to 
#perform MCMC on a coreset.  
def load_modified_cpp_code(code_folder, model_name, model_code):
  codeHash = hashlib.sha1(model_code.encode('utf-8')).hexdigest()
  file_to_find = model_name + "_weighted_coreset_version_" + codeHash + ".cpp"
  path_to_file_to_find = os.path.join(code_folder, file_to_find)
  if os.path.exists(path_to_file_to_find):
    f = open(path_to_file_to_find,'r')
    modified_code = f.read()
    f.close()
    return modified_code
  else: 
    if not os.path.exists(code_folder):
      os.mkdir(code_folder)

    sm = pystan.StanModel(model_code=model_code)
    file = open(path_to_file_to_find, "w")
    file.write("Remove this line once you have modified the code below to handle weighted coresets. See ReadMe for more information.\n")
    file.write(sm.model_cppcode)      
    file.close()
    backup_of_file_to_find = os.path.join(code_folder, 'unmodified_backup_of_' + model_name + "_weighted_coreset_version_" + codeHash + '_with_Cpp_code_hash_' + hashlib.sha1(sm.model_cppcode.encode('utf-8')).hexdigest() + '.cpp')
    file = open(backup_of_file_to_find, "w")
    file.write(sm.model_cppcode) 
    file.close()
    raise EnvironmentError("No modified code to handle weighted data present - unable to use stan for MCMC sampling. Please modify the file "+str(path_to_file_to_find)+" to handle weighted data. See the ReadMe for more information.")


# TODO test weighted CPP using randomly chosen integer weights
#arguments = parser.parse_args()
#dnm = arguments.dnm
#ID = arguments.ID
#model = arguments.model
#numIdcs = arguments.numIdcs
#mcmc_samples = arguments.mcmc_samples
#verbose_compiling = arguments.verbose_compiling
#
##lrdnms = ['synth_lr', 'phishing', 'ds1', 'synth_lr_large', 'phishing_large', 'ds1_large']
##prdnms = ['synth_poiss', 'biketrips', 'airportdelays', 'synth_poiss_large', 'biketrips_large', 'airportdelays_large']
#if model=="lr":
#  from model_lr import *
#elif model=="poiss":
#  from model_poiss import *
#
#np.random.seed(ID)
#
#print('Loading dataset '+dnm)
#X,Y,Z, Zt, D = load_data('../data/'+dnm+'.npz')
#print("N, d: "+str(X.shape))
#
#idcs = np.random.choice(X.shape[0],numIdcs, replace=True) #random.choice is a bit slower without replacement, but may still be worth trying
#weights = np.random.randint(1,10,numIdcs)
#curX = X[idcs, :]
#curY = Y[idcs]
##run the weighted version of sampler
## (we want to make sure we use a model based on the current cpp code for weighted coresets, so we don't use any cached models)
#samples_using_code_for_weights = mcmc.sampler(dnm, curX, curY, mcmc_samples, stan_representation, weights=weights, model_caching_folder = None, verbose_compiling = verbose_compiling, seed = ID) 
## build the modified dataset so that we can run the unweighted version of sampler
#for i in range(len(idcs)):
#    toAdd = np.ones(weights[i]-1,dtype=int) * (i) #list i a number of times equal to the weight of the ith index
#    curX = np.append(curX, curX[toAdd], axis=0)
#    curY = np.append(curY, curY[toAdd])
#samples_using_standard_stan_code = mcmc.sampler(dnm, curX, curY, mcmc_samples, stan_representation, seed = ID)
#
#relative_err = np.abs(np.linalg.norm(samples_using_code_for_weights - samples_using_standard_stan_code, axis=1)/np.linalg.norm(samples_using_standard_stan_code,axis=1))
#print("largest relative error between samples using code for weights and samples using standard stan code (should be 0): ")
#print(np.max(relative_err))
#print("average relative error")
#print(np.mean(relative_err))
#print("median relative error")
#print(np.quantile(relative_err,.5))
#print("first quartile relative error")
#print(np.quantile(relative_err,.25))
#print("third quartile relative error")
#print(np.quantile(relative_err,.75))

