import hashlib
import os
import json
import pickle as pk
import pandas as pd
import numpy as np

def hash_namespace(ns):
    return hashlib.md5(json.dumps(vars(ns), sort_keys=True).encode('utf-8')).hexdigest()

def check_exists(arguments, results_folder = 'results/', results_filename = 'results.csv'):
    arg_hash = hash_namespace(arguments)
    if os.path.exists(os.path.join(results_folder, arg_hash+'.npz')):
        return True
    return False

def save_result(arguments, results_folder = 'results/', results_filename = 'results.csv', **kwargs):
    #create results filenames/argument hashes/dict
    arg_hash = hash_namespace(arguments)
    argd = vars(arguments)
    argd['hash'] = arg_hash

    #save the results
    if not os.path.exists(results_folder):
      os.mkdir(results_folder)
    np.savez_compressed(os.path.join(results_folder, arg_hash+'.npz'), **kwargs)

    #append the hash/args to the results log
    resfn = os.path.join(results_folder, results_filename)
    new_row = pd.DataFrame(argd, index=[0])
    if not os.path.exists(resfn):
        args_df = new_row
    else:
        args_df = pd.read_csv(resfn)
        idxer = args_df.hash.isin(new_row.hash)
        if not idxer.any():
           args_df = args_df.append(new_row)

    args_df.to_csv(resfn, index=False)


