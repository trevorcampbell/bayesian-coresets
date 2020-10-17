import hashlib
import os
import json
import pickle as pk
import pandas as pd
import numpy as np

def hash_namespace(ns):
    return hashlib.md5(json.dumps(vars(ns), sort_keys=True).encode('utf-8')).hexdigest()

def check_exists(arguments, results_folder = 'results/'):
    arg_hash = hash_namespace(arguments)
    if os.path.exists(os.path.join(results_folder, arg_hash+'.csv')):
        return True
    return False

def load_matching(match_dict, results_folder = 'results/'):
    resfiles = os.listdir(results_folder)
    df = None
    for resfile in resfiles:
        #load the results file
        resdf = pd.read_csv(os.path.join(results_folder, resfile))
        #extract the matching rows
        resdf = resdf.loc[(resdf[list(match_dict)] == pd.Series(match_dict)).all(axis=1)]
        if resdf.shape[0] > 0:
            #if df hasn't been initialized yet, just use resdf; otherwise, append
            if df is None:
                df = resdf
            else:
                df = df.append(resdf)
    return df

def save(arguments, results_folder = 'results/', **kwargs):
    #create results filenames/argument hashes/dict
    arg_hash = hash_namespace(arguments)
    argd = vars(arguments)

    #make the results folder if it doesn't exist
    if not os.path.exists(results_folder):
      os.mkdir(results_folder)

    #add the kwargs to the dict
    for kw, arr in kwargs.items():
        argd[kw] = arr

    #save the df
    df = pd.DataFrame(argd)
    df.to_csv(os.path.join(results_folder, arg_hash+'.csv'), index=False)
    print('Saved result to ' + str(os.path.join(results_folder, arg_hash+'.csv')))


