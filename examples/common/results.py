import hashlib
import os
import json
import pickle as pk
import pandas as pd
import numpy as np

def hash_namespace(ns):
    nsdict = vars(ns)
    nsdict.pop('func', None) #can't hash function objects
    return hashlib.md5(json.dumps(nsdict, sort_keys=True).encode('utf-8')).hexdigest()

def check_exists(arguments, results_folder = 'results/'):
    arg_hash = hash_namespace(arguments)
    if os.path.exists(os.path.join(results_folder, arg_hash+'.csv')):
        return True
    return False

def load_matching(match_dict, results_folder = 'results/', log_file = 'manifest.csv'):
    resfiles = [fn for fn in os.listdir(results_folder) if fn != log_file and fn[-4:] == '.csv']
    df = None
    for resfile in resfiles:
        #load the results file
        resdf = pd.read_csv(os.path.join(results_folder, resfile))
        #get the intersection of column names and argnames
        cols_to_match = list(set(resdf.columns.tolist()).intersection(set(match_dict)))
        #extract the matching rows
        resdf = resdf.loc[(resdf[cols_to_match] == pd.Series({m:v for m, v in match_dict.items() if m in cols_to_match})).all(axis=1)]
        if resdf.shape[0] > 0:
            #if df hasn't been initialized yet, just use resdf; otherwise, append
            if df is None:
                df = resdf
            else:
                df = df.append(resdf)

    return df

def save(arguments, results_folder = 'results/', log_file = 'manifest.csv', **kwargs):
    #create results filenames/argument hashes/dict
    arg_hash = hash_namespace(arguments)
    argd = vars(arguments)

    manifest_line = arg_hash+': '+ str(argd) + '\n'

    #make the results folder if it doesn't exist
    if not os.path.exists(results_folder):
      os.mkdir(results_folder)

    #add the kwargs to the dict
    for kw, arr in kwargs.items():
        argd[kw] = arr

    #save the df
    df = pd.DataFrame(argd)
    df.to_csv(os.path.join(results_folder, arg_hash+'.csv'), index=False)

    #append to the manifest
    with open(os.path.join(results_folder, log_file), 'a') as f:
        f.write(manifest_line)




