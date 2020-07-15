from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import yaml
import numpy as np
import subprocess
import time
from collections import OrderedDict
from itertools import product

SLEEP = 10
CTR_LIM = 12

def num_procs_open(procs):
    k = 0
    for p in procs:
        k += (p.poll() is None)
    return k

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Launch IEF experiments')
    parser.add_argument('--script', type=str, default='main_trainer.py',
                        help='Path to the experiment run script.')
    parser.add_argument('--config', type=str,
                        help='Path to the experiment config file.')
    parser.add_argument('--n_models', type=int, default=0,
                        help='Expected number of models.')
    parser.add_argument('--procs_lim', type=int, default=1,
                        help='Max number of processes to run in parallel.')
    args = parser.parse_args()
    print(args)
    
    # Load config file
    with open(args.config, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = json.load(f, object_pairs_hook=OrderedDict)
    
    # String in JSON config
    name = config.get('fname')
    
    # Dictionary of lists in JSON config
    param_dict = config.get('parameters')
    param_names = param_dict.keys()
    param_vals = [param_dict[k] for k in param_names]

    # Number of expected models & subsampling prob. range
    n_total_configs   = len(list(product(*param_vals)))
    n_expected_models = args.n_models if args.n_models > 0 else n_total_configs
    param_keep_prob   = min(float(n_expected_models) / n_total_configs, 1.0)

    # Fix datetime here so all runs get same one
    # log_path = get_log_dir_path(args.log_root, name)
    # print("Log path:", log_path)
    
    # Iterate over the param configs and launch subprocesses
    procs    = []
    j        = -1
    for param_set in product(*param_vals):
        if np.random.rand() > param_keep_prob:
            continue
        j += 1

        # Assemble command line argument
        proc_args = [
            'python' , args.script, 
            '--fname', name
        ]
        for k, v in zip(param_names, param_set):
            proc_args += ['--{0}'.format(k), str(v)]

        # Launch as subprocess
        print("Launching model {0}".format(j))
        print("\t".join(
                    ["%s=%s" % (k, v) for k, v in zip(param_names, param_set)]
        ))
        p = subprocess.Popen(proc_args)
        procs.append(p)
        ctr = 0
        while True:
            k = num_procs_open(procs)
            ctr += 1
            if ctr >= CTR_LIM:
                ctr = 0
                print('{0} processes still running'.format(k))
            if num_procs_open(procs) >= args.procs_lim:
                time.sleep(SLEEP)
            else:
                break

    n = len(procs)
    ctr = 0
    while True:
        k = num_procs_open(procs)
        if k == 0:
            break
        ctr += 1
        if ctr >= CTR_LIM:
            ctr = 0
            print('{0} processes still running'.format(k))
        time.sleep(SLEEP)