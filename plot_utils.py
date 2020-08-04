# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from glob import glob
import numpy as np
import torch
from itertools import product

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.ticker import NullFormatter,NullLocator
plt.style.use('ggplot')
from matplotlib import rcParams
rcParams['xtick.color'] = 'black'
rcParams['ytick.color'] = 'black'
rcParams['axes.labelcolor'] = 'black'
rcParams['font.weight'] = 'normal'
rcParams['font.size'] = 14
colors = ['#E24A33', '#348ABD', '#FBC15E', '#8EBA42','#B118C8', '#988ED5', '#FFB5B8', '#777777', '#0BDEDA', '#078954', '#89072E', '#073C89']

def get_args_from_submitted(submitted):
    with open(submitted, 'rb') as f:
        submission = pickle.load(f)

    args = submission.args[0]
    args = vars(args)
    return args

def enc(s):
    s = str(s).replace(',','_').translate({ord(i): None for i in ' []'})
    return s

def get_result_dict(result_dir, verbose=False, take={}):
    result_files = glob(result_dir+'*/result.pth')
    jobids = []
    for res in result_files:
        jobids.append(int(res.split('/')[-2].split('_')[0]))
    submitted_jobs = []
    for jobid in jobids:
        submitted_jobs.append(glob(result_dir + f'jobs/{jobid}/*_submitted.pkl')[0])
    
    if verbose:
        print(len(submitted_jobs))
        print(submitted_jobs[0])

    all_args = []
    for result_file, submitted in zip(result_files, submitted_jobs):
        args = get_args_from_submitted(submitted)
        all_args.append(args)

    # get launching grid
    grid = {k:[] for k in all_args[0]}
    for args in all_args:
        for k,v in args.items():
            if k in take and take[k]!=v:
                continue
            if k not in grid:
                grid[k] = [v]
            if v not in grid[k]:
                grid[k].append(v)
    grid = dict([(k,v) for k,v in grid.items() if k != 'logdir' and len(v)>1])

    # get list of results
    results = []
    arg_take = None
    for arg, res_file in zip(all_args, result_files):
        x = torch.load(res_file)
        take_res = True
        for k in take.keys():
            if arg[k] != take[k]:
                take_res = False
        if not take_res:
            continue
        updated_arg = dict([(k,v) for k,v in arg.items() if k in grid.keys()])
        arg_take = arg
        results.append((updated_arg,x))
        
    return results, grid, arg_take

def get_kshot_mean_std(xvar, grid, results, verbose=False):
    xaxis = sorted([float(x) for x in grid[xvar]])
    top1ms = []; top1stds = []
    top5ms = []; top5stds = []
    xx = []
    for x in sorted(xaxis):
        xres = [res for arg, res in results if arg[xvar] == x]
        if len(xres) == 0:
            continue
        try:
            top1s = [res[1][0] for res in xres]
        except:
            top1s = []
        if len(xres) == 1:
            print(xres)
        top1m = np.mean(top1s); top1std = np.std(top1s)
        top5s = [res[5][0] for res in xres]; top5m = np.mean(top5s); top5std = np.std(top5s)

        if verbose: print(f'{xvar}:{x} -> 1-shot:{top1m:.2f}±{top1std:.2f} -- 5-shot:{top5m:.2f}±{top5std:.2f}')
        
        top1ms.append(top1m); top1stds.append(top1std)
        top5ms.append(top5m); top5stds.append(top5std)
        xx.append(x)
    return xx, top1ms, top1stds, top5ms, top5stds

def show_grid_and_results(result_dir, shot=5):
    results, grid, args = get_result_dict(result_dir)
    print(grid)
    
    keys = list([x for x in grid.keys() if x != 'seed'])
    for x in product(*[grid[k] for k in keys]):
        take = dict(zip(keys, x))
        r, g, a = get_result_dict(result_dir, take=take)
        if not len(r):
            continue

        top5s = [res[1][shot][0] for res in r]
        if not len(top5s):
            continue
        top5m = np.mean(top5s); top5std = np.std(top5s)
        print(f'{len(top5s):02d}  -  {take} : 5-shot:{top5m:.2f}±{top5std:.2f}')
    return grid