# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import submitit
import os
import torch
import itertools
from typing import Dict
from collections.abc import Iterable
from argparse import Namespace

def grid_parameters(grid: Dict):
    """
    Yield all combinations of parameters in the grid (as a dict)
    """
    grid_copy = dict([(k,v) for (k,v) in grid.items() if len(v)])
    # Turn single value in an Iterable
    for k in grid_copy:
        if not isinstance(grid_copy[k], Iterable):
            grid_copy[k] = [grid_copy[k]]
    for p in itertools.product(*grid_copy.values()):
        yield dict(zip(grid.keys(), p))
        
def launch_grid(fct, base_opt, grid, folder, partition='learnfair', fill_dev=-1):
    """
    Launched a grid of experiments given a dict of parameters
    """
    keys = [key for key,vals in grid.items() if len(vals)>1 or key in []]
    params_list = list(grid_parameters(grid))
    names_list = []
    for params in params_list:
        name = '_'.join(['{}_{}'.format(k,str(params[k]).replace(',','_').translate({ord(i): None for i in ' []'})) for k in keys])
        names_list.append(name)
    jobs = launch_list_params(fct, base_opt, params_list, folder=folder, exp_names=names_list, partition=partition, fill_dev=fill_dev)
    return jobs

def launch_list_params(fct, base_opt, params_list, folder, exp_names=[], partition='learnfair', fill_dev=-1):
    if len(exp_names):
        assert len(exp_names) == len(params_list)
    else:
        exp_names = ['exp{:02d}'.format(i) for i in range(len(exp_names))]
    executor = submitit.AutoExecutor(folder=os.path.join(folder,'jobs/%j'))
    jobs = []
    print(f'Launching {len(params_list)} experiments')
    num=0
    for params, name in zip(params_list, exp_names):
        s = '%j_{}'.format(name)
        logdir = os.path.join(folder,s)
        
        # get base opts
        opt = Namespace(**vars(base_opt))
        opt.logdir = logdir
        # update opt with params
        for k,v in params.items():
            setattr(opt,k,v)

        n_gpus = 1
        if hasattr(opt, 'ngpus'):
            n_gpus = opt.ngpus

        mem = 60*n_gpus
        
        part = partition
        
        if fill_dev > 0 and num < fill_dev:
            part = 'dev'
        
        comment = ''
        if part =='priority':
            comment = 'eccv deadline'
        executor.update_parameters(timeout_min=4320, partition=part, constraint="volta",
                           tasks_per_node=1, gpus_per_node=n_gpus, mem_gb=mem, 
                           cpus_per_task=10, nodes=1, signal_delay_s=120, comment=comment)
        
        job = executor.submit(fct, opt)
        logdir = logdir.replace('%j',str(job.job_id))
        print(num,logdir, job.job_id)
        jobs.append({
            'job':job, 'logdir':logdir, 'opt':opt, 'job_id':job.job_id
        })
        num+=1
        
    jobdirs_file = os.path.join(folder,'jobdirs.pth')
    if os.path.exists(jobdirs_file):
        jobdirs = torch.load(jobdirs_file)
    else:
        jobdirs = []
    jobdirs.extend(jobs)
    torch.save(jobdirs, jobdirs_file)
    print('Done!')
    return jobs

def cancel_folder_jobs(folder):
    jobdirs_file = os.path.join(folder,'jobdirs.pth')
    if os.path.exists(jobdirs_file):
        jobs = torch.load(jobdirs_file)
        jobids = [str(job['job'].job_id) for job in jobs]
        command = 'scancel {}'.format(','.join(jobids))
        print(command)
        os.system(command)
    else:
        print(f'No jobs found in {folder}')
        
        
if __name__ == '__main__':
    def simple_fct(opt):
        print(opt.x,opt.y)
    grid = {
        'x':[1,3,],
        'y':[2,4,],
    }
    base_opt = Namespace(**{
        'x':1, 'y':2, 'z':3,
    })
    folder = 'runs/test/'
    jobs = launch_grid(simple_fct, base_opt, grid, folder=folder, partition='dev')
    
    
#     cancel_folder_jobs('runs/test/')