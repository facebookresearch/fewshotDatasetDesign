# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from trainer import Trainer
from utils import *
from choice import *
from datasets import *
import os
import json
import uuid
    
def update_logdir(opt):
    try:
        import submitit
        job_env = submitit.JobEnvironment()
        opt.logdir = opt.logdir.replace('%j', str(job_env.job_id))
    except:
        print('No job id found')
        opt.logdir = 'runs/test/'
    if not os.path.exists(opt.logdir):
        os.mkdir(opt.logdir)
        
def run_setup(dset, opt):
    update_logdir(opt)
    assert 'setup' in opt
    train_type, arch = opt.setup.split('_')
    print(train_type, arch)
    if train_type == 'CC':
        opt.feat_arch = arch
        trainer = Trainer(opt, dset)
        return trainer.train()
    elif train_type in ['protonet','matchingnet','relationnet','baseline++','baseline']:
        # equalize dataset classes 
        l = [len(c) for c in dset.classes]
        if max(l) == min(l) + 1:# that means that the dataset has been split
            print('Equalizing dataset classes')
            dset = dset.equalize_classes()
            print(dset)
        # save dataset json
        baseJson = save_dataset_json(dset, opt.logdir)
        params = get_closerlook_params(opt)
        params.base_json = baseJson
        params.method = train_type
        params.model = CL_arch(arch)
        params.dataset = CL_bench(opt.benchmark)
        if 'stop_epoch' in opt:
            params.stop_epoch = opt.stop_epoch
        if 'n_episode' in opt:
            params.n_episode = opt.n_episode
        print('Running with dataset ', dset)
        launch_closerlook(params)
    else:
        raise ValueError('Unkown train type ')

def exp_miniIN_cub(opt):
    assert opt.dataset in ['miniIN','cub']
    dset = get_dataset(opt.dataset)
    opt.benchmark = opt.dataset
    print(f'benchmark {opt.benchmark}, dataset {opt.dataset}')

    run_setup(dset, opt)

def exp_tradeoff(opt):
    """
    N: total number of images of the dataset
    nc: number of dataset classes
    """
    dset = sample_random_classes_images_miniIN6k(benchmark=opt.benchmark)
    dset = dset.sample_n_classes(n_classes=opt.nc, seed=opt.seed).sample_N_images(nim=opt.N, seed=opt.seed)
    run_setup(dset, opt)
    
def exp_train_miniIN6k(opt):
    opt.bSz = 256
    opt.iSz = 84
    opt.lr = 0.1
    opt.n_workers = 20
    opt.steps = None
    
    dset = sample_random_classes_images_miniIN6k(benchmark=opt.benchmark)
    print(f'Steps: {opt.steps}')

    run_setup(dset, opt)

def exp_closest_farthest_test(opt):
    update_logdir(opt)
    if opt.rankmode == 'random':
        dset_6k = sample_random_classes_images_miniIN6k(benchmark=opt.benchmark)
        dset = dset_6k.sample_n_classes(n_classes=opt.nc, seed=opt.seed)
    else:
        testdset = get_dataset(opt.benchmark, split='test')
        testdset.transform = get_transform('miniIN', phase='test')
        dset = rank_select_miniIN6k_classes_by_distance_to_dset_classes(testdset, nc=opt.nc, feat=opt.feat, rankmode=opt.rankmode, benchmark=opt.benchmark)
    
    dset = dset.sample_n_im_per_class(nim=opt.nim, seed=opt.seed)
    
    run_setup(dset, opt)
    
def exp_IN_split(opt):
    update_logdir(opt)

    nsplits = opt.nsplits
    if nsplits > 1:
        dset = dset_from_json(f'data/IN/split_IN/IN_split_{nsplits}_h_median_dich_oracle_389.json')
    else:
        dset = get_dataset('IN')
        
    return run_IN_exp(dset, opt)

def exp_split_miniIN6k_cub(opt):
    update_logdir(opt)
    opt.benchmark = 'cub'
    opt.n_workers = 20
    opt.lr = 0.1
    opt.bSz = 256
    opt.iSz = 84
    nsplits = opt.nsplits
    if nsplits > 1:
        dset = dset_from_json(f'data/IN6k/split_miniIN6k/miniIN6k_split_{nsplits}_h_median_dich_oracle_5704.json')
    else:
        dset = sample_random_classes_images_miniIN6k(benchmark=opt.benchmark)
#     opt.steps = get_steps(len(dset), opt.bSz)
#     print(f'Steps: {opt.steps}')
    
    trainer = Trainer(opt, dset)
    return trainer.train()

def exp_group_split(opt):
    opt.steps = None # just added
    update_logdir(opt)
    seed = opt.seed
    nc = opt.nc
    class_ratio = opt.class_ratio
    N = opt.N # 38400
    nim = N // nc
    dset = sample_random_classes_images_miniIN6k(opt.nc, nim, seed, benchmark=opt.benchmark)
    nc = len(dset.classes)
    feat = opt.feat # default to oracle
    if class_ratio == 1:
        traindset = dset
    elif class_ratio > 1:
        # split
        assert class_ratio == int(class_ratio)
        nsplits = int(class_ratio)
        dset.transform = get_transform(phase='test')
        splitdset = split_dataset_classes(dset, method='cluster', 
                                      nsplits=nsplits, seed=seed, feat=feat, cluster_type=opt.cluster_type)
        traindset = splitdset
    else:
        # group
        ngroups = int(nc*class_ratio)
        dset.transform = get_transform(phase='test')
        groupdset = group_dataset_classes(dset, ngroups=ngroups, mode=opt.group_type, feat=feat, verbose=False)
        traindset = groupdset

    run_setup(traindset, opt)

def run_IN_exp(dset, opt):
    update_logdir(opt)
    opt.benchmark = 'IN'
    opt.lr = 0.1
    opt.bSz = 128
    trainer = Trainer(opt, dset)
    return trainer.train()

def exp_IN_base(opt):
    dset = get_dataset('IN')
    return run_IN_exp(dset, opt)

def exp_IN_IN6k(opt):
    dset = get_dataset('IN6k')
    return run_IN_exp(dset, opt)

def exp_IN_tradeoff(opt):
    dset = get_dataset('IN6k')
    dset = dset.sample_n_classes(n_classes=opt.nc).sample_N_images(nim=opt.N)
    return run_IN_exp(dset, opt)    

def get_closerlook_params(opt=None):
    from argparse import Namespace
    params = Namespace(base_json='', dataset='miniImagenet', method='protonet', model='Conv4', n_shot=5, num_classes=200, resume=False, save_freq=50, start_epoch=0, stop_epoch=-1, test_n_way=5, train_aug=True, train_n_way=5, warmup=False, use_tensorboard=True, save_iter=-1, split='novel', adaptation=False, iter_num=600, iSz=84)
    if opt is not None:
        if hasattr(opt, 'stop_epoch'):
            params.stop_epoch = 1

        if hasattr(opt, 'iter_num'):
            params.iter_num = opt.iter_num
        
        if hasattr(opt, 'logdir'):
            params.logdir = opt.logdir
    return params

def launch_closerlook(params):
    import sys
    sys.path.insert(0, '/private/home/sbaio/aa/dataset_design_few_shot/cl_fsl')
    from train import run
    from save_features import run_save
    from test import run_test
    print('Launching Closer Look training with params', params)
    
    # train
    run(params)
    
    # save features
    run_save(params)
    
    # test
    run_test(params)
    
def reproduce_closerlook(opt):
    update_logdir(opt)
    params = get_closerlook_params(opt)

    params.method = opt.method
    params.dataset = opt.dataset
    params.logdir = opt.logdir
    params.model = opt.model
    
    torch.save(params, os.path.join(opt.logdir, 'params.pth'))
    launch_closerlook(params)

def save_dataset_json(dset, logdir, filename='base.json'):
    # create json of the dataset
    data = {}
    data['image_names'] = []
    data['image_labels'] = []
    for c in dset.classes:
        for sample in c.samples:
            data['image_names'].append(sample)
            data['image_labels'].append(c.label)
    
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    baseJson = os.path.join(logdir, filename)
    with open(baseJson, 'w') as outfile:
        json.dump(data, outfile)
    
    return baseJson
    
def exp_closerlook_splitting(opt):
    update_logdir(opt)
    dset = get_dataset(opt.initial_dataset)
    seed = opt.seed
    if opt.class_ratio == 1:
        traindset = dset
    elif opt.class_ratio > 1:
        assert opt.class_ratio == int(opt.class_ratio)
        nsplits = int(opt.class_ratio)
        dset.transform = get_transform(phase='test')
        splitdset = split_dataset_classes(dset, method='cluster', nsplits=nsplits, seed=seed, feat='oracle', pcaDim=opt.pcaDim, cluster_type=opt.cluster_type)
        traindset = splitdset
    else:
        raise ValueError('class ratio < 1')
    
    baseJson = save_dataset_json(traindset, opt.logdir)
        
    params = get_closerlook_params(opt)
    params.base_json = baseJson
    if opt.initial_dataset == 'miniIN':
        params.dataset = 'miniImagenet'
    elif opt.initial_dataset == 'cub':
        params.dataset = 'CUB'
    else:
        raise ValueError('unknown initial dataset')

    launch_closerlook(params)


def exp_miniIN_CUB(opt):
#     opt.bSz = 256
    opt.ngpus = 1
    opt.iSz = 224 if '224' in opt.feat_arch else 84
#     opt.lr = 0.1
    dset = get_dataset(opt.dataset, iSz=opt.iSz)
    opt.benchmark = opt.dataset

    print(f'Steps: {opt.steps}')
    trainer = Trainer(opt, dset)
    return trainer.train()

def CL_arch(cc_arch):
    if cc_arch == 'wrn':
        arch = 'WRN'
    elif cc_arch == 'conv4':
        arch = 'Conv4'
    else:
        raise ValueError(f'unknown arch {cc_arch}')
    return arch

def CL_bench(cc_bench):
    if cc_bench == 'miniIN':
        dataset = 'miniImagenet'
    elif cc_bench == 'cub':
        dataset = 'CUB'
    else:
        raise ValueError(f'unknown benchmark {cc_bench}')
    return dataset


def exp_miniIN6k(opt):
    update_logdir(opt)
    miniIN6k = get_dataset('miniIN6k')
    if opt.benchmark == 'cub':
        miniIN6k = remove_bird_classes(miniIN6k)
    run_setup(miniIN6k, opt)

    
def exp_base_training(opt):
    trainer = Trainer(opt)
    trainer.train()
    
def exp_splitting(opt):
    # opt.benchmark
    # opt.feat
    # opt.nsplits
    # opt.splitmode
    seed = opt.seed
    dset = get_dataset('miniIN6k')
    sampled_dset = dset.sample_n_classes(n_classes=64, seed=seed)
    sampled_dset = sampled_dset.sample_n_im_per_class(nim=600, seed=seed)
    sampled_dset.transform = get_transform(phase='test')
    splitdset = split_dataset_classes(sampled_dset, method=opt.splitmode, 
                                  nsplits=opt.nsplits, seed=seed, feat=opt.feat)
    trainer = Trainer(opt, splitdset)
    return trainer.train()


def exp_proto_rand_classes(opt):
    nim = opt.N//opt.nc
    dset = sample_random_classes_images_miniIN6k(opt.nc, nim, opt.seed, benchmark=opt.benchmark)
    trainer = Trainer(opt, dset)
    return trainer.train_proto()

def exp_proto_split_classes(opt):
    seed = opt.seed
    dset = get_dataset('miniIN6k')
    sampled_dset = dset.sample_n_classes(n_classes=64, seed=seed)
    sampled_dset = sampled_dset.sample_n_im_per_class(nim=600, seed=seed)
    sampled_dset.transform = get_transform(phase='test')
    splitdset = split_dataset_classes(sampled_dset, method=opt.splitmode, 
                                  nsplits=opt.nsplits, seed=seed, feat=opt.feat)
    trainer = Trainer(opt, splitdset)
    return trainer.train_proto()


    
def group_split_miniIN(opt):
    class_ratio = opt.class_ratio
    seed = opt.seed
    dset = get_dataset('miniIN')
    nc = len(dset.classes)
    if class_ratio == 1:
        traindset = dset
    elif class_ratio > 1:
        # split
        assert class_ratio == int(class_ratio)
        nsplits = int(class_ratio)
        dset.transform = get_transform(phase='test')
        splitdset = split_dataset_classes(dset, method='cluster', nsplits=nsplits, 
                                          seed=seed, feat='oracle', pcaDim=opt.pcaDim, cluster_type=opt.cluster_type)
        traindset = splitdset
    else:
        # group
        dset.transform = get_transform(phase='test')
        ngroups = int(nc*class_ratio)
        groupdset = group_dataset_classes(dset, ngroups=ngroups, mode='agglo', feat='oracle', verbose=False)
        traindset = groupdset
        
    if opt.train_type == 'SS':
        import sys
        sys.path.insert(0, '/private/home/sbaio/aa/simple_shot/src/')
        from train_simpleshot import get_args, train_simpleshot
        
        args = get_args()
        args.seed = opt.seed
        args.arch = opt.arch #wideres
        args.logdir = opt.logdir
        args.batch_size = 64
        args.epochs = 90
        args.meta_val_interval = 4
        # for val and test data
        args.data = '/private/home/sbaio/aa/simple_shot/data/miniIN/'
        args.split_dir = '/private/home/sbaio/aa/simple_shot/split/mini/'
        args.disable_tqdm = True
        args.workers = 8
        args.clean_ckpts = True
        args.traindset = traindset
        args.num_classes = len(traindset.classes)
        
        oneshot,fiveshot = train_simpleshot(args)
        return {1:(oneshot[4],oneshot[5]) , 5:(fiveshot[4],fiveshot[5])}
    
    elif opt.train_type == 'FEAT':
        import sys
        sys.path.insert(0, '/private/home/sbaio/aa/FEAT/')
        from exp import train_feat_dset
        from argparse import Namespace
        args = Namespace()
        args.model_type = opt.model_type
        args.shot = opt.shot
        args.logdir = opt.logdir
        return train_feat_dset(traindset, args)
    else:
        trainer = Trainer(opt, traindset)
        return trainer.train()

from trainer import Trainer

def evaluate_ckpt(args):#ckpt_path, test_json='miniIN1k_nim100_seed0', logdir
    update_logdir(args)

    x = torch.load(args.ckpt_path)
    sd = dict([(k,v) for k,v in x['model'].items() if 'fc' not in k])
    
    opt = x['opt']
    if args.test_json in ['miniIN','cub','flower']:
        opt.test_json = ''
        opt.benchmark = args.test_json
        print(f'Evaluating on {opt.benchmark} benchmark')
    else:
        opt.test_json = args.test_json
        print(f'Evaluating using {opt.test_json} json')
    
    opt.logdir = args.logdir
    opt.ngpus = 1
    trainer = Trainer(opt)
    trainer.model.load_state_dict(sd, strict=False)
    nEpisodes=args.nEpisodes if 'nEpisodes' in args else 10000
    ret = trainer.eval_fewshot(trainer.testdset, nEpisodes=nEpisodes, test=True)
    if 'return_five_shot' in args and args.return_five_shot:
        return ret[5][0]
    if 'save' in args and args.save:
        torch.save(ret, os.path.join(args.logdir, 'result.pth'))
    return ret

def class_selection_exp(opt):
    update_logdir(opt)
    N = opt.N
    nim = N//opt.nc
    # remove bird classes to have comparable results between miniIN and cub
    benchmark = 'cub' if opt.remove_bird_classes else opt.benchmark
    
    dset = select_classes_miniIN6k_featvar(rank=opt.rank, nc=opt.nc, nim=nim, seed=opt.seed, benchmark=benchmark)
    
    run_setup(dset, opt)
    
def class_selection_exp_5kfarthest_5benchs(opt):
    update_logdir(opt)
    
    bins = 10
    nc = 384
    nim = 100
    feat = 'oracle'
    
    rank = opt.rank
    seed = opt.seed
    
    dset = get_dataset('miniIN6k')
    # remove classes first 1000 closest given 5 benchmarks
    # 'cub', 'miniIN', 'flower', 'miniIN1k_most_diverse','miniIN1k_least_diverse'
    inds = torch.load('data/exp_filter/sorted_by_average_dist_to_5_benchs.pth')#sorted by increasing distance
    inds = inds[1000:]
    dset = dset.sample_n_classes(class_inds=inds)
    print(f'Sub-sampled {len(inds)} inds from miniIN6k dataset (avoiding 1000 closest to 5 benchs)')
    
    measures = []
    print(f'Getting measures of {len(dset.classes)} classes')
    for i, c in enumerate(dset.classes):
        class_features = get_miniIN6k_features_of_class(c.initial_label, feat=feat, normalize=True)
        avg_feat = class_features.mean(0, keepdim=True)
        pdist = get_pdist(class_features, avg_feat).squeeze() # avg cosine distance
        measures.append(pdist.mean().item())
    print(f'Got {len(measures)} class avg feat var')
    inds = bin_select(measures, bins=bins, rank=rank, take=nc, verbose=True, seed=seed)

    dset = dset.sample_n_classes(class_inds=inds)
    if nim > 0:
        dset = dset.sample_n_im_per_class(nim, seed=seed)
        
    run_setup(dset, opt)
    

def class_selection_exp_cub_different_thresh(opt):
    update_logdir(opt)
    
    bins = 10
    nc = 384
    nim = 100
    feat = 'oracle'
    
    rank = opt.rank
    seed = opt.seed
    nc_to_remove = opt.nc_to_remove # nb of closest classes to remove
    
    dset = get_dataset('miniIN6k')
    # remove classes first 1000 closest given 5 benchmarks
    # 'cub', 'miniIN', 'flower', 'miniIN1k_most_diverse','miniIN1k_least_diverse'
    inds = torch.load('data/exp_filter/sorted_by_average_dist_to_cub_bench.pth')#sorted by increasing distance
    inds = inds[nc_to_remove:]
    dset = dset.sample_n_classes(class_inds=inds)
    print(f'Sub-sampled {len(inds)} inds from miniIN6k dataset (avoiding {nc_to_remove} closest to cub bench)')
    
    measures = []
    print(f'Getting measures of {len(dset.classes)} classes')
    for i, c in enumerate(dset.classes):
        class_features = get_miniIN6k_features_of_class(c.initial_label, feat=feat, normalize=True)
        avg_feat = class_features.mean(0, keepdim=True)
        pdist = get_pdist(class_features, avg_feat).squeeze() # avg cosine distance
        measures.append(pdist.mean().item())
    print(f'Got {len(measures)} class avg feat var')
    inds = bin_select(measures, bins=bins, rank=rank, take=nc, verbose=True, seed=seed)

    dset = dset.sample_n_classes(class_inds=inds)
    if nim > 0:
        dset = dset.sample_n_im_per_class(nim, seed=seed)
        
    run_setup(dset, opt)
    
    

    
def exp_closest_farthest_classes(opt):
    update_logdir(opt)
    nc = 64
    nim = 600
    if opt.benchmark == 'cub':
        filename = 'data/exp_closest_farthest/cub_inds.pth'
    
    inds, dist = torch.load(filename)[opt.rank]
    print(f'Rank {opt.rank}, dist: {dist}')
    
    dset = get_dataset('miniIN6k')
    dset = dset.sample_n_classes(class_inds=inds).sample_n_im_per_class(nim, seed=opt.seed)
    
    print('Running with dataset')
    print(dset)
    run_setup(dset, opt)

    
def class_selection_exp_min_max_similarity(opt):
    update_logdir(opt)
    bins = 10
    rank = opt.rank # bin rank in miniIN6k
    seed = opt.seed
    
    assert seed in range(5)
    assert rank in range(bins)
    if 'feat' in opt and opt.feat == 'moco':
        outDir = f'data/exp_class_diversity_moco/'
    else:
        outDir = f'data/exp_class_selection_minmax/'
    
    if 'selection' in opt and opt.selection == 'random':
        selection = 'random'
    else:
        selection = 'select'
    filename = outDir+f'{opt.benchmark}_{selection}_64_classes_seed_{seed}.pth'
    
    print(f'Loading inds from {filename}')
    xx = torch.load(filename)
    
    inds = xx[:,rank]
    
    nc = len(inds)
    nim = 38400//nc
    
    dset = get_dataset('miniIN6k')
    dset = dset.sample_n_classes(class_inds=inds)
    
    if nim > 0:
        dset = dset.sample_n_im_per_class(nim, seed=seed)
        
    print('Running with dataset')
    print(dset)
    run_setup(dset, opt)
    
    
def exp_rank_val_acc(opt):
    update_logdir(opt)
    bins = 10
    rank = opt.rank # bin rank in miniIN6k
    seed = opt.seed
    
    assert seed in range(5)
    assert rank in range(bins)
    if 'selection' in opt and opt.selection == 'random':
        selection = 'random'
    else:
        selection = 'select'
    filename = f'data/exp_validation_acc/{opt.benchmark}_{selection}_64_classes_seed_{seed}.pth'
    print(f'Loading inds from {filename}')
    xx = torch.load(filename)
    
    inds = xx[:,rank]
    
    nc = len(inds)
    nim = 38400//nc
    
    dset = get_dataset('miniIN6k')
    dset = dset.sample_n_classes(class_inds=inds)
    
    if nim > 0:
        dset = dset.sample_n_im_per_class(nim, seed=seed)
        
    print('Running with dataset')
    print(dset)
    run_setup(dset, opt)