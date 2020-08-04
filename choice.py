import torch
from utils import *
from datasets import *
import wordnet.wordnet_tools as wnt

def sample_random_classes_images_miniIN6k(nc=-1, nim=-1, seed=None, benchmark='miniIN'):
    dset = get_dataset('miniIN6k', classDset=True)
    if benchmark == 'cub':
        dset = remove_bird_classes(dset)
    elif benchmark == 'flower':
        dset = remove_flower_classes(dset)
        
    if nc > 0:
        dset = dset.sample_n_classes(n_classes=nc, seed=seed)
    if nim > 0:
        dset = dset.sample_n_im_per_class(nim=nim, seed=seed)
    return dset

def remove_bird_classes(dset):
    inds = torch.load('data/IN6k_initial_labels_ranked_by_dist_to_cub_train_5704.pth')
    dset = dset.sample_n_classes(class_inds=inds)
    print(f'Sub-sampled {len(inds)} inds from miniIN6k dataset (avoiding cub classes)')
    return dset

def remove_flower_classes(dset):
    inds = torch.load('data/IN6k_initial_labels_without_closest_flower_classes_5462.pth')
    dset = dset.sample_n_classes(class_inds=inds)
    print(f'Sub-sampled {len(inds)} inds from miniIN6k dataset (avoiding flower classes)')
    return dset

def rank_select_miniIN6k_classes_by_distance_to_dset_classes(dset, nc, feat='', rankmode='closest', benchmark='miniIN'):
    """
    - Compute feature of the input dataset 
    - Rank the miniIN6k classes w.r.t input dataset class features
    - rank by closest or farthest and select classes
    """
    miniIN6k = sample_random_classes_images_miniIN6k(benchmark=benchmark)
    if nc > len(miniIN6k.classes):
        print(f'Taking all {len(miniIN6k.classes)} classes instead of {nc}')
        return miniIN6k
    print('6k dataset :', miniIN6k)
    if feat == 'wordnet':
        syns_6k = [wnt.wnid_to_synset(c.name) for c in miniIN6k.classes]
        if benchmark == 'cub':
            bird_syn = wnt.wnid_to_synset('n1503061')
            syns_ref = [bird_syn]+bird_syn.hyponyms() # bird synset
        else:
            syns_ref = [wnt.wnid_to_synset(c.name) for c in dset.classes]
            
        print(f'Computing parwise distance between {len(syns_6k)} and {len(syns_ref)} synsets')
        pdist = torch.zeros(len(syns_6k),len(syns_ref))

        for i in range(len(syns_6k)):
            for j in range(len(syns_ref)):
                pdist[i,j] = wordnet_dist(syns_6k[i],syns_ref[j],fct='path')
    else:
        # extract dset classes
        features, targets = get_dataset_features(dset, feat=feat)
        class_features = get_class_mean_from_dset_features(features, targets)
        print(class_features.size())

        # get features of 6k classes
        class_feats = []
        for i,c in enumerate(miniIN6k.classes):
            class_feats_mean = get_miniIN6k_features_of_class(c.initial_label, feat=feat, normalize=True).mean(0)
            class_feats.append(class_feats_mean)
        miniIN6k_feats = torch.stack(class_feats)# of size nclassesxfSz

        pdist = get_pdist(miniIN6k_feats, class_features)
        
    print(f'Got pdist of size {pdist.size()}')
    
    print(f'Sorting classes using rank mode {rankmode}')
    if rankmode == 'closest':
        sorted_inds, sorted_vals = get_closest_inds_pdist(pdist, nc)
    elif rankmode == 'farthest':
        sorted_inds, sorted_vals = get_farthest_inds_pdist(pdist, nc)
    else:
        raise ValueError(f'Unknown rankmode {rankmode} for choosing classes.')
    print(sorted_vals[:nc])
#     initial_labels = [miniIN6k.classes[ind].initial_label for ind in sorted_inds]
#     class_inds = 
    print('Sampling classes from ', miniIN6k)
    dset = miniIN6k.sample_n_classes(class_inds=sorted_inds)
    return dset

def split_dataset_classes(dset, method='cluster', nsplits=4, seed=None, feat='', pcaDim=-1, cluster_type='h_median_dich'):
    
    if nsplits == 1:
        return dset
    print(f'Splitting dataset classes using method {method}, nsplits {nsplits}, seed {str(seed)}, feat {feat}, pcaDim {pcaDim}, cluster_type {cluster_type}')
    # compute dataset features
    if method != 'random':
        features, targets = get_dataset_features(dset, feat=feat)
    
    new_classes = []
    for i, c in enumerate(dset.classes):
        if method == 'cluster':
            f_i = features[targets==i]
            if cluster_type == 'kmeans':
                cluster_inds = kmeans_cluster(f_i, nsplits, pcaDim=pcaDim, verbose=False)
            elif cluster_type == 'h_median_dich':
                cluster_inds = hierarchical_splitting(f_i, nsplits=nsplits, split_step=split_median)
            else:
                raise ValueError(f'Unrecognized Clustering type: {cluster_type}')
            for j in range(nsplits):
                inds = torch.where(cluster_inds==j)[0]
                new_class, _ = c.sample_n_images(inds=inds.tolist())
                new_classes.append(new_class)
        elif method == 'random':
            perm_inds = torch.randperm(len(c), generator=get_generator(seed))
            for j in range(nsplits):
                inds = perm_inds[j::nsplits]
                new_class, _ = c.sample_n_images(inds=inds.tolist())
                new_classes.append(new_class)
        else:
            raise ValueError(f'Method {method} not recognized')

    new_dset = classDataset(classes=new_classes)
    new_dset.parent_dset = dset
    new_dset.desc = f'Dataset created by splitting each dataset class into {nsplits}'
    return new_dset

def select_classes_miniIN6k_featvar(feat='oracle', bins=10, rank=0, nc=64, nim=-1, seed=None, benchmark='miniIN'):
#     dset = get_dataset('miniIN6k')
#     if benchmark == 'cub':
#         dset = remove_bird_classes(dset)
    dset = sample_random_classes_images_miniIN6k(nc=-1, nim=-1, seed=None, benchmark=benchmark)
    measures = []
    print(f'Getting measures of {len(dset.classes)} classes')
    for i, c in enumerate(dset.classes):
        # get class features
        class_features = get_miniIN6k_features_of_class(c.initial_label, feat=feat, normalize=True)
        avg_feat = class_features.mean(0, keepdim=True)
        pdist = get_pdist(class_features, avg_feat).squeeze() # avg cosine distance
        measures.append(pdist.mean().item())
    print('Got 6000 class avg feat var')
    inds = bin_select(measures, bins=bins, rank=rank, take=nc, verbose=True, seed=seed)
#     print([measures[ind] for ind in inds])
    dset = dset.sample_n_classes(class_inds=inds)
    if nim > 0:
        dset = dset.sample_n_im_per_class(nim, seed=seed)
    return dset


def group_dataset_classes(dset, ngroups, mode='agglo', feat='', verbose=False):
    nc = len(dset.classes)
    if nc == ngroups:
        return dset
    print(f'Grouping dataset classes into {ngroups} groups, mode {mode}, feat {feat}')
    if mode == 'random':
        cluster_inds, _ = shuffle(torch.arange(ngroups).repeat(nc//ngroups), seed)
    elif mode in ['agglo','kmeans']:
        if 'wordnet' in feat:
            assert mode == 'agglo'
            cluster_inds = agglo_cluster_wordnet(dset, ngroups, fct='path')
        else:
            features, targets = get_dataset_features(dset, feat=feat)
            class_feats = get_class_mean_from_dset_features(features, targets)
            
            assert ngroups < class_feats.size(0)
            if mode == 'kmeans':
                cluster_inds = kmeans_cluster(class_feats, n_clusters=ngroups)
            elif mode == 'agglo':
                cluster_inds = agglo_cluster(class_feats, ngroups=ngroups)
            else:
                raise ValueError(f'Unknown clustering mode {mode}')

    new_classes = []
    nclasses_per_group = []
    for i in range(ngroups):
        inds = torch.where(cluster_inds==i)[0].tolist()
        nclasses_per_group.append(len(inds))
        samples = []
        if verbose:
            print('Group {} has {} classes'.format(i, len(inds)))
        for ind in inds:
            c = dset.classes[ind]
            samples.extend(c.samples)
        new_class = Class(samples)
        new_classes.append(new_class)
    
    new_dset = classDataset(classes=new_classes)
    new_dset.parent_dset = dset
    new_dset.desc = f'Dataset created by grouping dataset classes into {ngroups} groups using {mode} grouping'
    return new_dset

def sample_and_group_dataset_classes(nc, ngroups, nim, mode, seed=None, feat=''):
    """
    nc: number of sampled classes
    ngroups: number of resulting classes after grouping
    nim: total number of images unchanged with grouping
    mode: type of grouping, we use agglomerative grouping. It can be random also
    """
    dset_6k = get_dataset('miniIN6k')
    dset = dset_6k.sample_n_classes(nc, seed=seed)
    print(f'Sampling {nim} images from dataset of {nc} classes')
    dset = dset.sample_N_images(nim=nim, seed=seed)
    print(dset)
    print('Grouping dataset classes')
    dset.transform = get_transform('miniIN', phase='test')
    dset = group_dataset_classes(dset, ngroups=ngroups, mode=mode, feat=feat, verbose=False)
    return dset

