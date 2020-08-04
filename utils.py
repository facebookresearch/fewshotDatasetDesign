from argparse import Namespace
import torch
import faiss
import numpy as np
import math
from torchvision import transforms as T
from torch.utils.data import DataLoader

def get_opts():
    opt = {
        'train_type':'CC', # cosine classifier
        'setup':'CC_wrn',
        'logdir':'',
        'seed':None,
        'feat':'oracle',
        'ngpus':1,
        
        # optim
        'lr':0.05, 'momentum':0.9, 'wd':5e-4,
        'gamma':0.1, 'steps':None,
        'warmup_steps':6000, 'nesterov':False,
        
        # model
        'feat_arch': 'wrn', 
        
        # eval
        'nEpisodes':10000, 
        'eval_freq':1000,
        'test_json':'',
        'val_json':'',
        
        # dataset
        'dataset':'miniIN', 'benchmark':'miniIN',
        'iSz':84,
        
        'bSz':64, 
        'n_workers':8,
        
        'delete_ckpt':True,
        
        # exps params
        'nsplits':1, 'splitmode':'cluster',
        'cluster_type':'h_median_dich',
        'class_ratio':1,
        'group_type':'agglo',
        
        # proto params
        'nsupport':5, 'nway':5,
        'nquery':12,
        
        # splitting params
        'pcaDim':-1,
    }

    opt = Namespace(**opt)

    return opt

#### multistep warmup scheduler 

from torch.optim.lr_scheduler import *
from torch.optim.lr_scheduler import _LRScheduler

class MultiStepWarmupLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, warmup_steps=200):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.warmup_steps = warmup_steps
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        coef = (epoch + 1.0) * (1.0 / self.warmup_steps) if epoch < self.warmup_steps else 1.0
        return [coef*base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]
    
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_features(model, dataloader, verbose=False, model_eval=True, bSz=512):
    if dataloader.__class__.__name__ != 'DataLoader':
        dset = dataloader
        dataloader = DataLoader(dset, batch_size=bSz, shuffle=False, num_workers=4)
    if verbose:
        print('Computing features on dataset {}'.format(dataloader.dataset))
    if model_eval:
        model = model.eval()
    features = []; targets = []
    N = len(dataloader)
    device = list(model.parameters())[0].device
    for i, (batch,labels) in enumerate(dataloader):
        with torch.no_grad():
            batch = batch.to(device,non_blocking=True)
            output = model(batch)
            features.append(output.detach().cpu())
        targets.append(labels)
        if verbose and i%(max(1,N//10)) == 0:
            print(i)
    features = torch.cat(features)
    targets = torch.cat(targets)
    return features, targets


def get_generator(seed):
    if seed is not None:
        g = torch.Generator();g.manual_seed(seed)
    else:
        g = torch.default_generator
    return g

def get_mean_ci(l):
    l = np.array(l)
    mean = np.mean(l)
    std = np.std(l)
    ci95 = 1.96*std/np.sqrt(len(l))
    return mean, ci95


from model import Model
def get_feature_extractor(feat=''):
    model = Model(nClasses=6000)
    if feat == 'oracle':
        pretrained_path = 'data/oracle_features/best_checkpoint.pth'
        print(f'Loading oracle weights from {pretrained_path}')
        model.load_state_dict(torch.load(pretrained_path)['model'])
    elif feat == 'randfeat':
        pretrained_path = 'data/random_features/randfeat_wrn_miniIN6k_1epoch.pth'
        print(f'Loading randfeat weights from {pretrained_path}')
        model.feat.load_state_dict(torch.load(pretrained_path))
    else:
        raise ValueError(f'Empty feature type -{feat}-')
    
    model = model.feat.cuda().eval()
    return model

def get_dcIN_features(dset, bSz=512, verbose=True, model_eval=True):
    image_size = 224
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],)
    oldtransform = dset.transform
    dset.transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),normalize,
    ])

    from load_ssl_models import load_ssl_model
    model = load_ssl_model('dcIN').cuda()
    print('Got model')
    features, targets = compute_features(model, dset, verbose=False, model_eval=True, bSz=bSz)
    dset.transform = oldtransform
    return features, targets
    
def get_moco_features(dset, bSz=512, verbose=True, model_eval=True):
    image_size = 224
    crop_padding = 32
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = T.Normalize(mean=mean, std=std)
    transform = dset.transform
    dset.transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                normalize,
            ])
    CMC_path = '/private/home/sbaio/aa/dataset_design_few_shot/data/moco_features/'
    model_path = CMC_path+'/MoCo_softmax_16384_epoch200.pth'
    ckpt = torch.load(model_path)
    import sys
    path = CMC_path
    if path not in sys.path:
        sys.path.insert(0, path)

    from archs.moco_archs import InsResNet50
    model = InsResNet50()
    model.load_state_dict(ckpt['model'])
    model = model.cuda()
    
    dataloader = DataLoader(dset, batch_size=bSz, shuffle=False, num_workers=16)
    
    if verbose:
        print('Computing Moco features on dataset {}'.format(dset))
    if model_eval:
        model = model.eval()
    features = []; targets = []
    N = len(dataloader)
    device = list(model.parameters())[0].device
    for i, (batch,labels) in enumerate(dataloader):
        with torch.no_grad():
            batch = batch.to(device,non_blocking=True)
            output = model(batch, 6) # opt.layer=6
            features.append(output.detach().cpu())
        targets.append(labels)
        if verbose and i%(max(1,N//10)) == 0:
            print(i)
    features = torch.cat(features)
    targets = torch.cat(targets)
    dset.transform = transform
    return features, targets

def get_dataset_features(dset, feat='', normalize=True, feat_extractor=None):
    print(f'Computing {feat} features of {dset}')
    if feat in ['oracle', 'randfeat']:
        feat_extractor = get_feature_extractor(feat=feat)
        features, targets = compute_features(feat_extractor, dset, verbose=False)
    elif feat == 'moco':
        features, targets = get_moco_features(dset)
    elif feat == 'dcIN':
        features, targets = get_dcIN_features(dset, bSz=128)
    elif feat_extractor is not None:
        print('Computing features using given feature extractor')
        features, targets = compute_features(feat_extractor, dset, verbose=False)
    else:
        raise ValueError(f'Unknown feat {feat}')
    if normalize:
        features = torch.nn.functional.normalize(features, dim=1)
    return features, targets


def get_miniIN6k_features_of_class(class_ind, feat='', normalize=True):
    if feat=='oracle':
        f_dir = 'data/oracle_features/feats_miniIN6k/'
    elif feat == 'randfeat':
        f_dir = 'data/random_features/'
#     elif feat == 'dcIN':
#         f_dir = '/checkpoint/sbaio/features_miniIN6k_dcIN/'
#     elif feat == 'rotIN':
#         f_dir = '/checkpoint/sbaio/features_miniIN6k_rotIN/'
    elif feat == 'moco':
        f_dir = 'data/moco_features/'
    else:
        raise ValueError(f'Empty feature type -{feat}-')
    f_i = torch.load(f_dir+'features_class_{:04d}'.format(class_ind))
    if normalize:
        f_i = torch.nn.functional.normalize(f_i, dim=1)
    return f_i

def get_class_mean_from_dset_features(features, targets):
    nclasses = len(set(targets.tolist()))
    features = torch.nn.functional.normalize(features, dim=1)
    class_feats = []
    for i in range(nclasses):
        f_i = features[targets==i].mean(0)
        class_feats.append(f_i)
    class_feats = torch.stack(class_feats)
    test_nan(class_feats)
    return class_feats


def get_pdist(f1, f2):
    f1_ = torch.nn.functional.normalize(f1, dim=1)
    f2_ = torch.nn.functional.normalize(f2, dim=1)
    pdist = 1-torch.mm(f2_,f1_.t())
    return pdist.transpose(0,1)

def test_nan(x):
    y = torch.isnan(x).float().sum() == 0
    assert y
    return y



def get_min(pdist):
    vals, inds = pdist.min(1)
    val2, ind2 = vals.min(0)
    val3, ind3 = pdist[ind2].min(0)
    return val2.item(), (ind2.item(),ind3.item())

def get_closest_inds_pdist(pdist, take):
    # closest to each class, evenly, avoids having all classes selected class inds around only one class ... in opposition with using pdist.min(1).sort()
    # pdist 6000 x 20
    # return 20 indices among 6000 closest to the 20 classes
    inds = []; vals = []
    max_ = pdist.max().item()+1
    while len(inds) < take:
        pd = pdist.clone()
        for x in inds:
            pd[x] = max_
        for i in range(pd.size(1)):
            min_val, (x,y) = get_min(pd)
            pd[x] = max_
            pd[:,y] = max_
            inds.append(x)
            vals.append(min_val)
            if len(inds) == take:
                break
    return inds, vals

def get_farthest_inds_pdist(pdist, take):
    vals, inds = pdist.min(1)
    vals2, inds2 = vals.sort(descending=True)
    return inds2[:take], vals2[:take]


def wordnet_dist(syn1,syn2, fct='path'):
    # see https://github.com/nltk/nltk/blob/1805fe870635afb7ef16d4ff5373e1c3d97c9107/nltk/corpus/reader/wordnet.py
    if fct == 'path':
        return 1-syn1.path_similarity(syn2) # in [0,1]
    elif fct == 'lch':
        return -syn1.lch_similarity(syn2)
    elif fct == 'wup':
        return -syn1.wup_similarity(syn2)
    else:
        fcts = ['path','lch','wup']
        raise ValueError(f'Known set of wordnet similarities {fcts}')
        

def kmeans_cluster(x, n_clusters, verbose=False, pcaDim=-1):
    """
    K-means clustering using faiss library
    """
    assert x.dim() == 2
    n,d = x.size()
    X = x.numpy()
    if pcaDim > 0:
        if verbose: print(f'Applying PCA from {d}dimensions to {pcaDim}')
        pca = faiss.PCAMatrix(d, pcaDim)
        pca.train(X)
        assert pca.is_trained
        Xpca = pca.apply_py(X)
    else:
        Xpca = X
    if verbose: print('Clustering 2-dim tensor of size {}'.format(X.shape))

    kmeans = faiss.Kmeans(Xpca.shape[1], n_clusters, niter=20, verbose=verbose)
    kmeans.train(Xpca)
    D, I = kmeans.index.search(Xpca, 1)
    return torch.LongTensor(I).squeeze()

def grouping_step(X=None, pdist=None, verbose=False):
    """
    One step of agglomerative clustering implemented below
    Group closest two items and update pdist to select next closest ones
    Returns a list of couples of indices of half the size of the input
    """
    if X is not None and pdist is not None:
        raise ValueError('Should provide either X or pdist')
    if pdist is None:
        pdist = get_pdist(X,X)
    else:
        pdist = torch.Tensor(pdist).clone()
    assert pdist.size(0)%2 == 0
    m = pdist.max()+1
    for i in range(pdist.size(0)):
        pdist[i][i] = m
    couples = []
    for i in range(pdist.size(0)//2):
        # group closest
        ind1,ind2 = get_min(pdist)[1]
        couples.append((ind1,ind2))
        if verbose:
            print(ind1,ind2)
        pdist[ind1,:] = m; pdist[ind2,:] = m
        pdist[:,ind1] = m; pdist[:,ind2] = m
    return couples

def agglo_cluster(X, ngroups, verbose=False):
    """
    Agglomerative clustering, iterative grouping of items two-by-two
    """
    import math
    N = X.size(0)
    clusters = dict([(i,([i],X[i])) for i in range(N)])

    num_grouping = N/ngroups
    assert num_grouping == N//ngroups, f'ngroups {ngroups} needs to divide number of elements {N}'
    lg = int(math.log2(num_grouping))
    assert (2**lg) == num_grouping, f'number of elements {N}/ngroups {ngroups}={num_grouping} needs to be exponent of 2'
    for i in range(lg):
        # stack features of all elements
        X_clusters = torch.stack([xx[1] for xx in clusters.values()])
        if verbose: print(X_clusters.size())
        # group closest elements
        couples = grouping_step(X_clusters, verbose=verbose)
        keys = list(clusters.keys())
        couples = [(keys[j],keys[k]) for (j,k) in couples]
        if verbose: print(couples)
        # update clusters with couples
        for (j,k) in couples:
            clusters[j][0].extend(clusters[k][0])
            clusters[j] = (clusters[j][0], torch.stack([clusters[j][1],clusters[k][1]]).mean(0))
            clusters.pop(k)
    l = [xx[0] for xx in list(clusters.values())]
    if verbose: print(l)
    L = [-1]*N
    for i, cluster_inds in enumerate(l):
        for ind in cluster_inds:
            L[ind]=i
    return torch.LongTensor(L)

def bin_select(measures, bins=10, rank=0, take=384, verbose=False, seed=None):
    assert rank == int(rank) and rank < bins
    print(f'Selecting {take} classes, at rank {rank} with {bins} bins.')
    measures = torch.Tensor(measures)
    n = measures.size(0)
    vals_, inds_ = measures.sort()
    sub = int(n/bins)
    bin_inds = inds_[torch.arange(n)[rank*sub:(rank+1)*sub]]
    rand_inds = torch.randperm(sub, generator=get_generator(seed))[:take]
    inds = bin_inds[rand_inds]
    vals = measures[inds]
    if verbose: print(vals.min(),vals.max(), measures.min(),measures.max())
    return inds


def shuffle(x, seed):
    assert x.dim() == 1
    inds = torch.randperm(x.size(0), generator=get_generator(seed))
    return x[inds], inds


## grouping using wordnet
def get_dset_class_wordnet_pdist(dset, fct='path'):
    import wordnet.wordnet_tools as wnt
    from scipy.spatial.distance import squareform
    
    syns = [wnt.wnid_to_synset(c.name) for c in dset.classes]
    m = len(syns)
    dists = np.zeros((m * (m - 1)) // 2)
    k = 0
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            dists[k] = wordnet_dist(syns[i], syns[j], fct=fct)
            k = k + 1
    dists = squareform(dists)
    for i in range(m):
        dists[i,i]=1
    return dists

def agglo_cluster_wordnet(dset, ngroups, fct='path'):
    """
        Group the classes of the given classes using agglomerative clustering using wordnet similarity distances
        For new obtained groups, we compute the wordnet distance using pairwise distance of its sub-classes
    """
    import math
    from scipy.spatial.distance import squareform

    nc = len(dset.classes)
    ngroupings = math.log2(nc//ngroups)
    assert int(2**ngroupings) == nc/ngroups
    ngroupings = int(ngroupings)
    print(f'Agglomerative clustering using wordnet hierarchy ({fct} similarity distance) on dset of {nc} classes to obtain {ngroups} groups')
    print(f'Applying {ngroupings} steps of agglomerative pairing')
    pdist = torch.Tensor(get_dset_class_wordnet_pdist(dset, fct=fct))
    print(f'Computed pdist of size {pdist.size()}')
    couples = grouping_step(pdist=pdist)
    for _ in range(ngroupings-1):
        N = len(couples)
        dists = torch.zeros((N * (N - 1)) // 2)
        k = 0
        for i in range(0, N - 1):
            for j in range(i + 1, N):
                inds1 = torch.LongTensor(couples[i])
                inds2 = torch.LongTensor(couples[j])
                pdist_ij = pdist[inds1][:,inds2]
                dists[k] = pdist_ij.mean()
                k = k + 1
        pdist_ = torch.Tensor(squareform(dists.numpy()))
        print(f'Computed pdist of size: {pdist_.size()}')
        m = pdist_.max()
        for i in range(N):
            pdist_[i,i]=m
        new_couples = grouping_step(pdist=pdist_)
        couples = [tuple([*couples[nc[0]],*couples[nc[1]]]) for nc in new_couples]
    N = sum([len(c) for c in couples])
    L = [-1]*N
    for i,couple_inds in enumerate(couples):
        for ind in couple_inds:
            L[ind] = i
    return torch.LongTensor(L)

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

### Hierarchical splitting using maximum component variance
def split_median(X, inds=None):
    pcaDim = 2
    n,d = X.size()
    pca = faiss.PCAMatrix(d, 2)
    pca.train(X.numpy())
    Xpca = pca.apply_py(X.numpy())
    I = ((Xpca[:,0] > np.median(Xpca[:,0]))).astype(int)
    inds1 = np.where(I==0)[0]
    inds2 = np.where(I==1)[0]
    return inds1, inds2
    
def hierarchical_splitting(X, nsplits, split_step):
    nim_pclass = X.size(0)/nsplits
    def split_rec(X, nsplits, inds):
        N = inds.size(0)
        inds1, inds2 = split_step(X, inds)
        inds_ = []
        for indi in [inds1, inds2]:
            indi = torch.LongTensor(indi)
            if indi.size(0) >= 2*int(nim_pclass):
                inds_.extend(split_rec(X[indi], nsplits=nsplits//2, 
                                       inds=inds[indi]))
            else:
                inds_.extend([inds[indi].tolist()])
        return inds_
    
    assert 2**(int(math.log2(nsplits))) == nsplits
    N = X.size(0)
    inds = split_rec(X, nsplits, inds=torch.arange(N))
    l = list(range(N))
    for i, clus in enumerate(inds):
        for ind in clus:
            l[ind] = i
    return torch.LongTensor(l)


def get_steps(dset_size, bSz, nEpoch=90, epochSteps=[]):#, ngpus
    iters = dset_size//bSz#*ngpus
    totalIters = iters*nEpoch
    if not len(epochSteps):
        assert nEpoch == 90, print(f'didnt specify epochSteps for nEpoch {nEpoch}')
        epochSteps = [30,60]
    steps = [iters*x for x in epochSteps]
    return steps+[totalIters]
    
    
def save_features(dset, out_dir, feat='oracle'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    batch = 100
    N = len(dset.classes)
    for i in list(range(N))[::batch]:
        r = list(range(i,min(i+batch,N)))
        # calculate features of classes
        dset_ = dset.sample_n_classes(class_inds=r, keep_labels=True)
        loader = torch.utils.data.DataLoader(dset_, batch_size=512, num_workers=16, shuffle=False, pin_memory=True)
        print('Computing outputs of classes {} to {}'.format(r[0], r[-1]))
        features, targets = get_dataset_features(dset, feat=feat, normalize=False)
        for j in r:
            f_j = features[targets==j]
            torch.save(f_j, out_dir+'features_class_{:04d}'.format(j))
    print('Done')