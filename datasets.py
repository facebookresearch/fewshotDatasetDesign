# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import json
import torchvision.transforms as T
from utils import *
import torch
from copy import deepcopy
import os
import os.path as osp
import itertools

ROOT_PATH = osp.dirname(__file__)
miniIN_DIR = osp.join(ROOT_PATH, 'data/miniINTools/resized_images_84/')
miniIN_DIR_orig = osp.join(ROOT_PATH, 'data/miniINTools/original_images/')
CUB_DIR = osp.join(ROOT_PATH, 'data/CUB/')
TIEREDIN_DIR = osp.join(ROOT_PATH, 'data/tiered-imagenet/')
IN_dir = '/datasets01_101/imagenet_full_size/061417/'

def get_dataset(dset_name, split='train', classDset=True, iSz=84):
    transform = None
    if dset_name in ['miniIN', 'cub']:
        assert split in ['train', 'val', 'test']
        if dset_name == 'miniIN':
            DIR = miniIN_DIR if iSz == 84 else miniIN_DIR_orig
        elif dset_name == 'cub':
            DIR = CUB_DIR
        dset = ImageFolder(DIR+split)
    elif dset_name == 'miniIN6k':
        cache_file = '/private/home/sbaio/.cache/miniIN6k_clean.bin'
        dset = ImageFolderCached(cache_file=cache_file)
    elif dset_name == 'miniIN1k':
        cache_file = '/private/home/sbaio/.cache/miniIN1k.bin'
        dset = ImageFolderCached(cache_file=cache_file)
    elif 'miniIN1k' in dset_name:
        assert split == 'test'
        dset = dset_from_json('data/test_benchmarks/miniIN1k_nim100_seed0.json')
        if dset_name == 'miniIN1k_most_diverse':
            classInds = torch.load('data/test_benchmarks/100_high_diversity_inds.pth')
            dset = dset.sample_n_classes(class_inds=classInds)
        elif dset_name == 'miniIN1k_least_diverse':
            classInds = torch.load('data/test_benchmarks/100_low_diversity_inds.pth')
            dset = dset.sample_n_classes(class_inds=classInds)
        elif 'miniIN1k_diversity_rank_' in dset_name:
            rank = int(dset_name.replace('miniIN1k_diversity_rank_', ''))
            classInds = torch.load(f'data/exp_class_selection_minmax/miniIN1k_class_diversity_inds_rank_{rank}.pth')
            dset = dset.sample_n_classes(class_inds=classInds)
        elif 'miniIN1k_val_acc_rank_' in dset_name:
            rank = int(dset_name.replace('miniIN1k_val_acc_rank_', ''))
            classInds = torch.load(f'data/exp_validation_acc/miniIN1k_val_acc_inds_rank_{rank}.pth')
            dset = dset.sample_n_classes(class_inds=classInds)
    elif dset_name == 'IN':
        if split == 'train':
            dset = torch.load(osp.join('data/imagenet_train.pth'))
        else:
            dset = ImageFolder(IN_dir+'val/')
        dset = get_classDset_fromImageFolder(dset)
        splits_path = 'data/IN/IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS.json'
        with open(splits_path, 'r') as f:
            splits_dict = json.load(f)
        if split == 'train':
            inds = splits_dict['base_classes']
        elif split == 'val':
            inds  = splits_dict['novel_classes_1']
        else:
            inds = splits_dict['novel_classes_2']
        dset = dset.sample_n_classes(class_inds=inds)
    elif dset_name == 'IN6k':
        cache_file = '/private/home/sbaio/.cache/IN6k_clean.bin'
        dset = ImageFolderCached(cache_file=cache_file)
    elif dset_name == 'tieredIN':
        assert split in ['train','val','test']
        dset = ImageFolderCached(cache_file=TIEREDIN_DIR + f'cache/{split}.pth', 
                                 dset_dir=TIEREDIN_DIR + f'{split}/')
    elif dset_name == 'flower':
        assert split != 'train', 'Flower benchmark doesnt have train split'
        if split == 'test':
            dset = ImageFolder('data/oxfordFlowers/')
        elif split == 'val':
            print('| Using miniIN val for flowers !!!')
            return get_dataset('miniIN', split='val')
    else:
        raise NotImplementedError(f'Unknown dataset name {dset_name}')
    if classDset and dset.__class__.__name__ != 'classDataset':
        return get_classDset_fromImageFolder(dset)
    return dset

def get_transform(ttype='miniIN', phase='train', do_normalize=True, iSz=84):
    jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    if ttype in ['miniIN', 'cub', 'tieredIN']:
        normalize = T.Normalize(mean=[0.5006, 0.4755, 0.4274], std=[0.2225, 0.2187, 0.2231])
        transform_list = []
        if phase != 'train' :# 92
            transform_list = [T.Resize(int(iSz*1.15)),T.CenterCrop(iSz), T.ToTensor(),]
        else:
            transform_list = [T.RandomResizedCrop(iSz), jitter, T.RandomHorizontalFlip(), T.ToTensor()]

        if do_normalize:
            transform_list.append(normalize)
        return T.Compose(transform_list)
    elif ttype == 'IN':
        transform_list = []
        if phase != 'train':
            transform_list.append(T.Resize(256))
            transform_list.append(T.CenterCrop(224))
        else:
            transform_list.append(T.RandomResizedCrop(224))
            transform_list.append(jitter)
            transform_list.append(T.RandomHorizontalFlip())
        transform_list.append(lambda x: np.asarray(x))
        transform_list.append(T.ToTensor())
        transform_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return T.Compose(transform_list)
    else:
        raise NotImplementedError(f'Unknown transform type {ttype}')

    
# from torchvision.datasets import VisionDataset
import torch.utils.data as data
import numpy as np

class Class():
    def __init__(self, samples=[], label=None, name=None):
        self.samples = samples
        self.label = label
        self.initial_label = -1
        self.name = name if name is not None else self.label
        self.sampled_image_inds = None
        self.parent_class = None
        
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path_or_pilimg = self.samples[idx]
        if isinstance(path_or_pilimg, str):
            sample = default_loader(path_or_pilimg)
        elif path_or_pilimg.__class__.__name__ == 'Image':
            sample = path_or_pilimg
        else:
            raise NotImplementedError('Sample should be path or PIL image')
        return (sample, self.label)
    
    def sample_n_images(self, inds=None, nim=-1, seed=None):
        """
        return a new class with sampled images
        """
        if inds is None:
            assert nim > 0
            inds = torch.randperm(len(self.samples), generator=get_generator(seed))[:nim]
        
        new_samples = [self.samples[i] for i in inds]
        new_class = Class(new_samples, label=self.label)
        new_class.parent_class = self
        new_class.sampled_image_inds = inds
        new_class.initial_label = self.initial_label
        return new_class, inds
        
    def __repr__(self):
        body = ["Class '{}' has {} samples with target: {}".format(self.name, len(self.samples),self.label)]
        if self.initial_label != -1:
            body += ['Initial label: {}'.format(self.initial_label)]
        if self.sampled_image_inds is not None:
            body += ['Sampled {} images inds'.format(len(self.sampled_image_inds))]
        if self.parent_class:
            body += ['From : {}'.format(self.parent_class)]
        return '\n'.join(body)

class classDataset(data.Dataset):
    def __init__(self, classes=[], transform=None, keep_labels=False):
        super().__init__()
        self.classes = classes
        self.cum_len = np.cumsum([0]+[len(c) for c in classes])
        self.n_samples = self.cum_len[-1]
        self.len_classes = [len(c) for c in self.classes]
        self.desc = ''
        self.transform = transform
        
        self.parent_dset = None
        self.sampled_class_inds = None
        self.sampled_image_inds = None
        
        if not keep_labels:
            self._update_class_labels()
            
        d = {}
        for i in range(len(self.classes)):
            x = self.cum_len[i]
            x_1 = self.cum_len[i+1]
            for j in range(x,x_1):
                d[j] =  (i,j-x)
        self.index_to_sample_d = d
        
    def __getitem__(self, index):
        c,ind = self.index_to_sample_d[int(index)]
        sample, target = self.classes[c][ind]
        if self.transform:
            sample = self.transform(sample)
        return sample, target
        
    def __len__(self):
        return self.n_samples
    
    def __add__(self, dset):
        new_dset = classDataset(classes=self.classes+dset.classes)
        new_dset.desc = f'Dataset created by adding datasets {self} and {dset}'
        return new_dset
    
    def _class_labels_ok(self):
        """
        Verify if class labels are coherent
        """
        return [c.label for c in self.classes] == list(range(len(self.classes)))
    
    def _update_class_labels(self):
        """
        Reassign class labels when new dataset created from sampled classes
        """
        if not self._class_labels_ok():
            for i,c in enumerate(self.classes):
                c.label = i
    
    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += ["Number of classes: {}".format(len(self.classes))]
        if self.len_classes:
            body += ["min/class: {}, max/class: {}".format(min(self.len_classes), max(self.len_classes))]
        body += ["Transform: {}".format(self.transform)]
        if self.desc:
            body.append(self.desc)
        if self.sampled_class_inds is not None:
            body.append("Sampled class inds: {} ...".format(self.sampled_class_inds[:20]))
        if self.sampled_image_inds is not None:
            body.append("Sampled image inds: {}/class".format(len(self.sampled_image_inds[0])))
        if self.parent_dset is not None:
            body.append("Parent dataset: {}".format('\n'.join([" " *4 + line for line in self.parent_dset.__repr__().split('\n')])))
        body += ['----------------------------------------']
        lines = [head] + [" " *4 + line for line in body]
        return '\n'.join(lines)+'\n'
    
    def sample_n_classes(self, n_classes=None, method='random', seed=None, class_inds=None, keep_labels=False):
        if class_inds is not None:
            pass
        elif method == 'random':
            if class_inds is None:
                g = get_generator(seed)
                class_inds = torch.randperm(len(self.classes), generator=g).tolist()[:n_classes]
            else:
                assert n_classes == len(class_inds)
        else:
            assert class_inds is None
        new_classes = [deepcopy(self.classes[i]) for i in class_inds]
        new_dset = classDataset(classes=new_classes, transform=self.transform, keep_labels=keep_labels)
        new_dset.sampled_class_inds = class_inds
        new_dset.parent_dset = self
        return new_dset
    
    def sample_n_im_per_class(self, nim=-1, seed=None):
        """
        Sample nim per class
        """
        new_classes = []
        sampled_image_inds = []
        g = get_generator(seed)
        for i,c in enumerate(self.classes):
            inds = torch.randperm(len(c), generator=g).tolist()[:nim]
            new_class, sampled_inds = c.sample_n_images(inds=inds)
            new_classes.append(new_class)
            sampled_image_inds.append(inds)
        new_dset = classDataset(classes=new_classes, transform=self.transform)
        new_dset.sampled_image_inds = sampled_image_inds
        new_dset.parent_dset = self
        return new_dset
        
    def sample_N_images(self, nim, seed=None):
        """
        Sample nim images uniformly across all classes
        """
        if nim >= len(self):
            return self
        nclasses=len(self.classes)
        q = nim//nclasses
        ncmin = nclasses*(q+1)-nim
        ncmax = nclasses-ncmin
        print(q,ncmax,ncmin)
        l = [q+1]*ncmax + [q]*ncmin
        assert (q+1)*ncmax + q*ncmin == nim
        gen = get_generator(seed=seed)
        r = torch.randperm(len(l), generator=gen).tolist()
        l = [l[i] for i in r]

        new_classes = []
        for i,c in enumerate(self.classes):
            new_class, _ = c.sample_n_images(nim=l[i], seed=seed)
            new_class.name = c.name
            new_classes.append(new_class)
            
        new_dset = classDataset(classes=new_classes)
        new_dset.parent_dset = self
        new_dset.desc = f'Dataset created by sampling {nim} images from parent dset'
        new_dset.transform = self.transform
        nc_ = len(sorted([len(c) for c in new_dset.classes if len(c)>0]))
        print('Sampled {} images from {} classes'.format(nim, nc_))
        return new_dset
    
    def split_support_query(self, nsupport=1, nclasses=5, nquery=-1, seed=None):
        dset = self.sample_n_classes(nclasses, seed=seed) if nclasses > 0 else self
        g = get_generator(seed)
        Sclasses = []; Ssampled_inds = []
        Qclasses = []; Qsampled_inds = []
        for i,c in enumerate(dset.classes):
            inds = torch.randperm(len(c), generator=g).tolist()
            Sinds = inds[:nsupport]
            Qinds = inds[nsupport:nsupport+nquery] if nquery > 0 else inds[nsupport:]
            
            new_class, sampled_inds = c.sample_n_images(inds=Sinds)
            Sclasses.append(new_class)
            Ssampled_inds.append(Sinds)
            
            
            new_class, sampled_inds = c.sample_n_images(inds=Qinds)
            Qclasses.append(new_class)
            Qsampled_inds.append(Qinds)

        Sdset = classDataset(classes=Sclasses)
        Sdset.sampled_image_inds = Ssampled_inds
        Sdset.parent_dset = dset

        Qdset = classDataset(classes=Qclasses)
        Qdset.sampled_image_inds = Qsampled_inds
        Qdset.parent_dset = dset
           
        return Sdset, Qdset
    
    def get_labels(self):
        return list(itertools.chain(*[[i]*len(c) for i,c in enumerate(self.classes)]))
    
    def equalize_classes(self):
        # duplicate samples of some classes to have equal number of images for all classes
        l = [len(c) for c in self.classes]
        new_classes = []
        for i,c in enumerate(self.classes):
            new_c = Class(c.samples+c.samples[:max(l)-len(c)], label=i, name=c.name)
            new_classes.append(new_c)
        dset = classDataset(classes=new_classes)
        dset.parent_dset = self
        
        return dset
    
def get_classDset_fromImageFolder(dset):
    """
        dset needs to have .classes, .samples, .class_to_idx(optional), .root(optinal)
        .class_to_idx not present, we assume that targets in the samples are the same as class index in .classes
        
    """
    
    if not hasattr(dset, 'samples') and hasattr(dset, 'data'):
        from PIL import Image
        ims = [Image.fromarray(np_img) for np_img in list(dset.data)] # np_img of size 32,32,3
        dset.samples = list(zip(ims, dset.targets))
        
    samples_per_class = {}
    for im_path,target in dset.samples:
        if target not in samples_per_class:
            samples_per_class[target] = []
        samples_per_class[target].append(im_path)
    
    classes = []
    for i,c in enumerate(dset.classes):
        idx = i
        if hasattr(dset,'class_to_idx'):
            idx = dset.class_to_idx[c]
        samples = samples_per_class[idx]
        cs = Class(samples, label=idx, name=c)
        cs.initial_label = i
        classes.append(cs)
    
    root = getattr(dset, 'root') if hasattr(dset,'root') else ''
    cdset = classDataset(classes=classes)
    return cdset




#### Dataset caching
import os
import torch
from torchvision.datasets.vision import VisionDataset 
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

def make_cache(dset_dir, cache_file):
    ### To make a cache - run this separately
    import time
    start = time.time()
    dset = ImageFolder(dset_dir)
    print('Took {:0.2f}'.format(time.time()-start))
    tosave = {
        'classes':dset.classes,
        'class_to_idx':dset.class_to_idx,
        'samples':dset.samples
    }
    torch.save(tosave, cache_file)
    print('Saved cache of dataset from {} to {}'.format(dset_dir, cache_file))
    
class ImageFolderCached(VisionDataset):
    def __init__(self, cache_file, transform=None, target_transform=None, dset_dir=None):
        super().__init__(root=None)
        self.transform = transform
        self.target_transform = target_transform

        if not osp.exists(cache_file):
            make_cache(dset_dir, cache_file)
        
        data = torch.load(cache_file)
        self.classes = data['classes']
        self.class_to_idx = data['class_to_idx']
        self.samples = data['samples']
        self.targets = [s[1] for s in self.samples]
        self.imgs = self.samples
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
    

def get_dataset_class_features(dset, ttype='miniIN', feat='oracle'):
    if dset.transform is None:
        dset.transform = get_transform(ttype=ttype, phase='test')
    features, targets = get_dataset_features(dset, feat=feat)
    class_features = get_class_mean_from_dset_features(features, targets)
    return class_features
    
def get_miniIN6k_class_features(feat='oracle', verbose=False):
    feats = []
    
    miniIN6k = get_dataset('miniIN6k')
    for i,c in enumerate(miniIN6k.classes):
        features = get_miniIN6k_features_of_class(i, feat=feat)
        features = torch.nn.functional.normalize(features, dim=1)
        features = features.mean(0)
        feats.append(features)
        if verbose and i % 100 == 0:
            print(i)
    feats = torch.stack(feats)
    return feats

def dset_from_json(jsonfile):
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    classes = []
    label_2_paths = {}
    for path, label in zip(data['image_names'], data['image_labels']):
        if label not in label_2_paths:
            label_2_paths[label] = [path]
        else:
            label_2_paths[label].append(path)
            
    classes = []
    for label, paths in label_2_paths.items():
        c = Class(paths, label=label)
        classes.append(c)
    dset = classDataset(classes=classes)
    return dset