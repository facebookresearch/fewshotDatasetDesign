# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class CosineClassifier(nn.Module):
    def __init__(self, fSz, nClasses, phase='base', bias_grad=True, scale_grad=True):
        super(CosineClassifier, self).__init__()
        self.phase = phase
        self.out_features = nClasses
        assert self.phase in ['base', 'novel']

        self.weight_base = nn.Parameter(torch.FloatTensor(fSz, nClasses).normal_(0.0, np.sqrt(2.0/fSz)), requires_grad=True)

        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10.), requires_grad=scale_grad)
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=bias_grad)
        
    def forward(self, features_test, features_train=None, labels_train=None):
        if features_train is not None and labels_train is not None:
            self.phase = 'novel'
        elif features_train is None and labels_train is None:
            self.phase = 'base'
        else:
            raise ValueError('features train and labels train should be specified both or left None.')
        
        features_test  = F.normalize(features_test, p=2, dim=features_test.dim()-1)
        
        # get classification weights - depending on either train or test
        if self.phase == 'base':
            cls_weights = self.weight_base
        else:# novel
            # create weight with feature averaging using features_train and labels_train
            nExemplars = features_train.size(0)
            features_train = F.normalize(features_train, p=2, dim=features_train.dim()-1)
            
            # verify labels_train: should contain all labels until highest value
            # and should have at least one exemplar per novel class
            l = list(set(sorted(labels_train.tolist())))
            nKnovel = max(l)+1
            assert l == list(range(nKnovel)) and nExemplars >= nKnovel
            labels_train_onehot = torch.zeros(nExemplars, nKnovel, device=self.bias.device).scatter_(1,labels_train.view(-1,1), 1)
            avg_exemplar_feat = torch.mm(labels_train_onehot.t(), features_train).transpose(0,1) # fSz, nKnovel
            cls_weights = avg_exemplar_feat

        # normalize again
        cls_weights = F.normalize(cls_weights, p=2, dim=0) ## normalize along fSz dimension
        cls_scores = self.scale_cls * torch.addmm(1.0, self.bias, 1.0, features_test, cls_weights)
        return cls_scores
    
    def __repr__(self):
        bias = self.bias.item() if self.bias.view(-1).size(0) == 1 else self.bias.size()
        info = [
            'Cosine Classifier:',
            f'  Weight base size: {self.weight_base.size()}',
            f'  bias size: {bias}, requires grad {self.bias.requires_grad}',
            f'  Phase: {self.phase}',
            f'  Scale: {self.scale_cls.item()}, scale grad {self.scale_cls.requires_grad}',
        ]
        return '\n'.join(info)