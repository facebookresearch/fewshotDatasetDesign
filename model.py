# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn

class Model(nn.Module):
    def __init__(self, feat_arch='wrn', classifier_arch='cosine', nClasses=80, bias_grad=True, scale_grad=True, **kwargs):
        assert classifier_arch in ['dotproduct','cosine']
        super(Model, self).__init__()
        feat_kwargs = { k: kwargs[k] for k in ['depth', 'widen_factor', 'drop_rate'] if k in kwargs}
        self.feat, fSz = create_feature_extractor(feat_arch=feat_arch, **feat_kwargs)
        
        if classifier_arch == 'cosine':
            from archs.cosine_classifier import CosineClassifier
            self.fc = CosineClassifier(fSz, nClasses, bias_grad=bias_grad, scale_grad=scale_grad)
        else:
            self.fc = nn.Linear(fSz, nClasses)
            
        self.classifier_arch = classifier_arch

    def forward(self, x):
        x = self.feat(x)
        return self.fc(x)

def create_feature_extractor(feat_arch='wrn', **kwargs):
    if feat_arch == 'wrn':
        from archs.wide_resnet import WideResNet
        depth = kwargs['depth'] if 'depth' in kwargs else 28 
        widen_factor = kwargs['widen_factor']  if 'widen_factor' in kwargs else 10
        drop_rate = kwargs['drop_rate']  if 'drop_rate' in kwargs else 0.
        feat_extractor = WideResNet(depth=depth, widen_factor=widen_factor, dropRate=drop_rate)
        fSz = 64*widen_factor
    elif feat_arch == 'conv4':
        from archs.Conv4_CL import ConvNet
        feat_extractor = ConvNet(4)
        fSz = 1600
    elif feat_arch == 'ResNet18':
        from archs.cl_archs import ResNet18
        feat_extractor = ResNet18()
        fSz = 512
    elif feat_arch == 'ResNet18_224':
        from archs.cl_archs import ResNet18_224
        feat_extractor = ResNet18_224()
        fSz = 512
    elif feat_arch == 'ResNet10':
        from archs.cl_archs import ResNet10
        feat_extractor = ResNet10()
        fSz = 512
    elif feat_arch == 'ResNet10_224':
        from archs.cl_archs import ResNet10_224
        feat_extractor = ResNet10_224()
        fSz = 512
    elif feat_arch == 'ResNet34_224':
        from archs.cl_archs import ResNet34_224
        feat_extractor = ResNet34_224()
        fSz = 512
    elif feat_arch == 'ResNet50_224':
        from archs.cl_archs import ResNet50_224
        feat_extractor = ResNet50_224()
        fSz = 2048
    else:
        raise NotImplementedError('Unrecognized feature architecture: {}'.format(feat_arch))
    return feat_extractor, fSz