# Code from - https://github.com/wyharveychen/CloserLookFewShot

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

save_dir                    = dir_path
data_dir = {}
data_dir['CUB']             = os.path.join(dir_path,'filelists/CUB/' )
data_dir['miniImagenet']    = os.path.join(dir_path,'filelists/miniImagenet/' )
data_dir['omniglot']        = os.path.join(dir_path,'filelists/omniglot/' )
data_dir['emnist']          = os.path.join(dir_path,'filelists/emnist/')
