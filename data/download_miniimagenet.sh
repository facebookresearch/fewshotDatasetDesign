# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

wget https://www.dropbox.com/s/a2a0bll17f5dvhr/Mini-ImageNet.zip?dl=0
mv Mini-ImageNet.zip?dl=0 Mini-ImageNet.zip
unzip Mini-ImageNet.zip
rm Mini-ImageNet.zip
rm -r Mini-ImageNet/train_val Mini-ImageNet/train_test
mv  Mini-ImageNet/train_train Mini-ImageNet/train 