import os
from os import listdir
from os.path import isdir, join
import shutil


dataset_list = ['train','val','test']
for dataset in dataset_list:
    os.makedirs(join('CUB', dataset))

data_path = 'CUB_200_2011/images'
folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()

for i, folder in enumerate(folder_list):
    # use the split proposed in https://github.com/wyharveychen/CloserLookFewShot
    if i % 2 == 0:
        dataset = 'train'
    elif i % 4 == 1:
        dataset = 'val'
    else:
        dataset = 'test'
    shutil.move(join(data_path, folder), join('CUB', dataset, folder))