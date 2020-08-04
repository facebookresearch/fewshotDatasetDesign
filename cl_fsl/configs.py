import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

save_dir                    = dir_path
data_dir = {}
data_dir['CUB']             = os.path.join(dir_path,'filelists/CUB/' )
data_dir['miniImagenet']    = os.path.join(dir_path,'filelists/miniImagenet/' )
data_dir['omniglot']        = os.path.join(dir_path,'filelists/omniglot/' )
data_dir['emnist']          = os.path.join(dir_path,'filelists/emnist/')
