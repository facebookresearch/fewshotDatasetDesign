import numpy as np
from nltk.corpus import wordnet

def get_parent_wnid(wnid):
    """ Given wnid get parent wnid """
    return 'n'+str(synset_to_wnid(wnid_to_synset(wnid).hypernyms()[0])).zfill(8)

def synset_to_wnid(synset):
    """ Converts synset to wnid. Synset is a wordnet node object"""
    return synset.offset()

def wnid_to_synset(wnid):
    """ Converts wnid back into synset (only nouns) """
    if type(wnid) is str:
        if wnid[0] == 'n':
            wnid = int(wnid[1:])
        else:
            wnid = int(wnid)
    return wordnet.synset_from_pos_and_offset('n', wnid)

def wnid_to_noun(wnid):
    """ Converts wnid to noun (chooses the first definition)"""
    return wnid_to_synset(wnid).lemmas()[0].name().replace('_', ' ')

def is_hyponym(syn1, syn2):
    """ Checks if syn1 is a child of syn2 """
    while syn1 != syn2:
        hypernyms = syn1.hypernyms()
        if len(hypernyms) == 0:
            return False
        syn1 = hypernyms[0]
    return True

def wnid_depth(wnid):
    """ Computes the depth of the given wnid """
    syn = wnid_to_synset(wnid)
    depth = 0
    hyper = syn.hypernyms()
    while len(hyper) != 0:
        depth += 1
        # move up, choose the first parent. Sometimes there are more than 1 parent.
        syn = syn.hypernyms()[0]
        hyper = syn.hypernyms()
    return depth

def read_synset_file(synset_words_path):
    """
    Reads synset.txt or synset_words.txt
    Returns
        wnid_array - list of wnid strings
    """
    wnid_array  = [line.split(' ')[0] for line in open(synset_words_path, 'r')]
    return wnid_array

def read_txt_file(txt_file):
    """
    Reads imagenet train.txt and val.txt files
    """
    return [line for line in open(txt_file, 'r')]

def wnid_statistics(wnid_arr):
    """
    Computes some simple statistics on a list of wnid
    Args
        wnid_arr - An array of wnids
    Returns
        stats - Dictionary which summarizes the computed statistics
    """
    stats = {}
    depth_arr = [wnid_depth(w) for w in wnid_arr]
    stats['depth_arr'] = depth_arr
    stats['min_depth'], stats['max_depth'] = np.min(depth_arr), np.max(depth_arr)
    return stats