
import numpy as np
import pandas as pd
import random 
import os
import os.path as osp
import pickle


#import torch
def save_pkl(obj, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
		
def load_pkl(fname):
	with open(fname, 'rb') as handle:
		return pickle.load(handle)


def set_random_seed(seed):
	"""
	set random seed
	"""
	np.random.seed(seed)
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


	
        