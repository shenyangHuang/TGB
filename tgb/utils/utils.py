
import numpy as np
import pandas as pd
import random 
import os
import os.path as osp

#import torch

def set_random_seed(seed):
	"""
	set random seed
	"""
	np.random.seed(seed)
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	

	
        