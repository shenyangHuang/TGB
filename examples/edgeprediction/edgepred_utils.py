
"""
Utility functions required for the task of link prediction

"""


import numpy as np
import torch
import random
import os




def set_random_seed(seed):
  """
  set random seed
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
