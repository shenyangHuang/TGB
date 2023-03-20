
import numpy as np
import pandas as pd
import random 
import os
#import torch



def set_random_seed(seed):
  """
  set random seed
  """
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)


# def set_torch_seed(seed):
#   """
#   set random seed
#   """
#   # TODO add back torch dependency
#   torch.manual_seed(seed)
#   torch.cuda.manual_seed_all(seed)
#   torch.backends.cudnn.deterministic = True
#   torch.backends.cudnn.benchmark = False

