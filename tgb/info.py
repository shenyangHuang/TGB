import os.path as osp
r"""
General space to store global information used elsewhere such as url links, evaluation metrics etc.
"""


# from torchmetrics import MeanSquaredError
PROJ_DIR = osp.dirname(osp.abspath(__file__)) + "/"

DATA_URL_DICT ={
    "un_trade": None,
}


DATA_EVAL_METRIC_DICT = {
    "un_trade": None, # maybe torchmetrics https://torchmetrics.readthedocs.io/en/stable/regression/mean_squared_error.html
}