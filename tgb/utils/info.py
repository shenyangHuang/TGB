import os.path as osp
import os
r"""
General space to store global information used elsewhere such as url links, evaluation metrics etc.
"""
PROJ_DIR = osp.dirname(osp.abspath(os.path.join(__file__, os.pardir))) + "/"


class BColors:
    """
    A class to change the colors of the strings.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

DATA_URL_DICT ={
    "un_trade": "https://object-arbutus.cloud.computecanada.ca/tgb/un_trade.zip",
    "lastfmgenre": "https://object-arbutus.cloud.computecanada.ca/tgb/lastfmgenre.zip",
    "opensky": "https://object-arbutus.cloud.computecanada.ca/tgb/opensky.zip",
    "MAG": "https://object-arbutus.cloud.computecanada.ca/tgb/mag_cs.zip",
    "stablecoin": "https://object-arbutus.cloud.computecanada.ca/tgb/stablecoin.zip",
    "redditcomments": "https://object-arbutus.cloud.computecanada.ca/tgb/redditcomments.zip",
    "subreddits": "https://object-arbutus.cloud.computecanada.ca/tgb/subreddits.zip",
    "amazonreview": "https://object-arbutus.cloud.computecanada.ca/tgb/amazonreview.zip",
}


DATA_EVAL_METRIC_DICT = {
    "un_trade": None, # maybe torchmetrics https://torchmetrics.readthedocs.io/en/stable/regression/mean_squared_error.html
}

# "subreddits": 2220,

DATA_NUM_CLASSES = {
    "lastfmgenre": 513,
    "subreddits": 2221,
    "un_trade": 255,
}