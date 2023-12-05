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

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

DATA_URL_DICT = {
    "tgbl-wiki":"https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-wiki-v2.zip", #"https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-wiki.zip", #v1
    "tgbl-review": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-review-v2.zip", #"https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-review.zip", #v1
    "tgbl-coin": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-coin-v2.zip", #"https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-coin.zip",
    "tgbl-flight": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-flight-v2.zip", #"tgbl-flight": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-flight_edgelist_v2_ts.zip",
    "tgbl-comment": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-comment.zip",
    "tgbn-trade": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-trade.zip",
    "tgbn-genre": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-genre.zip",
    "tgbn-reddit": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-reddit.zip",
    "tgbn-token": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-token.zip",
}



DATA_VERSION_DICT = {
    "tgbl-wiki": 2,  
    "tgbl-review": 2,
    "tgbl-coin": 2,
    "tgbl-comment": 1,
    "tgbl-flight": 2,
    "tgbn-trade": 1,
    "tgbn-genre": 1,
    "tgbn-reddit": 1,
    "tgbn-token": 1,
}

#"tgbl-flight-v1": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-flight.zip",

DATA_EVAL_METRIC_DICT = {
    "tgbl-wiki": "mrr",
    "tgbl-review": "mrr",
    "tgbl-coin": "mrr",
    "tgbl-comment": "mrr",
    "tgbl-flight": "mrr",
    "tgbn-trade": "ndcg",
    "tgbn-genre": "ndcg",
    "tgbn-reddit": "ndcg",
    "tgbn-token": "ndcg",
}


DATA_NUM_CLASSES = {
    "tgbn-trade": 255,
    "tgbn-genre": 513,
    "tgbn-reddit": 698,
    "tgbn-token": 1001,
}
