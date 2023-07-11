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
    "tgbl-wiki": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-wiki.zip",
    "tgbl-review": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-review.zip",
    "tgbl-coin": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-coin.zip",
    "tgbl-flight": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-flight.zip",
    "tgbl-comment": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-comment.zip",
    "tgbn-trade": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-trade.zip",
    "tgbn-genre": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-genre.zip",
    "tgbn-reddit": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-reddit.zip",
}

#"https://object-arbutus.cloud.computecanada.ca/tgb/wiki_neg.zip" #for all negative samples of the wiki dataset
#"https://object-arbutus.cloud.computecanada.ca/tgb/review_ns100.zip" #for 100 ns samples in review

DATA_EVAL_METRIC_DICT = {
    "tgbl-wiki": "mrr",
    "tgbl-review": "mrr",
    "tgbl-coin": "mrr",
    "tgbl-comment": "mrr",
    "tgbl-flight": "mrr",
    "tgbn-trade": "ndcg",
    "tgbn-genre": "ndcg",
    "tgbn-reddit": "ndcg",
}


DATA_NUM_CLASSES = {
    "tgbn-trade": 255,
    "tgbn-genre": 513,
    "tgbn-reddit": 698,
}
