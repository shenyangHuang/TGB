import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ndcg_score
import math

from tgb.utils.info import DATA_EVAL_METRIC_DICT

try:
    import torch
except ImportError:
    torch = None


class Evaluator(object):
    """Evaluator for Node Property Prediction"""

    def __init__(self, name: str):
        r"""
        Parameters:
            name: name of the dataset
        """
        self.name = name
        self.valid_metric_list = ["mse", "rmse", "ndcg"]
        if self.name not in DATA_EVAL_METRIC_DICT:
            raise NotImplementedError("Dataset not supported")

    def _parse_and_check_input(self, input_dict):
        """
        check whether the input has the required format
        Parametrers:
            -input_dict: a dictionary containing "y_true", "y_pred", and "eval_metric"

            note: "eval_metric" should be a list including one or more of the followin metrics:
                    ["mse"]
        """
        # valid_metric_list = ['ap', 'au_roc_score', 'au_pr_score', 'acc', 'prec', 'rec', 'f1']

        if "eval_metric" not in input_dict:
            raise RuntimeError("Missing key of eval_metric")

        for eval_metric in input_dict["eval_metric"]:
            if eval_metric in self.valid_metric_list:
                if "y_true" not in input_dict:
                    raise RuntimeError("Missing key of y_true")
                if "y_pred" not in input_dict:
                    raise RuntimeError("Missing key of y_pred")

                y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]

                # converting to numpy on cpu
                if torch is not None and isinstance(y_true, torch.Tensor):
                    y_true = y_true.detach().cpu().numpy()
                if torch is not None and isinstance(y_pred, torch.Tensor):
                    y_pred = y_pred.detach().cpu().numpy()

                # check type and shape
                if not isinstance(y_true, np.ndarray) or not isinstance(
                    y_pred, np.ndarray
                ):
                    raise RuntimeError(
                        "Arguments to Evaluator need to be either numpy ndarray or torch tensor!"
                    )

                if not y_true.shape == y_pred.shape:
                    raise RuntimeError("Shape of y_true and y_pred must be the same!")

            else:
                print(
                    "ERROR: The evaluation metric should be in:", self.valid_metric_list
                )
                raise ValueError("Undefined eval metric %s " % (eval_metric))
        self.eval_metric = input_dict["eval_metric"]

        return y_true, y_pred

    def _compute_metrics(self, y_true, y_pred):
        """
        compute the performance metrics for the given true labels and prediction probabilities
        Parameters:
            -y_true: actual true labels
            -y_pred: predicted probabilities
        """
        perf_dict = {}
        for eval_metric in self.eval_metric:
            if eval_metric == "mse":
                perf_dict = {
                    "mse": mean_squared_error(y_true, y_pred),
                    "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
                }
            elif eval_metric == "ndcg":
                k = 10
                perf_dict = {"ndcg": ndcg_score(y_true, y_pred, k=k)}
        return perf_dict

    def eval(self, input_dict, verbose=False):
        """
        evaluation for edge regression task
        """
        y_true, y_pred = self._parse_and_check_input(input_dict)
        perf_dict = self._compute_metrics(y_true, y_pred)

        if verbose:
            print("INFO: Evaluation Results:")
            for eval_metric in input_dict["eval_metric"]:
                print(f"\t>>> {eval_metric}: {perf_dict[eval_metric]:.4f}")
        return perf_dict

    @property
    def expected_input_format(self):
        desc = "==== Expected input format of Evaluator for {}\n".format(self.name)
        if "mse" in self.valid_metric_list:
            desc += "{'y_pred': y_pred}\n"
            desc += "- y_pred: numpy ndarray or torch tensor of shape (num_edges, ). Torch tensor on GPU is recommended for efficiency.\n"
            desc += "y_pred is the predicted weight for edges.\n"
        else:
            raise ValueError("Undefined eval metric %s" % (self.eval_metric))
        return desc

    @property
    def expected_output_format(self):
        desc = "==== Expected output format of Evaluator for {}\n".format(self.name)
        if "mse" in self.valid_metric_list:
            desc += "{'mse': mse\n"
            desc += "- mse (float): mse score\n"
        else:
            raise ValueError("Undefined eval metric %s" % (self.eval_metric))
        return desc


def main():
    """
    simple test for evaluator
    """
    name = "tgbn-trade"
    evaluator = Evaluator(name=name)
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    input_dict = {"y_true": y_true, "y_pred": y_pred, "eval_metric": ["mse"]}

    result_dict = evaluator.eval(input_dict)
    print(result_dict)


if __name__ == "__main__":
    main()
