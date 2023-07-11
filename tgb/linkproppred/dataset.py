from typing import Optional, Dict, Any, Tuple
import os
import os.path as osp
import numpy as np
import pandas as pd
import zipfile
import requests
from clint.textui import progress

from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.utils.info import PROJ_DIR, DATA_URL_DICT, DATA_EVAL_METRIC_DICT, BColors
from tgb.utils.pre_process import (
    csv_to_pd_data,
    process_node_feat,
    csv_to_pd_data_sc,
    csv_to_pd_data_rc,
    load_edgelist_wiki,
)
from tgb.utils.utils import save_pkl, load_pkl


class LinkPropPredDataset(object):
    def __init__(
        self,
        name: str,
        root: Optional[str] = "datasets",
        meta_dict: Optional[dict] = None,
        preprocess: Optional[bool] = True,
    ):
        r"""Dataset class for link prediction dataset. Stores meta information about each dataset such as evaluation metrics etc.
        also automatically pre-processes the dataset.
        Args:
            name: name of the dataset
            root: root directory to store the dataset folder
            meta_dict: dictionary containing meta information about the dataset, should contain key 'dir_name' which is the name of the dataset folder
            preprocess: whether to pre-process the dataset
        """
        self.name = name  ## original name
        # check if dataset url exist
        if self.name in DATA_URL_DICT:
            self.url = DATA_URL_DICT[self.name]
        else:
            self.url = None
            print(f"Dataset {self.name} url not found, download not supported yet.")

        
        # check if the evaluatioin metric are specified
        if self.name in DATA_EVAL_METRIC_DICT:
            self.metric = DATA_EVAL_METRIC_DICT[self.name]
        else:
            self.metric = None
            print(
                f"Dataset {self.name} default evaluation metric not found, it is not supported yet."
            )


        root = PROJ_DIR + root

        if meta_dict is None:
            self.dir_name = "_".join(name.split("-"))  ## replace hyphen with underline
            meta_dict = {"dir_name": self.dir_name}
        else:
            self.dir_name = meta_dict["dir_name"]
        self.root = osp.join(root, self.dir_name)
        self.meta_dict = meta_dict
        if "fname" not in self.meta_dict:
            self.meta_dict["fname"] = self.root + "/" + self.name + "_edgelist.csv"
            self.meta_dict["nodefile"] = None

        # TODO update the logic here to load the filenames from info.py
        if name == "tgbl-flight":
            self.meta_dict["nodefile"] = self.root + "/" + "airport_node_feat.csv"

        # initialize
        self._node_feat = None
        self._edge_feat = None
        self._full_data = None
        self._train_data = None
        self._val_data = None
        self._test_data = None

        self.download()
        # check if the root directory exists, if not create it
        if osp.isdir(self.root):
            print("Dataset directory is ", self.root)
        else:
            # os.makedirs(self.root)
            raise FileNotFoundError(f"Directory not found at {self.root}")

        if preprocess:
            self.pre_process()

        self.ns_sampler = NegativeEdgeSampler(
            dataset_name=self.name, strategy="hist_rnd"
        )

    def download(self):
        """
        downloads this dataset from url
        check if files are already downloaded
        """
        # check if the file already exists
        if osp.exists(self.meta_dict["fname"]):
            print("file found, skipping download")
            return

        inp = input(
            "Will you download the dataset(s) now? (y/N)\n"
        ).lower()  # ask if the user wants to download the dataset

        if inp == "y":
            print(
                f"{BColors.WARNING}Download started, this might take a while . . . {BColors.ENDC}"
            )
            print(f"Dataset title: {self.name}")

            if self.url is None:
                raise Exception("Dataset url not found, download not supported yet.")
            else:
                r = requests.get(self.url, stream=True)
                # download_dir = self.root + "/" + "download"
                if osp.isdir(self.root):
                    print("Dataset directory is ", self.root)
                else:
                    os.makedirs(self.root)

                path_download = self.root + "/" + self.name + ".zip"
                with open(path_download, "wb") as f:
                    total_length = int(r.headers.get("content-length"))
                    for chunk in progress.bar(
                        r.iter_content(chunk_size=1024),
                        expected_size=(total_length / 1024) + 1,
                    ):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                # for unzipping the file
                with zipfile.ZipFile(path_download, "r") as zip_ref:
                    zip_ref.extractall(self.root)
                print(f"{BColors.OKGREEN}Download completed {BColors.ENDC}")
        else:
            raise Exception(
                BColors.FAIL + "Data not found error, download " + self.name + " failed"
            )

    def generate_processed_files(self) -> pd.DataFrame:
        r"""
        turns raw data .csv file into a pandas data frame, stored on disc if not already
        Returns:
            df: pandas data frame
        """
        node_feat = None
        if not osp.exists(self.meta_dict["fname"]):
            raise FileNotFoundError(f"File not found at {self.meta_dict['fname']}")

        if self.meta_dict["nodefile"] is not None:
            if not osp.exists(self.meta_dict["nodefile"]):
                raise FileNotFoundError(
                    f"File not found at {self.meta_dict['nodefile']}"
                )
        OUT_DF = self.root + "/" + "ml_{}.pkl".format(self.name)
        OUT_EDGE_FEAT = self.root + "/" + "ml_{}.pkl".format(self.name + "_edge")
        if self.meta_dict["nodefile"] is not None:
            OUT_NODE_FEAT = self.root + "/" + "ml_{}.pkl".format(self.name + "_node")

        if osp.exists(OUT_DF):
            print("loading processed file")
            df = pd.read_pickle(OUT_DF)
            edge_feat = load_pkl(OUT_EDGE_FEAT)
            if self.meta_dict["nodefile"] is not None:
                node_feat = load_pkl(OUT_NODE_FEAT)

        else:
            print("file not processed, generating processed file")
            if self.name == "tgbl-flight":
                df, edge_feat, node_ids = csv_to_pd_data(self.meta_dict["fname"])
            elif self.name == "tgbl-coin":
                df, edge_feat, node_ids = csv_to_pd_data_sc(self.meta_dict["fname"])
            elif self.name == "tgbl-comment":
                df, edge_feat, node_ids = csv_to_pd_data_rc(self.meta_dict["fname"])
            elif self.name == "tgbl-review":
                df, edge_feat, node_ids = csv_to_pd_data_sc(self.meta_dict["fname"])
            elif self.name == "tgbl-wiki":
                df, edge_feat, node_ids = load_edgelist_wiki(self.meta_dict["fname"])

            save_pkl(edge_feat, OUT_EDGE_FEAT)
            df.to_pickle(OUT_DF)
            if self.meta_dict["nodefile"] is not None:
                node_feat = process_node_feat(self.meta_dict["nodefile"], node_ids)
                save_pkl(node_feat, OUT_NODE_FEAT)

        return df, edge_feat, node_feat

    def pre_process(self):
        """
        Pre-process the dataset and generates the splits, must be run before dataset properties can be accessed
        generates the edge data and different train, val, test splits
        """
        # TODO for link prediction, y =1 because these are all true edges, edge feat = weight + edge feat

        # check if path to file is valid
        df, edge_feat, node_feat = self.generate_processed_files()
        sources = np.array(df["u"])
        destinations = np.array(df["i"])
        timestamps = np.array(df["ts"])
        edge_idxs = np.array(df["idx"])
        weights = np.array(df["w"])

        edge_label = np.ones(len(df))  # should be 1 for all pos edges
        # self._edge_feat = edge_feat + weights.reshape(
        #     -1, 1
        # )  # reshape weights as feature if available #! to remove
        self._edge_feat = edge_feat
        self._node_feat = node_feat

        full_data = {
            "sources": sources,
            "destinations": destinations,
            "timestamps": timestamps,
            "edge_idxs": edge_idxs,
            "edge_feat": edge_feat,
            "w": weights,
            "edge_label": edge_label,
        }
        self._full_data = full_data
        _train_mask, _val_mask, _test_mask = self.generate_splits(full_data)
        self._train_mask = _train_mask
        self._val_mask = _val_mask
        self._test_mask = _test_mask

    def generate_splits(
        self,
        full_data: Dict[str, Any],
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        r"""Generates train, validation, and test splits from the full dataset
        Args:
            full_data: dictionary containing the full dataset
            val_ratio: ratio of validation data
            test_ratio: ratio of test data
        Returns:
            train_data: dictionary containing the training dataset
            val_data: dictionary containing the validation dataset
            test_data: dictionary containing the test dataset
        """
        val_time, test_time = list(
            np.quantile(
                full_data["timestamps"],
                [(1 - val_ratio - test_ratio), (1 - test_ratio)],
            )
        )
        timestamps = full_data["timestamps"]

        train_mask = timestamps <= val_time
        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
        test_mask = timestamps > test_time

        return train_mask, val_mask, test_mask
    
    @property
    def eval_metric(self) -> str:
        """
        the official evaluation metric for the dataset, loaded from info.py
        Returns:
            eval_metric: str, the evaluation metric
        """
        return self.metric

    @property
    def negative_sampler(self) -> NegativeEdgeSampler:
        r"""
        Returns the negative sampler of the dataset, will load negative samples from disc
        Returns:
            negative_sampler: NegativeEdgeSampler
        """
        return self.ns_sampler

    def load_val_ns(self) -> None:
        r"""
        load the negative samples for the validation set
        """
        self.ns_sampler.load_eval_set(
            fname=self.root + "/" + self.name + "_val_ns.pkl", split_mode="val"
        )

    def load_test_ns(self) -> None:
        r"""
        load the negative samples for the test set
        """
        self.ns_sampler.load_eval_set(
            fname=self.root + "/" + self.name + "_test_ns.pkl", split_mode="test"
        )

    @property
    def node_feat(self) -> Optional[np.ndarray]:
        r"""
        Returns the node features of the dataset with dim [N, feat_dim]
        Returns:
            node_feat: np.ndarray, [N, feat_dim] or None if there is no node feature
        """
        return self._node_feat

    @property
    def edge_feat(self) -> Optional[np.ndarray]:
        r"""
        Returns the edge features of the dataset with dim [E, feat_dim]
        Returns:
            edge_feat: np.ndarray, [E, feat_dim] or None if there is no edge feature
        """
        return self._edge_feat

    @property
    def full_data(self) -> Dict[str, Any]:
        r"""
        the full data of the dataset as a dictionary with keys: 'sources', 'destinations', 'timestamps', 'edge_idxs', 'edge_feat', 'w', 'edge_label',

        Returns:
            full_data: Dict[str, Any]
        """
        if self._full_data is None:
            raise ValueError(
                "dataset has not been processed yet, please call pre_process() first"
            )
        return self._full_data

    @property
    def train_mask(self) -> np.ndarray:
        r"""
        Returns the train mask of the dataset
        Returns:
            train_mask: training masks
        """
        if self._train_mask is None:
            raise ValueError("training split hasn't been loaded")
        return self._train_mask

    @property
    def val_mask(self) -> np.ndarray:
        r"""
        Returns the validation mask of the dataset
        Returns:
            val_mask: Dict[str, Any]
        """
        if self._val_mask is None:
            raise ValueError("validation split hasn't been loaded")
        return self._val_mask

    @property
    def test_mask(self) -> np.ndarray:
        r"""
        Returns the test mask of the dataset:
        Returns:
            test_mask: Dict[str, Any]
        """
        if self._test_mask is None:
            raise ValueError("test split hasn't been loaded")
        return self._test_mask


def main():
    name = "tgbl-comment" 
    dataset = LinkPropPredDataset(name=name, root="datasets", preprocess=True)

    dataset.node_feat
    dataset.edge_feat  # not the edge weights
    dataset.full_data
    dataset.full_data["edge_idxs"]
    dataset.full_data["sources"]
    dataset.full_data["destinations"]
    dataset.full_data["timestamps"]
    dataset.full_data["edge_label"]


if __name__ == "__main__":
    main()
