import sys

from typing import Optional, Dict, Any, Tuple
import os
import os.path as osp
import numpy as np
import pandas as pd
import zipfile
import requests
from clint.textui import progress


from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.linkproppred.tkg_negative_sampler import TKGNegativeEdgeSampler
from tgb.linkproppred.thg_negative_sampler import THGNegativeEdgeSampler
from tgb.utils.info import (
    PROJ_DIR, 
    DATA_URL_DICT, 
    DATA_VERSION_DICT, 
    DATA_EVAL_METRIC_DICT, 
    DATA_NS_STRATEGY_DICT,
    BColors
)
from tgb.utils.pre_process import (
    csv_to_pd_data,
    process_node_feat,
    process_node_type,
    csv_to_pd_data_sc,
    csv_to_pd_data_rc,
    load_edgelist_wiki,
    csv_to_tkg_data,
    csv_to_thg_data,
    csv_to_forum_data,
    csv_to_wikidata,
    csv_to_staticdata,
)
from tgb.utils.utils import save_pkl, load_pkl
from tgb.utils.utils import add_inverse_quadruples


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

        if name == "tgbl-flight":
            self.meta_dict["nodefile"] = self.root + "/" + "airport_node_feat.csv"

        if name == "tkgl-wikidata" or name == "tkgl-smallpedia":
            self.meta_dict["staticfile"] = self.root + "/" + self.name + "_static_edgelist.csv"
        
        if "thg" in name:
            self.meta_dict["nodeTypeFile"] = self.root + "/" + self.name + "_nodetype.csv"
        else:
            self.meta_dict["nodeTypeFile"] = None
        
        self.meta_dict["val_ns"] = self.root + "/" + self.name + "_val_ns.pkl"
        self.meta_dict["test_ns"] = self.root + "/" + self.name + "_test_ns.pkl"

        #! version check
        self.version_passed = True
        self._version_check()

        # initialize
        self._node_feat = None
        self._edge_feat = None
        self._full_data = None
        self._train_data = None
        self._val_data = None
        self._test_data = None

        # for tkg and thg
        self._edge_type = None

        #tkgl-wikidata and tkgl-smallpedia only
        self._static_data = None

        # for thg only
        self._node_type = None
        self._node_id = None

        self.download()
        # check if the root directory exists, if not create it
        if osp.isdir(self.root):
            print("Dataset directory is ", self.root)
        else:
            # os.makedirs(self.root)
            raise FileNotFoundError(f"Directory not found at {self.root}")

        if preprocess:
            self.pre_process()

        self.min_dst_idx, self.max_dst_idx = int(self._full_data["destinations"].min()), int(self._full_data["destinations"].max())

        if ('tkg' in self.name):
            if self.name in DATA_NS_STRATEGY_DICT:
                self.ns_sampler = TKGNegativeEdgeSampler(
                    dataset_name=self.name,
                    first_dst_id=self.min_dst_idx,
                    last_dst_id=self.max_dst_idx,
                    strategy=DATA_NS_STRATEGY_DICT[self.name],
                    partial_path=self.root + "/" + self.name,
                )
            else:
                raise ValueError(f"Dataset {self.name} negative sampling strategy not found.")
        elif ('thg' in self.name):
            #* need to find the smallest node id of all nodes (regardless of types)
            
            min_node_idx = min(int(self._full_data["sources"].min()), int(self._full_data["destinations"].min()))
            max_node_idx = max(int(self._full_data["sources"].max()), int(self._full_data["destinations"].max()))
            self.ns_sampler = THGNegativeEdgeSampler(
                dataset_name=self.name,
                first_node_id=min_node_idx,
                last_node_id=max_node_idx,
                node_type=self._node_type,
            )
        else:
            self.ns_sampler = NegativeEdgeSampler(
                dataset_name=self.name,
                first_dst_id=self.min_dst_idx,
                last_dst_id=self.max_dst_idx,
            )


    def _version_check(self) -> None:
        r"""Implement Version checks for dataset files
        updates the file names based on the current version number
        prompt the user to download the new version via self.version_passed variable
        """
        if (self.name in DATA_VERSION_DICT):
            version = DATA_VERSION_DICT[self.name]
        else:
            print(f"Dataset {self.name} version number not found.")
            self.version_passed = False
            return None
        
        if (version > 1):
            #* check if current version is outdated
            self.meta_dict["fname"] = self.root + "/" + self.name + "_edgelist_v" + str(int(version)) + ".csv"
            self.meta_dict["nodefile"] = None
            if self.name == "tgbl-flight":
                self.meta_dict["nodefile"] = self.root + "/" + "airport_node_feat_v" + str(int(version)) + ".csv"
            self.meta_dict["val_ns"] = self.root + "/" + self.name + "_val_ns_v" + str(int(version)) + ".pkl"
            self.meta_dict["test_ns"] = self.root + "/" + self.name + "_test_ns_v" + str(int(version)) + ".pkl"
            
            if (not osp.exists(self.meta_dict["fname"])):
                print(f"Dataset {self.name} version {int(version)} not found.")
                print(f"Please download the latest version of the dataset.")
                self.version_passed = False
                return None
        

    def download(self):
        """
        downloads this dataset from url
        check if files are already downloaded
        """
        # check if the file already exists
        if osp.exists(self.meta_dict["fname"]):
            print("raw file found, skipping download")
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
                self.version_passed = True
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
        #* for thg must have nodetypes 
        if self.meta_dict["nodeTypeFile"] is not None:
            if not osp.exists(self.meta_dict["nodeTypeFile"]):
                raise FileNotFoundError(
                    f"File not found at {self.meta_dict['nodeTypeFile']}"
                )


        OUT_DF = self.root + "/" + "ml_{}.pkl".format(self.name)
        OUT_EDGE_FEAT = self.root + "/" + "ml_{}.pkl".format(self.name + "_edge")
        OUT_NODE_ID = self.root + "/" + "ml_{}.pkl".format(self.name + "_nodeid")
        if self.meta_dict["nodefile"] is not None:
            OUT_NODE_FEAT = self.root + "/" + "ml_{}.pkl".format(self.name + "_node")
        if self.meta_dict["nodeTypeFile"] is not None:
            OUT_NODE_TYPE = self.root + "/" + "ml_{}.pkl".format(self.name + "_nodeType")

        if (osp.exists(OUT_DF)) and (self.version_passed is True):
            print("loading processed file")
            df = pd.read_pickle(OUT_DF)
            edge_feat = load_pkl(OUT_EDGE_FEAT)
            if (self.name == "tkgl-wikidata") or (self.name == "tkgl-smallpedia"):
                node_id = load_pkl(OUT_NODE_ID)
                self._node_id = node_id
            if self.meta_dict["nodefile"] is not None:
                node_feat = load_pkl(OUT_NODE_FEAT)
            if self.meta_dict["nodeTypeFile"] is not None:
                node_type = load_pkl(OUT_NODE_TYPE)
                self._node_type = node_type

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
            elif self.name == "tgbl-subreddit":
                df, edge_feat, node_ids = load_edgelist_wiki(self.meta_dict["fname"])
            elif self.name == "tgbl-uci":
                df, edge_feat, node_ids = load_edgelist_wiki(self.meta_dict["fname"])
            elif self.name == "tgbl-enron":
                df, edge_feat, node_ids = load_edgelist_wiki(self.meta_dict["fname"])
            elif self.name == "tgbl-lastfm":
                df, edge_feat, node_ids = load_edgelist_wiki(self.meta_dict["fname"])
            elif self.name == "tkgl-polecat":
                df, edge_feat, node_ids = csv_to_tkg_data(self.meta_dict["fname"])
            elif self.name == "tkgl-icews":
                df, edge_feat, node_ids = csv_to_tkg_data(self.meta_dict["fname"])
            elif self.name == "tkgl-yago":
                df, edge_feat, node_ids = csv_to_tkg_data(self.meta_dict["fname"])
            elif self.name == "tkgl-wikidata":
                df, edge_feat, node_ids = csv_to_wikidata(self.meta_dict["fname"])
                save_pkl(node_ids, OUT_NODE_ID)
                self._node_id = node_ids
            elif self.name == "tkgl-smallpedia":
                df, edge_feat, node_ids = csv_to_wikidata(self.meta_dict["fname"])
                save_pkl(node_ids, OUT_NODE_ID)
                self._node_id = node_ids
            elif self.name == "thgl-myket":
                df, edge_feat, node_ids = csv_to_thg_data(self.meta_dict["fname"])
            elif self.name == "thgl-github":
                df, edge_feat, node_ids = csv_to_thg_data(self.meta_dict["fname"])
            elif self.name == "thgl-forum":
                df, edge_feat, node_ids = csv_to_forum_data(self.meta_dict["fname"])
            elif self.name == "thgl-software":
                df, edge_feat, node_ids = csv_to_thg_data(self.meta_dict["fname"])
            else:
                raise ValueError(f"Dataset {self.name} not found.")

            save_pkl(edge_feat, OUT_EDGE_FEAT)
            df.to_pickle(OUT_DF)
            if self.meta_dict["nodefile"] is not None:
                node_feat = process_node_feat(self.meta_dict["nodefile"], node_ids)
                save_pkl(node_feat, OUT_NODE_FEAT)
            if self.meta_dict["nodeTypeFile"] is not None:
                node_type = process_node_type(self.meta_dict["nodeTypeFile"], node_ids)
                save_pkl(node_type, OUT_NODE_TYPE)
                #? do not return node_type, simply set it
                self._node_type = node_type
            

        return df, edge_feat, node_feat

    def pre_process(self):
        """
        Pre-process the dataset and generates the splits, must be run before dataset properties can be accessed
        generates the edge data and different train, val, test splits
        """

        # check if path to file is valid
        df, edge_feat, node_feat = self.generate_processed_files()

        #* design choice, only stores the original edges not the inverse relations on disc
        if ("tkgl" in self.name):
            df = add_inverse_quadruples(df)

        sources = np.array(df["u"])
        destinations = np.array(df["i"])
        timestamps = np.array(df["ts"])
        edge_idxs = np.array(df["idx"])
        weights = np.array(df["w"])
        edge_label = np.ones(len(df))  # should be 1 for all pos edges
        if (self.name == "tgbl-coin") or (self.name == "tgbl-review"):
            self._edge_feat = weights.reshape(-1,1)
        elif (self.name == "tgbl-comment"):
            self._edge_feat = np.concatenate((edge_feat, weights.reshape(-1,1)), axis=1)
        else:
            self._edge_feat = edge_feat
        self._node_feat = node_feat

        full_data = {
            "sources": sources.astype(int),
            "destinations": destinations.astype(int),
            "timestamps": timestamps.astype(int),
            "edge_idxs": edge_idxs,
            "edge_feat": self._edge_feat,
            "w": weights,
            "edge_label": edge_label,
        }

        #* for tkg and thg
        if ("edge_type" in df):
            edge_type = np.array(df["edge_type"]).astype(int)
            self._edge_type = edge_type
            full_data["edge_type"] = edge_type

        self._full_data = full_data

        if ("yago" in self.name):
            _train_mask, _val_mask, _test_mask = self.generate_splits(full_data, val_ratio=0.1, test_ratio=0.10) #99) #val_ratio=0.097, test_ratio=0.099)
        else:
            _train_mask, _val_mask, _test_mask = self.generate_splits(full_data, val_ratio=0.15, test_ratio=0.15)
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
    
    def preprocess_static_edges(self):
        """
        Pre-process the static edges of the dataset
        """
        if ("staticfile" in self.meta_dict):
            OUT_DF = self.root + "/" + "ml_{}.pkl".format(self.name + "_static")
            if (osp.exists(OUT_DF)) and (self.version_passed is True):
                print("loading processed file")
                static_dict = load_pkl(OUT_DF)
                self._static_data = static_dict
            else:
                print("file not processed, generating processed file")
                static_dict, node_ids =  csv_to_staticdata(self.meta_dict["staticfile"], self._node_id)
                save_pkl(static_dict, OUT_DF)
                self._static_data = static_dict
        else:
            print ("static edges are only for tkgl-wikidata and tkgl-smallpedia datasets")

    
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
            fname=self.meta_dict["val_ns"], split_mode="val"
        )

    def load_test_ns(self) -> None:
        r"""
        load the negative samples for the test set
        """
        self.ns_sampler.load_eval_set(
            fname=self.meta_dict["test_ns"], split_mode="test"
        )

    @property
    def num_nodes(self) -> int:
        r"""
        Returns the total number of unique nodes in the dataset 
        Returns:
            num_nodes: int, the number of unique nodes
        """
        src = self._full_data["sources"]
        dst = self._full_data["destinations"]
        all_nodes = np.concatenate((src, dst), axis=0)
        uniq_nodes = np.unique(all_nodes, axis=0)
        return uniq_nodes.shape[0]
    

    @property
    def num_edges(self) -> int:
        r"""
        Returns the total number of edges in the dataset
        Returns:
            num_edges: int, the number of edges
        """
        src = self._full_data["sources"]
        return src.shape[0]
    

    @property
    def num_rels(self) -> int:
        r"""
        Returns the number of relation types in the dataset
        Returns:
            num_rels: int, the number of relation types
        """
        #* if it is a homogenous graph
        if ("edge_type" not in self._full_data):
            return 1
        else:
            return np.unique(self._full_data["edge_type"]).shape[0]

    @property
    def node_feat(self) -> Optional[np.ndarray]:
        r"""
        Returns the node features of the dataset with dim [N, feat_dim]
        Returns:
            node_feat: np.ndarray, [N, feat_dim] or None if there is no node feature
        """
        return self._node_feat
    
    @property
    def node_type(self) -> Optional[np.ndarray]:
        r"""
        Returns the node types of the dataset with dim [N], only for temporal heterogeneous graphs
        Returns:
            node_feat: np.ndarray, [N] or None if there is no node feature
        """
        return self._node_type

    @property
    def edge_feat(self) -> Optional[np.ndarray]:
        r"""
        Returns the edge features of the dataset with dim [E, feat_dim]
        Returns:
            edge_feat: np.ndarray, [E, feat_dim] or None if there is no edge feature
        """
        return self._edge_feat
    
    @property
    def edge_type(self) -> Optional[np.ndarray]:
        r"""
        Returns the edge types of the dataset with dim [E, 1], only for temporal knowledge graph and temporal heterogeneous graph
        Returns:
            edge_type: np.ndarray, [E, 1] or None if it is not a TKG or THG
        """
        return self._edge_type
    
    @property
    def static_data(self) -> Optional[np.ndarray]:
        r"""
        Returns the static edges related to this dataset, applies for tkgl-wikidata and tkgl-smallpedia, edges are (src, dst, rel_type)
        Returns:
            df: pd.DataFrame {"head": np.ndarray, "tail": np.ndarray, "rel_type": np.ndarray}
        """
        if (self.name == "tkgl-wikidata") or (self.name == "tkgl-smallpedia"):
            self.preprocess_static_edges()
        return self._static_data

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

    name = "tkgl-polecat"
    dataset = LinkPropPredDataset(name=name, root="datasets", preprocess=True)
    dataset.edge_type



    # name = "tgbl-comment" 
    # dataset = LinkPropPredDataset(name=name, root="datasets", preprocess=True)

    # dataset.node_feat
    # dataset.edge_feat  # not the edge weights
    # dataset.full_data
    # dataset.full_data["edge_idxs"]
    # dataset.full_data["sources"]
    # dataset.full_data["destinations"]
    # dataset.full_data["timestamps"]
    # dataset.full_data["edge_label"]


if __name__ == "__main__":
    main()
