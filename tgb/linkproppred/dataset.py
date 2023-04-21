from typing import Optional, Dict, Any, Tuple
import os
import os.path as osp
import numpy as np
import pandas as pd
import shutil
import zipfile
import requests
from clint.textui import progress

from tgb.utils.info import PROJ_DIR, DATA_URL_DICT, BColors
from tgb.utils.pre_process import _to_pd_data, reindex, csv_to_pd_data, process_node_feat
from tgb.utils.utils import save_pkl, load_pkl


class LinkPropPredDataset(object):
    def __init__(
        self, 
        name: str, 
        root: Optional[str] = 'datasets', 
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
        self.name = name ## original name
        #check if dataset url exist 
        if (self.name in DATA_URL_DICT):
            self.url = DATA_URL_DICT[self.name]
        else:
            self.url = None
            print (f"Dataset {self.name} url not found, download not supported yet.")

        root = PROJ_DIR + root

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) ## replace hyphen with underline
            meta_dict = {'dir_name': self.dir_name}
        else:
            self.dir_name = meta_dict['dir_name']
        self.root = osp.join(root, self.dir_name)
        self.meta_dict = meta_dict
        if ("fname" not in self.meta_dict):
            self.meta_dict["fname"] = self.root + "/" + self.name + ".csv"

        #TODO update the logic here to load the filenames from info.py
        if (name == "opensky"):
            self.meta_dict["fname"] = self.root + "/" + 'opensky_edgelist.csv'
            self.meta_dict["edgefile"] = self.root + "/" + 'opensky_edgelist.csv'
            self.meta_dict["nodefile"] = self.root + "/" + 'airport_node_feat.csv'

        #initialize
        self._node_feat = None
        self._edge_feat = None
        self._full_data = None
        self._train_data = None
        self._val_data = None
        self._test_data = None

        self.download()
        #check if the root directory exists, if not create it
        if osp.isdir(self.root):
            print("Dataset directory is ", self.root)
        else:
            #os.makedirs(self.root)
            raise FileNotFoundError(f"Directory not found at {self.root}")

        if preprocess:
            self.pre_process()


    def download(self):
        """
        downloads this dataset from url
        check if files are already downloaded
        """
        #check if the file already exists
        if osp.exists(self.meta_dict['fname']):
            print ("file found, skipping download")
            return

        inp = input('Will you download the dataset(s) now? (y/N)\n').lower() #ask if the user wants to download the dataset

        if inp == 'y':
            print(f"{BColors.WARNING}Download started, this might take a while . . . {BColors.ENDC}")
            print(f"Dataset title: {self.name}")

            if (self.url is None):
                raise Exception("Dataset url not found, download not supported yet.")
            else:
                r = requests.get(self.url, stream=True)
                #download_dir = self.root + "/" + "download"
                if osp.isdir(self.root):
                    print("Dataset directory is ", self.root)
                else:
                    os.makedirs(self.root)

                path_download = self.root + "/" + self.name + ".zip"
                with open(path_download, 'wb') as f:
                    total_length = int(r.headers.get('content-length'))
                    for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                        if chunk:
                            f.write(chunk)
                            f.flush()            
                #for unzipping the file
                with zipfile.ZipFile(path_download, 'r') as zip_ref:
                    zip_ref.extractall(self.root)
                print(f"{BColors.OKGREEN}Download completed {BColors.ENDC}")
        else:
            raise Exception(
                BColors.FAIL + "Data not found error, download " + self.name + " failed")

    def generate_processed_files(self) -> pd.DataFrame:
        r"""
        turns raw data .csv file into a pandas data frame, stored on disc if not already
        Returns:
            df: pandas data frame
        """
        if (not osp.exists(self.meta_dict['edgefile'])):
            raise FileNotFoundError(f"File not found at {self.meta_dict['edgefile']}")
        if (not osp.exists(self.meta_dict['nodefile'])):
            raise FileNotFoundError(f"File not found at {self.meta_dict['nodefile']}")
        OUT_DF = self.root + '/' + 'ml_{}.pkl'.format(self.name)
        OUT_EDGE_FEAT = self.root + '/' + 'ml_{}.pkl'.format(self.name+"_edge")
        OUT_NODE_FEAT = self.root + '/' + 'ml_{}.pkl'.format(self.name+"_node")

        if osp.exists(OUT_DF):
            print ("loading processed file")
            df = pd.read_pickle(OUT_DF)
            edge_feat = load_pkl(OUT_EDGE_FEAT)
            node_feat = load_pkl(OUT_NODE_FEAT)

        else:
            print ("file not processed, generating processed file")
            df, edge_feat, node_ids = csv_to_pd_data(self.meta_dict['edgefile'])  
            #df = reindex(df, bipartite=False)  #this is simplying shifting the index by 1
            node_feat = process_node_feat(self.meta_dict['nodefile'], node_ids)
            save_pkl(edge_feat, OUT_EDGE_FEAT)
            save_pkl(node_feat, OUT_NODE_FEAT)
            df.to_pickle(OUT_DF)
            node_feat = process_node_feat(self.meta_dict['nodefile'], node_ids)

        return df, edge_feat, node_feat




    def pre_process(self):
        '''
        Pre-process the dataset and generates the splits, must be run before dataset properties can be accessed
        generates self.full_data, self.train_data, self.val_data, self.test_data
        Parameters:
            feat_dim: dimension for feature vectors, padded to 172 with zeros
        '''
        #check if path to file is valid 
        df, edge_feat, node_feat = self.generate_processed_files()
        self._edge_feat = edge_feat
        self._node_feat = node_feat
        sources = np.array(df['u'])
        destinations = np.array(df['i'])
        timestamps = np.array(df['ts'])
        edge_idxs = np.array(df['idx'])
        y = np.array(df['w'])


        full_data = {
            'sources': sources,
            'destinations': destinations,
            'timestamps': timestamps,
            'edge_idxs': edge_idxs,
            'edge_feat': edge_feat,
            'y': y,
        }
        self._full_data = full_data
        _train_data, _val_data, _test_data = self.generate_splits(full_data)
        self._train_data = _train_data
        self._val_data = _val_data
        self._test_data = _test_data



    def generate_splits(self,
                        full_data: Dict[str, Any],
                        val_ratio=0.15, 
                        test_ratio=0.15,
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
        val_time, test_time = list(np.quantile(full_data['timestamps'], [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
        timestamps = full_data['timestamps']
        sources = full_data['sources']
        destinations = full_data['destinations']
        edge_idxs = full_data['edge_idxs']
        edge_feat = full_data['edge_feat']
        y = full_data['y']
        
        train_mask = timestamps <= val_time
        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
        test_mask = timestamps > test_time

        
        train_data = {
            'sources': sources[train_mask],
            'destinations': destinations[train_mask],
            'timestamps': timestamps[train_mask],
            'edge_idxs': edge_idxs[train_mask],
            'edge_feat': edge_feat[train_mask],
            'y': y[train_mask],
        }

        val_data = {
            'sources': sources[val_mask],
            'destinations': destinations[val_mask],
            'timestamps': timestamps[val_mask],
            'edge_idxs': edge_idxs[val_mask],
            'edge_feat': edge_feat[val_mask],
            'y': y[val_mask]
        }

        test_data = {
            'sources': sources[test_mask],
            'destinations': destinations[test_mask],
            'timestamps': timestamps[test_mask],
            'edge_idxs': edge_idxs[test_mask],
            'edge_feat': edge_feat[test_mask],
            'y': y[test_mask]
        }
        return train_data, val_data, test_data

       


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
        Returns the full data of the dataset as a dictionary with keys:
            sources, destinations, timestamps, edge_idxs, y (edge weight)
        Returns:
            full_data: Dict[str, Any]
        """
        if (self._full_data is None):
            raise ValueError("dataset has not been processed yet, please call pre_process() first")
        return self._full_data
    

    @property
    def train_data(self) -> Dict[str, Any]:
        r"""
        Returns the train data of the dataset as a dictionary with keys:
            sources, destinations, timestamps, edge_idxs, y (edge weight)
        Returns:
            train_data: Dict[str, Any]
        """
        if (self._train_data is None):
            raise ValueError("dataset has not been processed yet, please call pre_process() first")
        return self._train_data
    
    @property
    def val_data(self) -> Dict[str, Any]:
        r"""
        Returns the validation data of the dataset as a dictionary with keys:
            sources, destinations, timestamps, edge_idxs, y (edge weight)
        Returns:
            val_data: Dict[str, Any]
        """
        if (self._val_data is None):
            raise ValueError("dataset has not been processed yet, please call pre_process() first")
        return self._val_data
    
    @property
    def test_data(self) -> Dict[str, Any]:
        r"""
        Returns the test data of the dataset as a dictionary with keys:
            sources, destinations, timestamps, edge_idxs, y (edge weight)
        Returns:
            test_data: Dict[str, Any]
        """
        if (self._test_data is None):
            raise ValueError("dataset has not been processed yet, please call pre_process() first")
        return self._test_data
    




def main():
    name = "opensky"
    dataset = LinkPropPredDataset(name=name, root="datasets", preprocess=True)
    
    dataset.node_feat
    dataset.edge_feat #not the edge weights
    dataset.full_data
    dataset.full_data["edge_idxs"]
    dataset.full_data["sources"]
    dataset.full_data["destinations"]
    dataset.full_data["timestamps"] 
    dataset.full_data["y"]

    dataset.train_data
    dataset.val_data
    dataset.test_data

if __name__ == "__main__":
    main()