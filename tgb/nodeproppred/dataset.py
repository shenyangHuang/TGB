from typing import Optional, Dict, Any, Tuple
import os
import os.path as osp
import numpy as np
import pandas as pd
import shutil
import zipfile
import requests
from clint.textui import progress

from tgb.utils.info import PROJ_DIR, DATA_URL_DICT, BColors, DATA_NUM_CLASSES
from tgb.utils.utils import save_pkl, load_pkl, find_nearest
from tgb.utils.pre_process import load_genre_list, load_label_dict, load_edgelist_sr, load_labels_sr, load_edgelist_datetime

# TODO add node label loading code, node label convertion to unix time code etc.
class NodePropertyDataset(object):
    def __init__(
        self, 
        name: str, 
        root: Optional[str] = 'datasets', 
        meta_dict: Optional[dict] = None,
        preprocess: Optional[bool] = True,
        ):
        r"""Dataset class for edge regression tasks. Stores meta information about each dataset such as evaluation metrics etc.
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
            self.meta_dict["fname"] = self.root + "/" + self.name + "_edgelist.csv"
            self.meta_dict["nodefile"] = self.root + "/" + self.name + "_node_labels.csv"

        self._num_classes = DATA_NUM_CLASSES[self.name]

        #initialize
        self._node_feat = None
        self._edge_feat = None
        self._full_data = None
        self.download()
        #check if the root directory exists, if not create it
        if osp.isdir(self.root):
            print("Dataset directory is ", self.root)
        else:
            #os.makedirs(self.root)
            raise FileNotFoundError(f"Directory not found at {self.root}")

        if preprocess:
            self.pre_process()

        self.label_ts_idx = 0  #index for which node lables to return now


    def download(self):
        """
        downloads this dataset from url
        check if files are already downloaded
        """
        #check if the file already exists
        if (osp.exists(self.meta_dict["fname"]) and osp.exists(self.meta_dict["nodefile"])):
            print ("file found, skipping download")
            return

        else:
            inp = input('Will you download the dataset(s) now? (y/N)\n').lower() #ask if the user wants to download the dataset
            if inp == 'y':
                print(f"{BColors.WARNING}Download started, this might take a while . . . {BColors.ENDC}")
                print(f"Dataset title: {self.name}")

                if (self.url is None):
                    raise Exception("Dataset url not found, download not supported yet.")
                else:
                    r = requests.get(self.url, stream=True)
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
        returns an edge list of pandas data frame
        Parameters:
            fname: path to raw data file
        Returns:
            df: pandas data frame
        """
        OUT_DF = self.root + '/' + 'ml_{}.pkl'.format(self.name)
        OUT_NODE_DF = self.root + '/' + 'ml_{}_node.pkl'.format(self.name)
        
        '''
        reprocess the node label dataset again
        '''     
        if (osp.exists(OUT_DF) and osp.exists(OUT_NODE_DF)):
            print ("loading processed file")
            df = pd.read_pickle(OUT_DF)
            node_label_dict = load_pkl(OUT_NODE_DF)
            #node_df = pd.read_pickle(OUT_NODE_DF)
        else:
            #! now directly load the processed files and not providing the raw files
            print ("file not processed, generating processed file")
            #print ("processing will take around 10 minutes and then the panda dataframe object will be saved to disc")
            if (self.name == "subreddits"):
                df, edge_feat, node_ids, labels_dict = load_edgelist_sr(self.meta_dict["fname"], label_size=self._num_classes)
                df.to_pickle(OUT_DF)
                node_label_dict = load_label_dict(self.meta_dict["nodefile"], node_ids, labels_dict)
                save_pkl(node_label_dict, OUT_NODE_DF)
                # node_df = load_labels_sr(self.meta_dict["nodefile"], node_ids, rd_dict)
                # node_df.to_pickle(OUT_NODE_DF)

            if (self.name == "lastfmgenre"):
                df, edge_feat, node_ids, labels_dict = load_edgelist_datetime(self.meta_dict["fname"], label_size=self._num_classes) 
                df.to_pickle(OUT_DF)
                # save_pkl(node_ids, self.root + '/' +"node_ids.pkl")
                # save_pkl(labels_dict, self.root + '/' +"labels_dict.pkl")
                # node_ids = load_pkl(self.root + '/' +"node_ids.pkl")
                # labels_dict = load_pkl(self.root + '/' +"labels_dict.pkl")

                node_label_dict = load_label_dict(self.meta_dict["nodefile"], node_ids, labels_dict, date_format=True)
                save_pkl(node_label_dict, OUT_NODE_DF)
                
        return df, node_label_dict
    




    def pre_process(self):
        '''
        Pre-process the dataset and generates the splits, must be run before dataset properties can be accessed
        generates self.full_data, self.train_data, self.val_data, self.test_data
        Parameters:
            feat_dim: dimension for feature vectors, padded to 172 with zeros
        '''

        #first check if all files exist
        if ("fname" not in self.meta_dict) or ("nodefile" not in self.meta_dict):
            raise Exception("meta_dict does not contain all required filenames")        
        
        df, node_label_dict = self.generate_processed_files() 
        sources = np.array(df['u'])
        destinations = np.array(df['i'])
        timestamps = np.array(df['ts'])
        edge_idxs = np.array(df['idx'])
        y = np.ones(sources.shape[0])
        self._edge_feat = np.array(df['w'])

        full_data = {
            'sources': sources,
            'destinations': destinations,
            'timestamps': timestamps,
            'edge_idxs': edge_idxs,
            'edge_feat': self._edge_feat,
            'y': y,
        }
        self._full_data = full_data

        #! store the masks here instead of full arrays
        _train_mask, _val_mask, _test_mask = self.generate_splits(full_data)
        
        self._train_mask = _train_mask
        self._val_mask = _val_mask
        self._test_mask = _test_mask

        self.label_dict = node_label_dict
        self.label_ts = np.array(list(node_label_dict.keys()))
        self.label_ts = np.sort(self.label_ts) 


    # TODO load from fixed split index from disc

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
        train_mask = timestamps <= val_time
        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
        test_mask = timestamps > test_time

        return train_mask, val_mask, test_mask   

    def find_next_labels_batch(self, cur_t):
        r"""
        this function simply returns the next day's label if the cur_t >= current label ts
        Parameters:
            cur_t: current timestamp of the batch of edges
        Returns:
            ts: timestamp of the node labels
            source_idx: node ids
            labels: the stacked label vectors
        """
        try:
            ts = self.label_ts[self.label_ts_idx]
        except:
            print("node labels need to be reset, please run dataset.reset_label_time()")
            return None

        #! double check the logic here 
        self.label_ts_idx += 1 #move to the next ts
        # {ts: {node_id: label_vec}}
        node_ids = np.array(list(self.label_dict[ts].keys()))

        node_labels = []
        for key in self.label_dict[ts]:
            node_labels.append(np.array(self.label_dict[ts][key]))
        node_labels = np.stack(node_labels, axis=0)
        #node_labels = np.stack(list(self.label_dict[ts].items()), axis=0)
        label_ts = np.full(node_ids.shape[0], ts, dtype="int")
        return (label_ts, node_ids, node_labels)
         
        # if (cur_t >= ts):
        #     self.label_ts_idx += 1 #move to the next ts
        #     # {ts: {node_id: label_vec}}
        #     node_ids = np.array(list(self.label_dict[ts].keys()))

        #     node_labels = []
        #     for key in self.label_dict[ts]:
        #         node_labels.append(np.array(self.label_dict[ts][key]))
        #     node_labels = np.stack(node_labels, axis=0)
        #     #node_labels = np.stack(list(self.label_dict[ts].items()), axis=0)
        #     label_ts = np.full(node_ids.shape[0], ts, dtype="int")
        #     return (label_ts, node_ids, node_labels)
        # else:
        #     return None

    def reset_label_time(self):
        self.label_ts_idx = 0


    def return_label_ts(self):
        return self.label_ts[self.label_ts_idx]


    @property
    def num_classes(self) -> int:
        """
        number of classes in the node label
        """    
        return self._num_classes


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
    def train_mask(self) -> Dict[str, Any]:
        r"""
        Returns the train mask of the dataset 
        Returns:
            train_mask: Dict[str, Any]
        """
        if (self._train_mask is None):
            raise ValueError("training split hasn't been loaded")
        return self._train_mask
    
    @property
    def val_mask(self) -> Dict[str, Any]:
        r"""
        Returns the validation mask of the dataset 
        Returns:
            val_mask: Dict[str, Any]
        """
        if (self._val_mask is None):
            raise ValueError("validation split hasn't been loaded")
        
        return self._val_mask
    
    @property
    def test_mask(self) -> Dict[str, Any]:
        r"""
        Returns the test mask of the dataset:
        Returns:
            test_mask: Dict[str, Any]
        """
        if (self._test_mask is None):
            raise ValueError("test split hasn't been loaded")
        
        return self._test_mask
    




def main():
    # download files
    name = "subreddits" #"lastfmgenre"
    dataset = NodePropertyDataset(name=name, root="datasets", preprocess=True)

    
    dataset.node_feat
    dataset.edge_feat #not the edge weights
    dataset.full_data
    dataset.full_data["edge_idxs"]
    dataset.full_data["sources"]
    dataset.full_data["destinations"]
    dataset.full_data["timestamps"] 
    dataset.full_data["y"]

    train_data = dataset.full_data[dataset.train_mask]
    val_data = dataset.full_data[dataset.val_mask]
    test_data = dataset.full_data[dataset.test_mask]

if __name__ == "__main__":
    main()