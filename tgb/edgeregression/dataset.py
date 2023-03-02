from typing import Optional, cast, Union, List, overload, Literal
import os.path as osp
import numpy as np
import pandas as pd


from tgb.info import PROJ_DIR
from tgb.utils.pre_process import _to_pd_data, reindex


class EdgeRegressionDataset(object):
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

        #check if the root directory exists, if not create it
        if osp.isdir(self.root):
            print("Dataset directory is ", self.root)
        else:
            raise FileNotFoundError(f"Directory not found at {self.root}")


        #TODO Andy: add url logic here from info.py to manage the urls in a centralized file

        if preprocess:
            self.pre_process()
        


    def pre_process(self):
        r"""Turns raw data .csv file into TG learning ready format such as for TGN, stores the processed file locally for faster access later
        Returns:
            'ml_<network>.csv': source, destination, timestamp, state_label, index 	# 'index' is the index of the line in the edgelist
            'ml_<network>.npy': contains the edge features; this is a numpy array where each element corresponds to the features of the corresponding line specifying one edge. If there are no features, should be initialized by zeros
            'ml_<network>_node.npy': contains the node features; this is a numpy array that each element specify the features of one node where the node-id is equal to the element index.
        """

        #check if path to file is valid 
        if not osp.exists(self.meta_dict['fname']):
            raise FileNotFoundError(f"File not found at {self.meta_dict['fname']}")
        
        #output file names 
        OUT_DF = self.root + '/' + 'ml_{}.csv'.format(self.name)
        OUT_FEAT = self.root + '/' + 'ml_{}.npy'.format(self.name)
        OUT_NODE_FEAT =  self.root + '/' + 'ml_{}_node.npy'.format(self.name)

        #check if the output files already exist, if so, skip the pre-processing
        if osp.exists(OUT_DF) and osp.exists(OUT_FEAT) and osp.exists(OUT_NODE_FEAT):
            print ("pre-processed files found, skipping file generation")
        else:
            df, feat = _to_pd_data(self.meta_dict['fname'])
            df = reindex(df, bipartite=False)
            empty = np.zeros(feat.shape[1])[np.newaxis, :]
            feat = np.vstack([empty, feat])

            max_idx = max(df.u.max(), df.i.max())
            rand_feat = np.zeros((max_idx + 1, 172))

            df.to_csv(OUT_DF)
            np.save(OUT_FEAT, feat)
            np.save(OUT_NODE_FEAT, rand_feat)




    
   
    

        
    



def main():
    dataset = EdgeRegressionDataset(name="un_trade", root="datasets", preprocess=True)

if __name__ == "__main__":
    main()