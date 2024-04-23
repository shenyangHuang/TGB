import numpy as np

import sys
sys.path.insert(0, '/home/mila/j/julia.gastinger/TGB2')
sys.path.insert(0,'/../../../')

## imports


import numpy as np

import os
import os.path as osp
from pathlib import Path
import pandas as pd

#internal imports 
from tgb_modules.recurrencybaseline_predictor import apply_baselines, apply_baselines_remote
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset import LinkPropPredDataset 
from tgb.utils.utils import set_random_seed, create_basis_dict, group_by, reformat_ts


# create a dictionary with all the stats and save to json and csv
def create_dict_and_save(dataset_name, num_rels, num_nodes, num_train_quads, num_val_quads, num_test_quads, num_all_quads,
                         num_train_timesteps, num_val_timesteps, num_test_timesteps, num_all_timesteps):
    stats_dict = {
        "dataset_name": dataset_name,
        "num_rels": num_rels,
        "num_nodes": num_nodes,
        "num_train_quads": num_train_quads,
        "num_val_quads": num_val_quads,
        "num_test_quads": num_test_quads,
        "num_all_quads": num_all_quads,
        "num_train_timesteps": num_train_timesteps,
        "num_val_timesteps": num_val_timesteps,
        "num_test_timesteps": num_test_timesteps,
        "num_all_timesteps": num_all_timesteps
        # "train_nodes": train_nodes
    }

    df = pd.DataFrame.from_dict(stats_dict, orient='index')

    # save
    # Get the current directory of the script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate one folder up
    parent_dir = os.path.dirname(current_dir)

    # Save stats_dict as CSV
    modified_dataset_name = dataset_name.replace('-', '_')
    save_path = (os.path.join(parent_dir, modified_dataset_name, "dataset_stats.csv"))
    df.to_csv(save_path)

    print("Stats saved to csv and json in folder: ", save_path)

names = ['tkgl-yago', 'tkgl-polecat', 'tkgl-icews']
for dataset_name in names:
    dataset = LinkPropPredDataset(name=dataset_name, root="datasets", preprocess=True)

    relations = dataset.edge_type
    num_rels = dataset.num_rels
    num_rels_without_inv = int(num_rels/2)

    rels = np.arange(0,num_rels)
    subjects = dataset.full_data["sources"]
    objects= dataset.full_data["destinations"]
    num_nodes = dataset.num_nodes 
    timestamps_orig = dataset.full_data["timestamps"]
    timestamps = reformat_ts(timestamps_orig) # stepsize:1

    all_quads = np.stack((subjects, relations, objects, timestamps, timestamps_orig), axis=1)
    train_data = all_quads[dataset.train_mask]
    val_data = all_quads[dataset.val_mask]
    test_data = all_quads[dataset.test_mask]


    # compute number of quads in train/val/test set
    num_train_quads = train_data.shape[0]
    num_val_quads = val_data.shape[0]
    num_test_quads = test_data.shape[0]
    num_all_quads = num_train_quads + num_val_quads + num_test_quads

    # compute number of timesteps in train/val/test set
    num_train_timesteps = len(np.unique(train_data[:,-1]))
    num_val_timesteps = len(np.unique(val_data[:,-1]))
    num_test_timesteps = len(np.unique(test_data[:,-1]))
    num_all_ts = num_train_timesteps + num_val_timesteps + num_test_timesteps

    # compute number on nodes in valid set or test set that have not been seen in train set
    train_nodes = np.unique(train_data[:,0]) + np.unique(train_data[:,2])


    # compute graph parameters (density and such stuff)

    # compute number of triples per timestep

    # compute recurrency factor

    # compute average duration of facts


    create_dict_and_save(dataset_name, num_rels_without_inv, num_nodes, num_train_quads, num_val_quads, num_test_quads, 
                         num_all_quads, num_train_timesteps, num_val_timesteps, num_test_timesteps, num_all_ts)
