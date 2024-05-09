import numpy as np

import sys
import os
import os.path as osp
from pathlib import Path
tgb_modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(tgb_modules_path)
## imports


import numpy as np
import seasonal

import pandas as pd
from datetime import datetime
#internal imports 
from tgb.linkproppred.dataset import LinkPropPredDataset 
from tgb.utils.utils import  reformat_ts, get_original_ts

import networkx as nx
import matplotlib.pyplot as plt


def t2s(s,r,o):
    return str(s) + " " + str(r) + " " + str(o)

class TripleSet:

    def __init__(self):
        self.sub_rel_2_obj = {}
        self.obj_rel_2_sub = {}
        self.t_2_triple = {}
        self.triples = []
        
        self.timestep_lists = []
        self.counter_s = 0
        self.max_ts = 0
        self.min_ts = 1000
        self.num_timesteps = 0
        self.num_triples = 0

    def add_triples(self, data, num_rels, timestep_range):
        t_min = None
        t_max = None
        count = 0
        self.t_2_triple = {}
        for t in range(timestep_range):
            self.t_2_triple[t] = []

        for line in data:
            s = line[0]
            r = line[1] 
            o = line[2]
            t = line[3]
            

            if r >= num_rels:
                self.index_triple(self.obj_rel_2_sub, s,r-num_rels,o,t)
            else:  
                self.index_triple(self.sub_rel_2_obj, s,r,o,t)
                self.index_timestamp(self.t_2_triple, s, r, o, t)
                self.triples.append([s,r,o,t]) 
            
            if t_min == None:
                t_min = t
                t_max = t
            if t < t_min: t_min = t
            if t > t_max: t_max = t
            count += 1

        # print(counter_s)
        print(">>> read " + str(count) + " triples from time " + str(t_max) + " to " + str(t_min))
        if t_min < self.min_ts:
            self.min_ts = t_min
        if t_max > self.max_ts:
            self.max_ts = t_max
        

    def compute_stat(self):
        self.timestep_lists = self.create_timestep_lists(self.sub_rel_2_obj)
        self.num_timesteps = 1+ self.max_ts - self.min_ts
        self.num_triples = len(self.triples)

    def create_timestep_lists(self, x_y_2_z):
        timestep_lists = list(self.flatten_dict(x_y_2_z))
        return timestep_lists

    def flatten_dict(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                yield from self.flatten_dict(value)
            else:
                yield value

    def index_triple(self, x_y_2_z, x, y, z, t):
        if x not in x_y_2_z:
            x_y_2_z[x] = {}
        if y not in x_y_2_z[x]:
            x_y_2_z[x][y] = {}
        if z not in x_y_2_z[x][y]:
            x_y_2_z[x][y][z] = []
            # counter +=1
        if t not in x_y_2_z[x][y][z]:
            x_y_2_z[x][y][z].append(t)
        # return counter

    def index_timestamp(self, t_2_triple, s, r, o, t):
        if t not in t_2_triple:
            t_2_triple[t] = []
        t_2_triple[t].append([s,r,o])

    def get_latest_ts(self, s, r, o, t):
        closest = -1
        if s in self.sub_rel_2_obj:
            if r in self.sub_rel_2_obj[s]:
                if o in self.sub_rel_2_obj[s][r]:
                    ts = self.sub_rel_2_obj[s][r][o]
                    for k in ts:
                        if k < t and k > closest:
                            closest = k
        return closest

    def count(self, e):
        return len(list(filter(lambda x : x[0] == e or x[2] == e, self.triples)))

    def show(self, num=100):
        count = 0
        len_sum = 0
        for s in self.sub_rel_2_obj:
            for r in self.sub_rel_2_obj[s]:
                for o in self.sub_rel_2_obj[s][r]:
                    ts = self.sub_rel_2_obj[s][r][o]
                    print(t2s(s,r,o) + ": " + str(len(ts)))
                    len_sum += len(ts)
                    count +=1
                    if count > num:
                        return
        # print("mean length: " + str(len_sum / count))
    
    def unique_nodes(self, snapshot):
        """
            Returns the set of unique nodes in a snapshot."""
        return   set([x[0] for x in snapshot] + [x[2] for x in snapshot])

def max_consecutive_numbers(lst):
    max_count = 0
    current_count = 1
    
    for i in range(1, len(lst)):
        if lst[i] == lst[i-1] + 1:
            current_count += 1
        else:
            max_count = max(max_count, current_count)
            current_count = 1
    
    return max(max_count, current_count)


def extend_features(self, seasonal_markers_dict, all_features, timesteps_of_interest):
    """ dict with key: timestep, values: all the extended features.
    """

    extended_features = {}
    for ts in timesteps_of_interest:
        index = self._timestep_indexer[ts]
        features_ts = [all_features[feat][index] for feat in range(len(all_features))]
        features_ts.append(seasonal_markers_dict[ts])
        extended_features[ts] = features_ts
    
    return extended_features

def extract_timeseries_from_graphs(self, graph_dict):
    """ extracts multivariate timeseries from quadruples based on graph params

    :param graph_dict: dict, with keys: timestep, values: triples; training quadruples.

    """
    num_nodes = []
    num_triples = []
    max_deg = []
    mean_deg = []
    mean_deg_c = [] 
    max_deg_c = [] 
    min_deg_c = [] 
    density = []

    for ts, triples_snap in graph_dict.items():

        # create graph for that timestep
        e_list_ts = [(triples_snap[line][0], triples_snap[line][2]) for line in range(len(triples_snap))]
        G = nx.MultiGraph()
        G.add_nodes_from(graph_dict[ts][:][ 0])
        G.add_nodes_from(graph_dict[ts][:][2])
        G.add_edges_from(e_list_ts)  # default edge data=1

        # extract relevant parameters and append to list
        num_nodes.append(G.number_of_nodes())
        num_triples.append(G.number_of_edges())

        # degree
        deg_list = list(dict(G.degree(G.nodes)).values())
        max_deg.append(np.max(deg_list))
        mean_deg.append(np.mean(deg_list))
        
        # degree centrality
        deg_clist = list(dict(nx.degree_centrality(G)).values())
        mean_deg_c.append(np.mean(deg_clist))
        max_deg_c.append(np.max(deg_clist))
        min_deg_c.append(np.min(deg_clist))
        
        density.append(nx.density(G))

    return [num_triples, num_nodes, max_deg, mean_deg, mean_deg_c, max_deg_c, min_deg_c, density]

def estimate_seasons(train_data):
    """ Estimate seasonal effects in a series.
            
    Estimate the major period of the data by testing seasonal differences for various period lengths and returning 
    the seasonal offsets that best predict out-of-sample variation.   
        
    First, a range of likely periods is estimated via periodogram averaging. Next, a time-domain period 
    estimator chooses the best integer period based on cross-validated residual errors. It also tests
    the strength of the seasonal effect using the R^2 of the leave-one-out cross-validation.

    :param data: list, data to be analysed, time-series;
    :return: NBseason int. if no season found: 1; else: seasonality that was discovered (e.g. if seven and 
            time granularity is daily: weekly seasonality)
    """
    seasons, trended = seasonal.fit_seasons(train_data)
    
    if seasons is None:
        Nbseason = int(1)
    else: 
        Nbseason = len(seasons)
        
    return Nbseason


# create a dictionary with all the stats and save to json and csv
def create_dict_and_save(dataset_name, num_rels, num_nodes, num_train_quads, num_val_quads, num_test_quads, num_all_quads,
                         num_train_timesteps, num_val_timesteps, num_test_timesteps, num_all_timesteps,
                         test_ind_nodes, test_ind_nodes_perc, val_ind_nodes, val_ind_nodes_perc, 
                         direct_recurrency_degree, recurrency_degree, consecutiveness_degree,
                         mean_edge_per_ts, std_edge_per_ts, min_edge_per_ts, max_edge_per_ts,
                         mean_node_per_ts, std_node_per_ts, min_node_per_ts, max_node_per_ts,
                         seasonal_value, collision_trainval, collision_valtest):
    if  'tkgl' in dataset_name:
        num_train_quads = int(num_train_quads/2)
        num_val_quads = int(num_val_quads/2)
        num_test_quads = int(num_test_quads/2)
        num_all_quads = int(num_all_quads/2)

    stats_dict = {
        "dataset_name": dataset_name,
        "num_rels": num_rels,
        "num_nodes": num_nodes,
        "num_train_quads": num_train_quads,
        "num_val_quads": num_val_quads,
        "num_test_quads": num_test_quads,
        "num_all_quads": num_all_quads,
        "test_ind_nodes": test_ind_nodes,
        "test_ind_nodes_perc": test_ind_nodes_perc,
        "val_ind_nodes": val_ind_nodes,
        "val_ind_nodes_perc": val_ind_nodes_perc,
        "num_train_timesteps": num_train_timesteps,
        "num_val_timesteps": num_val_timesteps,
        "num_test_timesteps": num_test_timesteps,
        "num_all_timesteps": num_all_timesteps,
        "direct_recurrency_degree": direct_recurrency_degree,
        "recurrency_degree": recurrency_degree,
        "consecutiveness_degree": consecutiveness_degree,
        "mean_edge_per_ts": mean_edge_per_ts,
        "std_edge_per_ts": std_edge_per_ts,
        "min_edge_per_ts": min_edge_per_ts,
        "max_edge_per_ts": max_edge_per_ts,
        "mean_node_per_ts": mean_node_per_ts,
        "std_node_per_ts": std_node_per_ts,
        "min_node_per_ts": min_node_per_ts,
        "max_node_per_ts": max_node_per_ts,
        "seasonal_value": seasonal_value,
        "collision_trainval": collision_trainval,
        "collision_valtest": collision_valtest        
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

def num_nodes_not_in_train(train_data, test_data):
    """ Calculate the number of nodes in the test set that are not in the train set.
    :param train_data: np.array, training data
    :param test_data: np.array, test data
    :return: int, number of nodes in the test set that are not in the train set
    """
    train_nodes =np.unique(np.concatenate((np.unique(train_data[:,0]), np.unique(train_data[:,2]))))
    test_nodes = np.unique(np.concatenate((np.unique(test_data[:,0]), np.unique(test_data[:,2]))))
    num_nodes_not_in_train = len(np.setdiff1d(test_nodes, train_nodes))
    return num_nodes_not_in_train



names = [ 'thgl-myket','thgl-github'] #'tkgl-wikidata', 'tkgl-yago', 'thgl-forum',
for dataset_name in names:
    dataset = LinkPropPredDataset(name=dataset_name, root="datasets", preprocess=True)

    relations = dataset.edge_type
    num_rels = dataset.num_rels
    if 'tkgl' in dataset_name:
        num_rels_without_inv = int(num_rels/2)
    else:
        num_rels_without_inv = num_rels

    rels = np.arange(0,num_rels)
    subjects = dataset.full_data["sources"]
    objects= dataset.full_data["destinations"]
    num_nodes = dataset.num_nodes 
    timestamps_orig = dataset.full_data["timestamps"]
    timestamps = reformat_ts(timestamps_orig, dataset_name) # stepsize:1

    all_quads = np.stack((subjects, relations, objects, timestamps, timestamps_orig), axis=1)
    train_data = all_quads[dataset.train_mask]
    val_data = all_quads[dataset.val_mask]
    test_data = all_quads[dataset.test_mask]
    collision_trainval = np.intersect1d(list(set(timestamps_orig[dataset.train_mask])), list(set(timestamps_orig[dataset.val_mask])))
    collision_valtest = np.intersect1d(list(set(timestamps_orig[dataset.val_mask])), list(set(timestamps_orig[dataset.test_mask])))
    if len(collision_trainval) > 0:
        print("!!!!!!!!!Collision between train and val set!!!!!!!!!")
    if len(collision_valtest) > 0:
        print("!!!!!!!!!Collision between val and test set!!!!!!!!!")
    print(subjects.shape)

    # compute number of quads in train/val/test set
    num_train_quads = train_data.shape[0]
    num_val_quads = val_data.shape[0]
    num_test_quads = test_data.shape[0]
    num_all_quads = num_train_quads + num_val_quads + num_test_quads
    print(num_all_quads)

    # compute inductive nodes
    test_ind_nodes = num_nodes_not_in_train(train_data, test_data)
    val_ind_nodes = num_nodes_not_in_train(train_data, val_data)
    test_ind_nodes_perc = test_ind_nodes/num_nodes
    val_ind_nodes_perc = val_ind_nodes/num_nodes

    # compute number of timesteps in train/val/test set
    num_train_timesteps = len(np.unique(train_data[:,-1]))
    num_val_timesteps = len(np.unique(val_data[:,-1]))
    num_test_timesteps = len(np.unique(test_data[:,-1]))
    num_all_ts = num_train_timesteps + num_val_timesteps + num_test_timesteps

    # compute number on nodes in valid set or test set that have not been seen in train set


    # compute recurrency factor
    # compute average duration of facts
    timestep_range = 1+np.max(timestamps) - np.min(timestamps)
    all_possible_timestep_indices = [i for i in range(timestep_range)]
    ts_all = TripleSet()
    ts_all.add_triples(all_quads, num_rels_without_inv, timestep_range)
    ts_all.compute_stat()
    ts_test = TripleSet()
    ts_test.add_triples(test_data, num_rels_without_inv, timestep_range)
    ts_test.compute_stat()

    lens = []
    for timesteps in ts_all.timestep_lists:
        lens.append(len(timesteps))

    count_previous = 0
    count_sometime = 0
    count_all = 0
    for qtriple in ts_test.triples:    
        (s,r,o,t) = qtriple
        k = ts_all.get_latest_ts(s,r,o, t)
        count_all += 1
        if k + 1 == t: count_previous += 1
        if k > -1 and k < t: count_sometime += 1

    print("DATATSET:  " + dataset_name)
    print("all:       " +  str(count_all))
    print("previous:  " +  str(count_previous))
    print("sometime:  " +  str(count_sometime))
    print("f-direct (DRec):   " +  str(count_previous / count_all))
    print("f-sometime (Rec): " +  str(count_sometime / count_all))

    print(f"the mean number of timesteps that a triple appears in is {np.mean(lens)}")
    print(f"the median number of timesteps that a triple appears in is {np.median(lens)}")
    print(f"the maximum number of timesteps that a triple appears in is {np.max(lens)}")

    # Compute max consecutive timesteps per triple
    results = [max_consecutive_numbers(inner_list) for inner_list in ts_all.timestep_lists]
    print(f"number of timesteps is {ts_all.num_timesteps}")
    print(f"number of total triples is {ts_all.num_triples}")
    print(f"number of distinct triples is {len(ts_all.timestep_lists)}")
    print(f"the mean max number of 100*consecutive timesteps/number of timesteps that a triple appears in is {100*np.mean(results)/ts_all.num_timesteps}")
    print(f"the median max number of 100*consecutive timesteps/number of timesteps that a triple appears in is {100*np.median(results)/ts_all.num_timesteps}")
    print(f"the maximum max number of 100*consecutive timesteps/number of timesteps that a triple appears in is {100*np.max(results)/ts_all.num_timesteps}")
    print(f"the mean max number of consecutive timesteps that a triple appears in is {np.mean(results)}")
    print(f"the median max number of consecutive timesteps that a triple appears in is {np.median(results)}")
    print(f"the maximum max number of consecutive timesteps that a triple appears in is {np.max(results)}")
    print(f"the std for max number of consecutive timesteps that a triple appears in is {np.std(results)}")

    direct_recurrency_degree = count_previous / count_all
    recurrency_degree = count_sometime / count_all
    consecutiveness_degree =  np.mean(results) # the mean max number of consecutive timesteps that a triple appears in
    # compute graph parameters (density and such stuff)

    # compute number of triples per timestep
    n_nodes_list = []
    n_edges_list = []

    ts_set = list(set(timestamps_orig))
    ts_set.sort()
    ts_dist = ts_set[1] - ts_set[0]
    if 'tkg' in dataset_name:
        all_possible_orig_timestamps =get_original_ts(all_possible_timestep_indices, ts_dist, np.min(ts_set))

    no_nodes_list = []
    no_nodes_list_orig = []
    no_nodes_datetime = []
    for t in ts_all.t_2_triple.keys():
        num_nodes_ts = len(ts_all.unique_nodes(ts_all.t_2_triple[t]))
        n_nodes_list.append(num_nodes_ts)
        n_edges_list.append(len(ts_all.t_2_triple[t]))
        if 'tkg' in dataset_name:
            if num_nodes_ts == 0:
                if t not in no_nodes_list:
                    no_nodes_list.append(t)
                    no_nodes_list_orig.append(all_possible_orig_timestamps[t])
                    no_nodes_datetime.append(datetime.utcfromtimestamp(all_possible_orig_timestamps[t]))
    # compute seasonality of num nodes over time: 
    seasonal_value = estimate_seasons(n_nodes_list)
    if seasonal_value == 1:
        print('there was no seasonality for number of nodes found')
    else:
        print(f'the seasonality for number of nodes is {seasonal_value}')
    if 'tkgl' in dataset_name:
        print('we have 0 nodes for' + str(len(no_nodes_list)) + ' timesteps')
        print('0 nodes for timesteps: ', no_nodes_list)
        print('this is original unix timestamps: ', no_nodes_list_orig)
        print('this is datetime: ', no_nodes_datetime)
    else:
        print('we have 0 nodes for' + str(len(no_nodes_list)) + ' timesteps')

            
    print(f"average number of triples per ts is {np.mean(n_edges_list)}")
    print(f"std for average number of triples per ts is {np.std(n_edges_list)}")
    print(f"min/max number of triples per ts is {np.min(n_edges_list), np.max(n_edges_list)}")

    print(f"average number of nodes per ts is {np.mean(n_nodes_list)}")
    print(f"std for average number of nodes per ts is {np.std(n_nodes_list)}")
    print(f"min/max number of nodes per ts is {np.min(n_nodes_list), np.max(n_nodes_list)}")

    plt.figure()
    plt.scatter(range(ts_all.num_timesteps), n_edges_list, s=0.2)
    plt.xlabel('timestep')
    plt.ylabel('number of triples')
    plt.title(f'Number of triples per timestep for {dataset_name}')
    # save
    # Get the current directory of the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate one folder up
    parent_dir = os.path.dirname(current_dir)
    # Save stats_dict as CSV
    modified_dataset_name = dataset_name.replace('-', '_')
    save_path = (os.path.join(parent_dir, modified_dataset_name, f"num_edges_per_ts_{dataset_name}.png"))
    plt.savefig(save_path)

    plt.figure()
    plt.scatter(range(ts_all.num_timesteps), n_nodes_list, s=0.2)
    plt.xlabel('timestep')
    plt.ylabel('number of nodes')
    plt.title(f'Number of nodes per timestep for {dataset_name}')
    save_path = (os.path.join(parent_dir, modified_dataset_name, f"num_nodes_per_ts_{dataset_name}.png"))
    plt.savefig(save_path)

    
   

    create_dict_and_save(dataset_name, num_rels_without_inv, num_nodes, num_train_quads, num_val_quads, num_test_quads, 
                         num_all_quads, num_train_timesteps, num_val_timesteps, num_test_timesteps, num_all_ts,
                         test_ind_nodes, test_ind_nodes_perc, val_ind_nodes, val_ind_nodes_perc, 
                         direct_recurrency_degree, recurrency_degree, consecutiveness_degree,
                         np.mean(n_edges_list), np.std(n_edges_list), np.min(n_edges_list), np.max(n_edges_list),
                         np.mean(n_nodes_list), np.std(n_nodes_list), np.min(n_nodes_list), np.max(n_nodes_list),
                         seasonal_value, collision_trainval, collision_valtest)
