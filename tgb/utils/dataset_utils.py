import numpy as np

import sys
import os
import os.path as osp
from pathlib import Path
tgb_modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(tgb_modules_path)
import json

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
    """ class to store triples and their indices
    """

    def __init__(self):
        self.sub_rel_2_obj = {}
        self.obj_rel_2_sub = {}
        self.rel_sub_obj_t = {}
        self.t_2_triple = {}
        self.r_2_triple = {}
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
                self.index_triple(self.obj_rel_2_sub, s,r-num_rels,o,t) # for each triple we store all timestamps that it has been observed. keys: [s][r][o], values: list of timestamps
            else:  
                self.index_triple(self.sub_rel_2_obj, s,r,o,t) #for each triple we store all timestamps that it has been observed.
                self.index_triple(self.rel_sub_obj_t, r,s, o, t) #for each triple we store all timestamps that it has been observed. keys: [r][s][o], values: list of timestamps
                self.index_timestamp(self.t_2_triple, s, r, o, t) #for each timestamp, we store a list of triples that have that timestamp keys: timestmp, values: list of triples
                self.index_rels(self.r_2_triple, s, r, o, t)
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
        """
        Indexes a triple in a dictionary of dictionaries of lists.
        for each triple we store all timestamps that it has been observed."""
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

    def index_rels(self, r_2_triple, s, r, o, t):
        """
        Indexes a triple in a dictionary of lists of triples.
        for each relation, we store a list of triples that have that relation."""
        if r not in r_2_triple:
            r_2_triple[r] = []
        r_2_triple[r].append([s,o,t])

    def index_timestamp(self, t_2_triple, s, r, o, t):
        """"
        Indexes a triple in a dictionary of lists of triples.
        for each timestamp, we store a list of triples that have that timestamp."""
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

def discretize_values(num_nodes_per_timestep, k):
    """
    Discretize the values of a time series into k bins and return the mean value of each bin.
    """
    # Calculate the number of elements per bin
    elements_per_bin = len(num_nodes_per_timestep) // k
    
    # Create an empty list to store the mean values of each bin
    means = []
    ts_discretized_min = []
    ts_discretized_max = []
    
    # Iterate over the bins
    start_indices =[]
    end_indices = []
    mid_indices = []
    sums = []
    for i in range(k):
        # Calculate the start and end indices of the current bin
        start_index = i * elements_per_bin
        end_index = (i + 1) * elements_per_bin
        
        # Extract the elements of the current bin
        bin_values = num_nodes_per_timestep[start_index:end_index]
        
        # Calculate the mean of the bin values and append it to the list
        bin_mean = np.mean(bin_values)
        bin_sum = np.sum(bin_values)
        sums.append(bin_sum)
        means.append(bin_mean)
        try:
            ts_discretized_min.append(min(bin_values))
            ts_discretized_max.append(max(bin_values))
        except:
            ts_discretized_min.append(0)
            ts_discretized_max.append(0)

        start_indices.append(start_index)
        end_indices.append(end_index)
        mid_indices.append((start_index + end_index) // 2)
    
    return means, sums, ts_discretized_min, ts_discretized_max, start_indices, end_indices, mid_indices

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


def compute_rec_drec(triples_for_rel, rel, ts_all):
    """ Compute the recurrency and direct recurrency for a given relation.
    :param triples_for_rel: list of  triples for a given relation [s, o, t]
    :param rel: int, relation
    :param ts_all: object of class Tripleset
    :return: 
    float: recurrency degree
    float: direct recurrency degree"""
    count_previous = 0
    count_sometime = 0
    count_all = 0
    for qtriple in triples_for_rel:    
        (s,r,o,t) = qtriple[0], rel, qtriple[1], qtriple[2]
        k = ts_all.get_latest_ts(s,r,o, t)
        count_all += 1
        if k + 1 == t: count_previous += 1
        if k > -1 and k < t: count_sometime += 1

    direct_recurrency_degree = count_previous / count_all
    recurrency_degree = count_sometime / count_all
    return recurrency_degree, direct_recurrency_degree


def compute_consecutiveness(sub_obj_t_for_rel, ts_all):
    """ Compute the consecutiveness for a given relation."""
    ts_lists = ts_all.create_timestep_lists(sub_obj_t_for_rel) # list that contains for each [s,o] the list of ts
    all_cons = []
    for list_r in ts_lists:
        max_cn = max_consecutive_numbers(list_r) #maximum consecutive number of timesteps for each [s,o]
        all_cons.append(max_cn)
    mean_con = np.mean(all_cons)
    return mean_con

def compute_number_distinct_triples(sub_obj_t_for_rel):
    """ Compute the number of distinct triples for a given relation.
    :param sub_obj_t_for_rel: dict, {sub: [ob, t]}"""
    counter = 0
    for sub, ob_t in sub_obj_t_for_rel.items():
        counter += len(ob_t)

    return counter

def read_dict_compute_mrr_per_rel(perrel_results_path, model_name, dataset_name, seed, num_rels, split_mode='test'):
    """ Read the csv file and compute the MRR per relation.
    :param perrel_results_path: str, path to the directory containing the results csv files,
    :param model_name: str, name of the model,
    :param dataset_name: str, name of the dataset,
    :param seed: int, seed used for the experiment,
    :param num_rels: int, number of relations,
    :param split_mode: str, split mode (val, test),
    :return: 
    dict, MRR per relation.
    float, average MRR across all triples.
    """
    mrr_per_rel = {}

    csv_file = f'{perrel_results_path}/{model_name}_NONE_{dataset_name}_results_{seed}'+split_mode+'.csv'
    # Initialize an empty dictionary to store the data
    results_per_rel_dict = {}

    all_mrrs = []
    # Open the file for reading
    with open(csv_file, 'r') as f:
        # Read each line in the file
        for line in f:
            # Split the line at the comma
            parts = line.strip().split(',')
            # Extract the key (the first part)
            key = int(parts[0])
            # Extract the values (the rest of the parts), remove square brackets
            values = [float(value.strip('[]')) for value in parts[1:]]
            # Add the key-value pair to the dictionary
            if key in results_per_rel_dict.keys():
                print(f"Key {key} already exists in the dictionary!!! might have duplicate entries in results csv")
            results_per_rel_dict[key] = values
            all_mrrs.extend(values)
            mrr_per_rel[key] = np.mean(values)

    if len(list(results_per_rel_dict.keys())) != num_rels:
        print("we do not have entries for each rel in the results csv file. only num enties: ", len(list(results_per_rel_dict.keys())))

    print("Split mode: "+split_mode +" Mean MRR: ", np.mean(all_mrrs))
    print("mrr per relation: ", mrr_per_rel)

    return mrr_per_rel, np.mean(all_mrrs)


def set_plot_names(top_k, sorted_dict, dataset_name, rel_id2type_dict):
    """ Set the plot names for the relations.
    :param top_k: dict, top k relations,
    :param sorted_dict: dict, sorted dictionary by descending order of the relation occurrences,
    :param dataset_name: str, name of the dataset,
    :param rel_id2type_dict: dict, relation id to type dictionary,
    :return: dict, plot names for the relations.
    """

    # string_names ={}
    names = {}
    plot_names = {}

    # names['P131'] = 'located in the administrative territorial entity (P131)'
    # names['P39'] = 'position held (P39)'
    # names['P17'] = 'country (P17)'
    # names['P937'] = 'work location (P937)'
    # names['P127']= 'owned by (P127)'
    # names['P26'] = 'spouse (P26)'
    # names['P611'] = 'religious order (P611)'
    # names['P276'] = 'location (P276)'
    # names['P1376'] = 'capital of (P1376)'
    # names['P793']= 'significant event (P793)'
    # #TODO: add for wikidata

    for key, val in top_k.items():
        
        if 'wiki' in dataset_name or 'smallpedia' in dataset_name:
            name = fetch_wikidata_property_name(rel_id2type_dict[key])
            # string_names[rel_id2type_dict[key]] = val
            plot_names[str(key)+': ' + name] = val
        else:
            # string_names[str(key)] = val
            plot_names[str(key)] = val
    others_value = sum(sorted_dict.values()) - sum(top_k.values())

    # Create a new dictionary with the top k key-value pairs and the sum of the remaining values as "others"
    plot_names = {**plot_names, 'others': others_value}
    return plot_names


import requests
import re

def fetch_wikidata_property_name(property_id):
    url = f"https://www.wikidata.org/wiki/Property:{property_id}"
    response = requests.get(url)

    if response.status_code == 200:
        # Use regular expression to find the title tag content
        match = re.search(r'<title>(.*?) - Wikidata</title>', response.text)
        
        if match:
            property_name = match.group(1).strip()
            return f"{property_name} ({property_id})"
        else:
            return property_id
    else:
        return property_id
