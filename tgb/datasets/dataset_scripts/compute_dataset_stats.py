import numpy as np

import sys
import os
import os.path as osp
tgb_modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(tgb_modules_path)
import json

## imports
import numpy as np
from datetime import datetime
#internal imports 
from tgb.linkproppred.dataset import LinkPropPredDataset 
from tgb_modules.tkg_utils import  reformat_ts, get_original_ts
import tgb.datasets.dataset_scripts.dataset_utils as du

import networkx as nx
import matplotlib.pyplot as plt

from datetime import datetime


names = ['tkgl-polecat', 'tkgl-smallpedia','tkgl-polecat',  'thgl-software', 'tkgl-icews','thgl-github', 'thgl-forum', 'tkgl-wikidata', 'thgl-myket','tkgl-yago']
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    modified_dataset_name = dataset_name.replace('-', '_')
    csv_dir = os.path.join( parent_dir, modified_dataset_name)
    np.savetxt(csv_dir +"/"+dataset_name+"timestamps.csv", timestamps,fmt='%i', delimiter=",")
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

    first_ts = timestamps_orig[0]
    last_ts = timestamps_orig[-1]

    if 'wikidata' in dataset_name or 'smallpedia' in dataset_name or 'yago' in dataset_name:
        first_ts_string = str(first_ts)
        last_ts_string = str(last_ts)
    elif 'thgl' in dataset_name:
        first_ts_string = datetime.utcfromtimestamp(first_ts).strftime('%Y-%m-%d %H:%M:%S')
        last_ts_string = datetime.utcfromtimestamp(last_ts).strftime('%Y-%m-%d %H:%M:%S')
    else:
        first_ts_string = datetime.utcfromtimestamp(first_ts).strftime('%Y-%m-%d')
        last_ts_string = datetime.utcfromtimestamp(last_ts).strftime('%Y-%m-%d')

    print(dataset_name, "first timestamp:", first_ts_string, "last timestamp:", last_ts_string)
    

    # compute number of quads in train/val/test set
    num_train_quads = train_data.shape[0]
    num_val_quads = val_data.shape[0]
    num_test_quads = test_data.shape[0]
    num_all_quads = num_train_quads + num_val_quads + num_test_quads
    print(num_all_quads)

    # compute inductive nodes
    test_ind_nodes = du.num_nodes_not_in_train(train_data, test_data)
    val_ind_nodes = du.num_nodes_not_in_train(train_data, val_data)
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
    ts_all = du.TripleSet()
    ts_all.add_triples(all_quads, num_rels_without_inv, timestep_range)
    ts_all.compute_stat()
    ts_test = du.TripleSet()
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
    results = [du.max_consecutive_numbers(inner_list) for inner_list in ts_all.timestep_lists]
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
    seasonal_value =1
    seasonal_value = du.estimate_seasons(n_nodes_list)
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
    colortgb = '#60ab84'
    fontsize =12
    labelsize=12
    bars_list = [20]
    for num_bars in bars_list:
        if num_bars < 100:
            capsize=2
            capthick=2
            elinewidth=2
        else:
            capsize=1 
            capthick=1
            elinewidth=1
        ts_discretized_mean, ts_discretized_sum, ts_discretized_min, ts_discretized_max, start_indices, end_indices, mid_indices = du.discretize_values(n_edges_list, num_bars)
        plt.figure()
        plt.tick_params(axis='both', which='major', labelsize=labelsize)
        # plt.bar(mid_indices, ts_discretized_mean, width=(len(n_edges_list) // num_bars), label ='Mean Value', color =colortgb)
        plt.step(mid_indices, ts_discretized_mean, where='mid', linestyle='-', label ='Mean Value', color=colortgb)
        plt.scatter(mid_indices, ts_discretized_min, label ='min value')
        plt.scatter(mid_indices, ts_discretized_max, label ='max value')
        plt.xlabel('Timestep', fontsize=fontsize)
        plt.ylabel('Number of Edges', fontsize=fontsize)
        plt.legend()
        #plt.title(dataset_name+ ' - Number of Edges aggregated across multiple timesteps')
        modified_dataset_name = dataset_name.replace('-', '_')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate one folder up
        parent_dir = os.path.dirname(current_dir)
        figs_dir = os.path.join( parent_dir, modified_dataset_name, 'figs')
        # Create the 'figs' directory if it doesn't exist
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)
        save_path = (os.path.join(figs_dir, f"num_edges_discretized_{num_bars}_{dataset_name}.png"))
        plt.savefig(save_path, bbox_inches='tight')
        save_path = (os.path.join(figs_dir, f"num_edges_discretized_{num_bars}_{dataset_name}.pdf"))
        plt.savefig(save_path, bbox_inches='tight')

        plt.figure()
        plt.tick_params(axis='both', which='major', labelsize=labelsize)
        mins = np.array(ts_discretized_min)
        maxs = np.array(ts_discretized_max)
        means = np.array(ts_discretized_mean)
        # plt.bar(mid_indices, ts_discretized_mean, width=(len(n_edges_list) // num_bars), label='Mean', color =colortgb)
        plt.step(mid_indices, ts_discretized_mean, where='mid', linestyle='-', label ='Mean Value', color=colortgb, linewidth=2)
        #plt.scatter(mid_indices, ts_discretized_mean, label ='Mean Value', color=colortgb)
        plt.errorbar(mid_indices, maxs, yerr=[maxs-mins, maxs-maxs], fmt='none', alpha=0.9, color='grey',capsize=capsize, capthick=capthick, elinewidth=elinewidth, label='Min-Max Range')
        plt.xlabel('Timestep', fontsize=fontsize)
        plt.ylabel('Number of Edges', fontsize=fontsize)
        plt.legend()
        #plt.title(dataset_name+ ' - Number of Edges aggregated across multiple timesteps')
        plt.show()
        save_path2 = (os.path.join(figs_dir,f"num_edges_discretized_{num_bars}_{dataset_name}2.png"))
        plt.savefig(save_path2, bbox_inches='tight')
        save_path2 = (os.path.join(figs_dir,f"num_edges_discretized_{num_bars}_{dataset_name}2.pdf"))
        plt.savefig(save_path2, bbox_inches='tight')

        plt.figure()
        plt.tick_params(axis='both', which='major', labelsize=labelsize)
        mins = np.array(ts_discretized_min)
        maxs = np.array(ts_discretized_max)
        means = np.array(ts_discretized_mean)
        plt.bar(mid_indices, ts_discretized_sum, width=(len(n_edges_list) // num_bars), label='Sum', color =colortgb)
        # plt.step(mid_indices, ts_discretized_mean, where='mid', linestyle='-', label ='Mean Value', color=colortgb)
        # plt.errorbar(mid_indices, sums, yerr=[mins, maxs], fmt='none', alpha=0.9, color='grey',capsize=1.5, capthick=1.5, elinewidth=2, label='Min-Max Range')
        plt.xlabel('Timestep', fontsize=fontsize)
        plt.ylabel('Number of Edges', fontsize=fontsize)
        plt.legend()
        #plt.title(dataset_name+ ' - Number of Edges aggregated across multiple timesteps')
        plt.show()
        save_path2 = (os.path.join(figs_dir,f"num_edges_discretized_{num_bars}_{dataset_name}3.png"))
        plt.savefig(save_path2, bbox_inches='tight')
        save_path2 = (os.path.join(figs_dir,f"num_edges_discretized_{num_bars}_{dataset_name}3.pdf"))
        plt.savefig(save_path2, bbox_inches='tight')

        try:
            plt.figure()
            plt.tick_params(axis='both', which='major', labelsize=labelsize)
            mins = np.array(ts_discretized_min)
            maxs = np.array(ts_discretized_max)
            means = np.array(ts_discretized_mean)
            # plt.bar(mid_indices, ts_discretized_mean, width=(len(n_edges_list) // num_bars), label='Mean', color =colortgb)
            plt.step(mid_indices, ts_discretized_mean, where='mid', linestyle='-', label ='Mean Value', color=colortgb)
            #plt.scatter(mid_indices, ts_discretized_mean, label ='Mean Value', color=colortgb)
            plt.errorbar(mid_indices, maxs, yerr=[maxs-mins, maxs-maxs], fmt='none', alpha=0.9, color='grey',capsize=capsize, capthick=capthick, elinewidth=elinewidth, label='Min-Max Range')
            plt.xlabel('Timestep', fontsize=fontsize)
            plt.ylabel('Number of Edges', fontsize=fontsize)
            #plt.title(dataset_name+ ' - Number of Edges aggregated across multiple timesteps')
            plt.yscale('log')
            plt.legend(fontsize=fontsize)
            plt.show()
            save_path2 = (os.path.join(figs_dir,f"num_edges_discretized_{num_bars}_{dataset_name}2log.png"))
            plt.savefig(save_path2, bbox_inches='tight')
            save_path2 = (os.path.join(figs_dir,f"num_edges_discretized_{num_bars}_{dataset_name}2log.pdf"))
            plt.savefig(save_path2, bbox_inches='tight')
        except:
            print('Could not plot log scale')
        plt.close('all')
        
    plt.figure()
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.scatter(range(ts_all.num_timesteps), n_edges_list, s=0.2)
    plt.xlabel('Timestep', fontsize=fontsize)
    plt.ylabel('number of triples', fontsize=fontsize)
    #plt.title(f'Number of triples per timestep for {dataset_name}')
    # save
    # Get the current directory of the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate one folder up
    parent_dir = os.path.dirname(current_dir)
    # Save stats_dict as CSV
    modified_dataset_name = dataset_name.replace('-', '_')
    save_path = (os.path.join(figs_dir,f"num_edges_per_ts_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')

    to_be_saved_dict = {}
    to_be_saved_dict['num_edges'] = n_edges_list
    to_be_saved_dict['num_nodes'] = n_nodes_list
    parent_dir = os.path.dirname(current_dir)
    save_path = (os.path.join(figs_dir,f"numedges_{dataset_name}.json")) 
    save_file = open(save_path, "w")
    json.dump(to_be_saved_dict, save_file)
    save_file.close()

    plt.figure()
    plt.scatter(range(ts_all.num_timesteps), n_nodes_list, s=0.2)
    plt.xlabel('Timestep', fontsize=fontsize)
    plt.ylabel('number of nodes', fontsize=fontsize)
    #plt.title(f'Number of nodes per timestep for {dataset_name}')
    save_path = (os.path.join(figs_dir,f"num_nodes_per_ts_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')
    
   

    du.create_dict_and_save(dataset_name, num_rels_without_inv, num_nodes, num_train_quads, num_val_quads, num_test_quads, 
                        num_all_quads, num_train_timesteps, num_val_timesteps, num_test_timesteps, num_all_ts,
                        test_ind_nodes, test_ind_nodes_perc, val_ind_nodes, val_ind_nodes_perc, 
                        direct_recurrency_degree, recurrency_degree, consecutiveness_degree,
                        np.mean(n_edges_list), np.std(n_edges_list), np.min(n_edges_list), np.max(n_edges_list),
                        np.mean(n_nodes_list), np.std(n_nodes_list), np.min(n_nodes_list), np.max(n_nodes_list),
                        seasonal_value, collision_trainval, collision_valtest, first_ts_string, last_ts_string)
