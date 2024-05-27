import numpy as np

import sys
import os
import os.path as osp
tgb_modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(tgb_modules_path)
import json

## imports
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import tgb.datasets.dataset_scripts.dataset_utils as du




# specify params
names = [ 'tkgl-polecat', 'tkgl-icews',  'tkgl-smallpedia', 'tkgl-wikidata', 'thgl-myket','tkgl-yago', 'thgl-github', 'thgl-forum']
methods = ['recurrency', 'regcn', 'cen'] #'recurrency'
colortgb = '#60ab84'
colortgb2 = '#eeb641'
colortgb3 = '#dd613a'
#colortgb4 ='#bce9ef'
#colortgb5 ='#d6e9d9'

fontsize =12
labelsize=12
for dataset_name in names:    
    modified_dataset_name = dataset_name.replace('-', '_')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate one folder up
    parent_dir = os.path.dirname(current_dir)
    figs_dir = os.path.join( parent_dir, modified_dataset_name, 'figs')
    save_path = (os.path.join(figs_dir,f"numedges_{dataset_name}.json")) 

    n_edgesnodes_list_all = json.load(open(save_path)) 
    n_edges_list = n_edgesnodes_list_all['num_edges']
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
        
    # plt.figure()
    # plt.tick_params(axis='both', which='major', labelsize=labelsize)
    # plt.scatter(range(ts_all.num_timesteps), n_edges_list, s=0.2)
    # plt.xlabel('Timestep', fontsize=fontsize)
    # plt.ylabel('number of triples', fontsize=fontsize)
    # #plt.title(f'Number of triples per timestep for {dataset_name}')
    # # save
    # # Get the current directory of the script
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # # Navigate one folder up
    # parent_dir = os.path.dirname(current_dir)
    # # Save stats_dict as CSV
    # modified_dataset_name = dataset_name.replace('-', '_')
    # save_path = (os.path.join(figs_dir,f"num_edges_per_ts_{dataset_name}.png"))
    # plt.savefig(save_path, bbox_inches='tight')

    # to_be_saved_dict = {}
    # to_be_saved_dict['num_edges'] = n_edges_list
    # to_be_saved_dict['num_nodes'] = n_nodes_list
    # parent_dir = os.path.dirname(current_dir)
    # save_path = (os.path.join(figs_dir,f"numedges_{dataset_name}.json")) 
    # save_file = open(save_path, "w")
    # json.dump(to_be_saved_dict, save_file)
    # save_file.close()

    # plt.figure()
    # plt.scatter(range(ts_all.num_timesteps), n_nodes_list, s=0.2)
    # plt.xlabel('Timestep', fontsize=fontsize)
    # plt.ylabel('number of nodes', fontsize=fontsize)
    # #plt.title(f'Number of nodes per timestep for {dataset_name}')
    # save_path = (os.path.join(figs_dir,f"num_nodes_per_ts_{dataset_name}.png"))
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.close('all')