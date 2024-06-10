import numpy as np

import sys
import os
import os.path as osp
tgb_modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(tgb_modules_path)
import json
import csv 

## imports
import matplotlib.pyplot as plt
import numpy as np
import stats_figures.dataset_utils as du


# specify params
names = ['thgl-github', 'tkgl-polecat', 'tkgl-icews',  'tkgl-smallpedia', 'tkgl-wikidata', 'thgl-myket',  'thgl-forum', 'thgl-software']
granularity ={} #for labels
granularity['tkgl-polecat'] = 'days'
granularity['tkgl-icews'] = 'days'
granularity['tkgl-smallpedia'] = 'years'
granularity['tkgl-wikidata'] = 'years'
granularity['tkgl-yago'] = 'years'
granularity['thgl-myket'] = 's.'
granularity['thgl-github'] = 's.'
granularity['thgl-software'] = 's.'
granularity['thgl-forum'] = 's.'

# colors from tgb logo
colortgb = '#60ab84'
colortgb2 = '#eeb641'
colortgb3 = '#dd613a'
#colortgb4 ='#bce9ef'
#colortgb5 ='#d6e9d9'


fontsize =12
labelsize=12
for dataset_name in names:    

    # dataset_name = dataset_name.replace('-', '_')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate one folder up

    figs_dir = os.path.join( current_dir, dataset_name, 'figs')
    data_dir =  os.path.join(current_dir, dataset_name)
    save_path = (os.path.join(figs_dir,f"numedges_{dataset_name}.json")) 
    stats_path = (os.path.join(data_dir,f"dataset_stats.csv"))

    n_edgesnodes_list_all = json.load(open(save_path)) 
    n_edges_list = n_edgesnodes_list_all['num_edges']
    bars_list = [20] #number of bins

    # Read the CSV file
    with open(stats_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == 'first_ts_string':
                start_date = row[1]
            elif row[0] == 'last_ts_string':
                end_date = row[1]

    for num_bars in bars_list:
        # Create the 'figs' directory if it doesn't exist
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)
        if num_bars < 100:
            capsize=2
            capthick=2
            elinewidth=2
        else:
            capsize=1 
            capthick=1
            elinewidth=1
        ts_discretized_mean, ts_discretized_sum, ts_discretized_min, ts_discretized_max, start_indices, end_indices, mid_indices = du.discretize_values(n_edges_list, num_bars)
        
        # line chart
        plt.figure()
        plt.tick_params(axis='both', which='major', labelsize=labelsize)
        mins = np.array(ts_discretized_min)
        maxs = np.array(ts_discretized_max)
        means = np.array(ts_discretized_mean)
        # plt.bar(mid_indices, ts_discretized_mean, width=(len(n_edges_list) // num_bars), label='Mean', color =colortgb)
        plt.step(mid_indices, ts_discretized_mean, where='mid', linestyle='-', label ='Mean Value', color=colortgb, linewidth=2)
        #plt.scatter(mid_indices, ts_discretized_mean, label ='Mean Value', color=colortgb)
        plt.errorbar(mid_indices, maxs, yerr=[maxs-mins, maxs-maxs], fmt='none', alpha=0.9, color='grey',capsize=capsize, capthick=capthick, elinewidth=elinewidth, label='Min-Max Range')
        plt.xlabel(f'Ts. [{granularity[dataset_name]}] from {start_date} to {end_date}', fontsize=fontsize)
        plt.ylabel('Number of Edges', fontsize=fontsize)
        plt.legend()
        plt.tight_layout()
        #plt.title(dataset_name+ ' - Number of Edges aggregated across multiple timesteps')
        plt.show()
        save_path2 = (os.path.join(figs_dir,f"num_edges_discretized_{num_bars}_{dataset_name}2.png"))
        plt.savefig(save_path2, bbox_inches='tight')
        save_path2 = (os.path.join(figs_dir,f"num_edges_discretized_{num_bars}_{dataset_name}2.pdf"))
        plt.savefig(save_path2, bbox_inches='tight')

        # bar chart
        plt.figure()
        plt.tick_params(axis='both', which='major', labelsize=labelsize)
        mins = np.array(ts_discretized_min)
        maxs = np.array(ts_discretized_max)
        means = np.array(ts_discretized_mean)
        plt.bar(mid_indices, ts_discretized_sum, width=(len(n_edges_list) // num_bars), label='Sum', color =colortgb)
        # plt.step(mid_indices, ts_discretized_mean, where='mid', linestyle='-', label ='Mean Value', color=colortgb)
        # plt.errorbar(mid_indices, sums, yerr=[mins, maxs], fmt='none', alpha=0.9, color='grey',capsize=1.5, capthick=1.5, elinewidth=2, label='Min-Max Range')
        plt.xlabel(f'Timestep [{granularity[dataset_name]}] from {start_date} to {end_date}', fontsize=fontsize)
        plt.ylabel('Number of Edges', fontsize=fontsize)
        plt.legend()
        #plt.title(dataset_name+ ' - Number of Edges aggregated across multiple timesteps')
        plt.show()
        save_path2 = (os.path.join(figs_dir,f"num_edges_discretized_{num_bars}_{dataset_name}3.png"))
        plt.savefig(save_path2, bbox_inches='tight')
        save_path2 = (os.path.join(figs_dir,f"num_edges_discretized_{num_bars}_{dataset_name}3.pdf"))
        plt.savefig(save_path2, bbox_inches='tight')


        try:
            # try log scale
            plt.figure()
            plt.tick_params(axis='both', which='major', labelsize=labelsize)
            mins = np.array(ts_discretized_min)
            maxs = np.array(ts_discretized_max)
            means = np.array(ts_discretized_mean)
            # plt.bar(mid_indices, ts_discretized_mean, width=(len(n_edges_list) // num_bars), label='Mean', color =colortgb)
            plt.step(mid_indices, ts_discretized_mean, where='mid', linestyle='-', label ='Mean Value', color=colortgb)
            #plt.scatter(mid_indices, ts_discretized_mean, label ='Mean Value', color=colortgb)
            plt.errorbar(mid_indices, maxs, yerr=[maxs-mins, maxs-maxs], fmt='none', alpha=0.9, color='grey',capsize=capsize, capthick=capthick, elinewidth=elinewidth, label='Min-Max Range')
            plt.xlabel(f'Timestep [{granularity[dataset_name]}] from {start_date} to {end_date}', fontsize=fontsize)
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
        
