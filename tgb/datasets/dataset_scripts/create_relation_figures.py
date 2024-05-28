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
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import tgb.datasets.dataset_scripts.dataset_utils as du




# specify params
names = [ 'tkgl-smallpedia','tkgl-icews', 'tkgl-polecat'] #'tkgl-polecat','tkgl-smallpedia',  'tkgl-yago',  'tkgl-icews' ,'tkgl-smallpedia','thgl-myket','tkgl-yago',  'tkgl-icews','thgl-github', 'thgl-forum', 'tkgl-wikidata']
methods = ['recurrency', 'regcn', 'cen'] #'recurrency'
colortgb = '#60ab84'
colortgb2 = '#eeb641'
colortgb3 = '#dd613a'
head_tail_flag = False # if true, the head and tail of the relation are shown in the plot, otherwise just the mean across both directions
#colortgb4 ='#bce9ef'
#colortgb5 ='#d6e9d9'

colors = [colortgb,colortgb2,colortgb3]  # from tgb logo
capsize=1.5
capthick=1.5
elinewidth=1.5
occ_threshold = 5
k=10 # how many slices in the cake +1
plots_flag = True
ylimdict = {'tkgl-polecat': 0.25, 'tkgl-icews':0.6, 'tkgl-smallpedia': 1.01}

model_names = {'recurrency': {'tkgl-polecat': ['saved_models/RecurrencyBaseline', 1],
                              'tkgl-icews': ['saved_models/RecurrencyBaseline', 500],
                              'tkgl-smallpedia': ['saved_models/RecurrencyBaseline', 1]},
               'regcn': {'tkgl-polecat': 'saved_results/REGCN_tkgl-polecat_results_per_rel.json',
                         'tkgl-icews': 'saved_results/REGCN_tkgl-icews_results_per_rel.json',
                         'tkgl-smallpedia': 'saved_results/REGCN_tkgl-smallpedia_results_per_rel.json'},
                'cen': {'tkgl-polecat': 'saved_results/CEN_tkgl-polecat_results_per_rel.json',
                        'tkgl-icews': 'saved_results/CEN_tkgl-icews_results_per_rel.json',
                        'tkgl-smallpedia': 'saved_results/CEN_tkgl-smallpedia_results_per_rel.json'}, 
                'tlogic': {'tkgl-smallpedia': 'saved_results/TLogic_tkgl-smallpedia_results_per_rel.json',
                            'tkgl-polecat': 'saved_results/TLogic_tkgl-polecat_results_per_rel.json'}
                } # where are the relations stored for each model and dataset

overall_min = -1 # for the correlation matrix colorbar
overall_max =1 # for the correlation matrix colorbar
num_rels_plot = 10 # how many relations to we want to plot
for dataset_name in names:
    # some directory stuff
    modified_dataset_name = dataset_name.replace('-', '_')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate one folder up
    parent_dir = os.path.dirname(current_dir)
    stats_dir = os.path.join( parent_dir, modified_dataset_name, 'stats')
    tgb_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    figs_dir = os.path.join(parent_dir, modified_dataset_name, 'figs_rel')
    stats_df = pd.read_csv(os.path.join(stats_dir, f"relation_statistics_{dataset_name}.csv"))

    # Create the 'figs' directory if it doesn't exist
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    stats_dir = os.path.join( parent_dir, modified_dataset_name, 'stats')
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)


    ### plot the mrr for each relation for each method, different color for different number of occurences or for different recurrency degree
    
    # prepare the dataframe: only take the top ten relations according to number of occurences and sort by recurrency degree
    # we use selected_df_sorted to plot the relations in the order of recurrency degree
    rels_sorted =  np.array(stats_df['relation'])[0:num_rels_plot]
    mask = stats_df['relation'].isin(rels_sorted)
    selected_df = stats_df[mask] #only the parts of the dataframe that contain the top ten relations according to number of occurences
    selected_df_sorted = selected_df.sort_values(by='recurrency_degree', ascending=False) # Sort selected_df by 'recurrency_degree' column in descending order
    rels_to_plot = list(selected_df_sorted['relation'])
    labels = np.array(selected_df_sorted['relation'])# 'rel_string_word'])

    mrr_per_rel_freq = [] # list of mrr values for each relation - three lists for three methods
    mrr_per_rel_freq2 = []
    mrr_per_rel_freq3 = []
    lab = []
    lab_ht = []
    lab_rel = []
    # rel_oc_dict[rel] = count_occurrences
    count_occurrences_sorted = []
    rec_degree_sorted = []
    for index, r in enumerate(rels_to_plot):   
        if head_tail_flag:
            lab_ht.append('h')
            lab_ht.append('t')
            lab_rel.append(str(labels[index])+'    ') # add spaces to make the labels longer
        
        lab.append(labels[index])
        if head_tail_flag: # if we do head and tail separately we need the value for head and tail direction
            mrr_per_rel_freq.append(selected_df_sorted['recurrency_head'].iloc[index])
            mrr_per_rel_freq.append(selected_df_sorted['recurrency_tail'].iloc[index])
            mrr_per_rel_freq2.append(selected_df_sorted['regcn_head'].iloc[index])
            mrr_per_rel_freq2.append(selected_df_sorted['regcn_tail'].iloc[index])
            mrr_per_rel_freq3.append(selected_df_sorted['cen_head'].iloc[index])
            mrr_per_rel_freq3.append(selected_df_sorted['cen_tail'].iloc[index])
            count_occurrences_sorted.append(selected_df_sorted['number_total_occurences'].iloc[index])#append twice for head and tail
            count_occurrences_sorted.append(selected_df_sorted['number_total_occurences'].iloc[index])
            rec_degree_sorted.append(selected_df_sorted['recurrency_degree'].iloc[index]) #append twice for head and tail
            rec_degree_sorted.append(selected_df_sorted['recurrency_degree'].iloc[index])
        else:# if we do  NOT head and tail separately we need the mean value for head and tail direction
            mrr_per_rel_freq.append(np.mean([selected_df_sorted['recurrency_head'].iloc[index], selected_df_sorted['recurrency_tail'].iloc[index]]))
            mrr_per_rel_freq2.append(np.mean([selected_df_sorted['regcn_head'].iloc[index],selected_df_sorted['regcn_tail'].iloc[index]]))
            mrr_per_rel_freq3.append(np.mean([selected_df_sorted['cen_head'].iloc[index], selected_df_sorted['cen_tail'].iloc[index]]))
            count_occurrences_sorted.append(selected_df_sorted['number_total_occurences'].iloc[index])#append twice for head and tail
            rec_degree_sorted.append(selected_df_sorted['recurrency_degree'].iloc[index])
       
    # rel_ids_range

    # these are the x-values of the ticks. in case we plot head and tail separately, we need to have two ticks per relation
    x_values = []
    x_values_rel = []
    for i in range(0,num_rels_plot):
        if head_tail_flag:
            x_values.append(i*2+0.4)
            x_values.append(i*2+0.8)
        else:
            x_values.append(i*2+0.4)
        x_values_rel.append(i*2+0.4)

    lab_lines = lab_rel #labels, for now
    a = count_occurrences_sorted 

    # vs 1) colors are the reucrrency degree
    plt.figure()
    sca = plt.scatter(x_values, mrr_per_rel_freq2,  marker='p',s=150,   c = rec_degree_sorted, alpha=1, edgecolor='grey',  cmap='jet',  norm=Normalize(vmin=0, vmax=1), label='REGCN')           # cmap='gist_rainbow',
    sca = plt.scatter(x_values, mrr_per_rel_freq3 , marker='*',s=150,   c = rec_degree_sorted, alpha=1,  edgecolor='grey', cmap='jet',  norm=Normalize(vmin=0, vmax=1), label='CEN')      
    sca = plt.scatter(x_values, mrr_per_rel_freq,   marker='o',s=60,    c = rec_degree_sorted, alpha=1,  edgecolor='grey', cmap='jet', norm=Normalize(vmin=0, vmax=1), label='Recurrency Baseline')
    plt.ylabel('MRR', fontsize=14) 
    plt.xlabel('Relation', fontsize=14) 
    plt.legend(fontsize=14)
    cbar =plt.colorbar(sca)
    plt.ylim([0,ylimdict[dataset_name]])
    cbar.ax.yaxis.label.set_color('gray')

    if head_tail_flag:
        plt.xticks(x_values, lab_ht, size=13) #, verticalalignment="center") #  ha='right', 
        plt.xticks(x_values_rel, lab_lines,  size=14, minor=True)
        plt.tick_params(axis='x', which='minor',  rotation=90,  length=0)
    else:
        plt.xticks(x_values_rel, lab_lines,  size=14)
        plt.tick_params(axis='x',  rotation=90,  length=0)
    plt.yticks(size=13)
    # Create a locator for the second set of x-ticks
    # plt.secondary_xaxis('top', x_values_rel)

    plt.grid()
    save_path = (os.path.join(figs_dir, f"rel_mrrperrel_recdeg_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')
    save_path = (os.path.join(figs_dir, f"rel_mrrperrel_recdeg_{dataset_name}.pdf"))
    plt.savefig(save_path, bbox_inches='tight')
    print('saved')


    # vs 2) colors are the number of occurences
    plt.figure()
    sca = plt.scatter(x_values, mrr_per_rel_freq2, marker='p',s=150,   c = a, alpha=1, edgecolor='grey', norm=LogNorm(), cmap='jet', label='REGCN')          
    sca = plt.scatter(x_values, mrr_per_rel_freq3 , marker='*',s=150,   c = a, alpha=1, edgecolor='grey', norm=LogNorm(), cmap='jet', label='CEN')      
    sca = plt.scatter(x_values, mrr_per_rel_freq,   marker='o',s=60,    c = a, alpha=1, edgecolor='grey', norm=LogNorm(), cmap='jet', label='Recurrency Baseline')
    plt.ylabel('MRR', fontsize=14) 
    plt.xlabel('Relation', fontsize=14) 
    plt.legend(fontsize=14)
    cbar =plt.colorbar(sca)
    plt.ylim([0,ylimdict[dataset_name]])
    cbar.ax.yaxis.label.set_color('gray')

    plt.xticks(x_values, lab_ht, size=13) #, verticalalignment="center") #  ha='right', 
    plt.yticks(size=13)
    # Create a locator for the second set of x-ticks
    # plt.secondary_xaxis('top', x_values_rel)
    plt.xticks(x_values_rel, lab_lines,  size=14, minor=True)
    plt.tick_params(axis='x', which='minor',  rotation=90,  length=0)
    plt.grid()
    save_path = (os.path.join(figs_dir, f"rel_mrrperrel_occ_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')
    
    ### now we plot all sorts of correlation matrix. I specify different columns for the different plots    
    df = stats_df[['recurrency_degree', 'direct_recurrency-degree', 'recurrency_tail', 'recurrency_head', 'regcn_tail', 'regcn_head', 'cen_tail', 'cen_head']]
    corrmat= df.corr()
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corrmat, fignum=f.number,  vmin=overall_min, vmax=overall_max)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    save_path = (os.path.join(figs_dir, f"corr_rec_meth_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')
    
    df = stats_df[['consecutiveness_value', 'recurrency_tail', 'recurrency_head', 'regcn_tail', 'regcn_head', 'cen_tail', 'cen_head']]
    corrmat= df.corr()
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corrmat, fignum=f.number,  vmin=overall_min, vmax=overall_max)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    save_path = (os.path.join(figs_dir, f"corr_con_meth_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')
    
    df = stats_df[['recurrency_degree', 'direct_recurrency-degree', 'consecutiveness_value', 'mean_occurence_per_triple','number_total_occurences',  'recurrency_tail', 'recurrency_head', 'regcn_tail', 'regcn_head', 'cen_tail', 'cen_head']]
    corrmat= df.corr()
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corrmat, fignum=f.number,  vmin=overall_min, vmax=overall_max)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16)
    for i in range(corrmat.shape[0]):
        for j in range(corrmat.shape[1]):
            plt.text(j, i, "{:.2f}".format(corrmat.iloc[i, j]), ha='center', va='center', color='black', fontsize=16)
    cb = plt.colorbar()
    # fig.colorbar(cax, ticks=[-1,0,1], shrink=0.8)
    cb.ax.tick_params(labelsize=16)    
    # Plot the correlation matrix
    save_path = (os.path.join(figs_dir, f"corr_all_meth_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')

print('done')