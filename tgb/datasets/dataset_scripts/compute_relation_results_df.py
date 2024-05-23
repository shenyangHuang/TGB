import numpy as np

import sys
import os
import os.path as osp
tgb_modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(tgb_modules_path)
import json

## imports
import numpy as np
import pandas as pd
import tgb.datasets.dataset_scripts.dataset_utils as du




# specify params
names = [ 'tkgl-icews', 'tkgl-polecat'] #'tkgl-polecat','tkgl-smallpedia',  'tkgl-yago',  'tkgl-icews' ,'tkgl-smallpedia','thgl-myket','tkgl-yago',  'tkgl-icews','thgl-github', 'thgl-forum', 'tkgl-wikidata']
methods = [ 'regcn'] #'recurrency'
colortgb = '#60ab84'
colortgb2 = '#eeb641'
colortgb3 = '#dd613a'
#colortgb4 ='#bce9ef'
#colortgb5 ='#d6e9d9'

colors = [colortgb,colortgb2,colortgb3]  # from tgb logo
capsize=1.5
capthick=1.5
elinewidth=1.5
occ_threshold = 5
k=10 # how many slices in the cake +1
plots_flag = True

model_names = {'recurrency': {'tkgl-polecat': ['saved_models/RecurrencyBaseline', 1],
                              'tkgl-icews': ['saved_models/RecurrencyBaseline', 500]},
               'regcn': {'tkgl-polecat': 'saved_results/REGCN_tkgl-polecat_results_per_rel.json',
                         'tkgl-icews': 'saved_results/REGCN_tkgl-icews_results_per_rel.json'}}
# run through each datasest
for dataset_name in names:
    # read dataframe with the stats for this dataset from csv
    print(dataset_name)
    modified_dataset_name = dataset_name.replace('-', '_')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate one folder up
    parent_dir = os.path.dirname(current_dir)

    tgb_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    figs_dir = os.path.join(parent_dir, modified_dataset_name, 'figs')
    # Create the 'figs' directory if it doesn't exist
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    stats_dir = os.path.join( parent_dir, modified_dataset_name, 'stats')
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    stats_df = pd.read_csv(os.path.join(stats_dir, f"relation_statistics_{dataset_name}.csv"))
    
    for method in methods:
        results_dict = os.path.join(tgb_dir, 'examples', 'linkproppred', dataset_name)


        # if the method is recurrency, we need to load the results from the csv and compute the mean value per relation
        # load csv, create dict with the mean value per relation
        if method == 'recurrency':
            name = model_names[method][dataset_name][0]
            seed = model_names[method][dataset_name][1]
            results_filename = f'{results_dict}/{name}'
            # results_df = pd.read_csv(results_filename)
            # for each relation, compute the mean value of the relation_mrr
            # csv i
            mrr_per_rel, full_mrr = du.read_dict_compute_mrr_per_rel(results_dict, name, dataset_name, seed, num_rels=0, split_mode='test')
        
        else:
            name = model_names[method][dataset_name]
            results_filename = f'{results_dict}/{name}'
            with open(results_filename, 'r') as json_file:
                mrr_per_rel = json.load(json_file)


        # else we can just load the results from the json file
        # load json where for each relation the results are stored

        # for each entry in the stats dataframe: append a column for each method with the mean value of the relation_mrr
        # if the column is not present, add it
        # if the column is present, append the value
        if method in stats_df.columns:
            print('Column already present')
        else:
            # each line of the original dataframe has a relation id
            for rel in stats_df['relation']:
                if str(rel) in mrr_per_rel:
                    stats_df.loc[stats_df['relation'] == rel, method] = mrr_per_rel[str(rel)]
                elif rel in mrr_per_rel:
                    stats_df.loc[stats_df['relation'] == rel, method] = mrr_per_rel[rel]
                else:
                    stats_df.loc[stats_df['relation'] == rel, method] = 'N/A'
        # save the dataframe with the new columns
        stats_df.to_csv(os.path.join(stats_dir, f"relation_statistics_{dataset_name}.csv"), index=False)

        # save dataframe with the new columns
