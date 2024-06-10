"""
for every method and dataset: load the mrr per relation and add it to new columns to the
dataset_name/stats/relation_statistics_dataset_name.csv dataframe
compute it separately for head and tail direction
for this, we need to have extracted the mrr per relation for each method and dataset
"""

## imports
import numpy as np
import sys
import os
import os.path as osp
tgb_modules_path = osp.abspath(os.path.join(os.path.dirname(__file__),  '..'))
sys.path.append(tgb_modules_path)
import json
import numpy as np
import pandas as pd
import stats_figures.dataset_utils as du


# specify params
names = [ 'tkgl-polecat', 'tkgl-icews', 'tkgl-polecat', 'tkgl-smallpedia'] #'tkgl-polecat','tkgl-smallpedia',  'tkgl-yago',  'tkgl-icews' ,'tkgl-smallpedia','thgl-myket','tkgl-yago',  'tkgl-icews','thgl-github', 'thgl-forum', 'tkgl-wikidata']
methods = ['recurrency', 'regcn', 'cen'] #'recurrency'

# this is where the results per relation are stored
model_names = {'recurrency': {'tkgl-polecat': ['saved_models/RecurrencyBaseline', 1],
                              'tkgl-icews': ['saved_models/RecurrencyBaseline', 500],
                              'tkgl-smallpedia': ['saved_models/RecurrencyBaseline', 1]},
               'regcn': {'tkgl-polecat': 'saved_results/REGCN_tkgl-polecat_results_per_rel.json',
                         'tkgl-icews': 'saved_results/REGCN_tkgl-icews_results_per_rel.json',
                         'tkgl-smallpedia': 'saved_results/REGCN_tkgl-smallpedia_results_per_rel.json'},
                'cen': {'tkgl-polecat': 'saved_results/CEN_tkgl-polecat_results_per_rel.json',
                        'tkgl-icews': 'saved_results/CEN_tkgl-icews_results_per_rel.json',
                        'tkgl-smallpedia': 'saved_results/CEN_tkgl-smallpedia_results_per_rel.json'}, 
                'tlogic': {'tkgl-smallpedia': 'saved_results/TLogic_tkgl-smallpedia_results_per_rel.json'}
                }

def inverse_rel(rel_id, max_id):
    inverse_rel = rel_id + max_id + 1
    return inverse_rel
# run through each datasest
for dataset_name in names:
    # read dataframe with the stats for this dataset from csv
    print(dataset_name)
    modified_dataset_name = dataset_name.replace('-', '_')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate one folder up
    parent_dir = os.path.dirname(current_dir)

    tgb_dir = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
    figs_dir = os.path.join(current_dir, dataset_name, 'figs')
    # Create the 'figs' directory if it doesn't exist
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    stats_dir = os.path.join( current_dir, dataset_name, 'stats')
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

            mrr_per_rel, full_mrr = du.read_dict_compute_mrr_per_rel(results_dict, name, dataset_name, seed, num_rels=0, split_mode='test')
        
        else:
            name = model_names[method][dataset_name]
            results_filename = f'{results_dict}/{name}'
            with open(results_filename, 'r') as json_file:
                mrr_per_rel = json.load(json_file)
        num_rels = len(list(set(stats_df['relation'])))
        max_id = max(list(set(stats_df['relation'])))
        assert num_rels == max_id+1

        for rel in stats_df['relation']:
            if str(rel) in mrr_per_rel:
                stats_df.loc[stats_df['relation'] == rel, method+'_tail'] = mrr_per_rel[str(rel)]
                stats_df.loc[stats_df['relation'] == rel, method+'_head'] = mrr_per_rel[str(inverse_rel(rel, max_id))]
            elif rel in mrr_per_rel:
                stats_df.loc[stats_df['relation'] == rel, method+'_tail'] = mrr_per_rel[rel]
                stats_df.loc[stats_df['relation'] == rel, method+'_head'] = mrr_per_rel[inverse_rel(rel, max_id)]
            else:
                stats_df.loc[stats_df['relation'] == rel, method+'_tail'] = 'N/A'
                stats_df.loc[stats_df['relation'] == rel, method+'_head'] = 'N/A'
        # save the dataframe with the new columns
        stats_df.to_csv(os.path.join(stats_dir, f"relation_statistics_{dataset_name}.csv"), index=False)

        # save dataframe with the new columns
