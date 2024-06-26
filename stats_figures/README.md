### How to compute stats and figures

You run each of the scripts running 'python stats_figures/nameofthescript.py'
Each of them computes the stats or creates the figures automatically for all datasets. If you want to only focus on some subset of datasets you need to specify that in the list
'names = [ 'tkgl-polecat', ... ]' at the beginning of each script.
The same applies to methods of interest, if it is about correlation of methods results and stats.

For *creating the figures* you only need to run 2. and 5., provided that you have previously stored all stats in the respective dataset subfolder.


## 1. compute_dataset_stats.py
- loads datasets and computes all stats that we put in paper table and writes it in stats_figures/dataset_name/dataset_stats.csv 
- computes number of nodes and stores in stats_figures/dataset_name/figs/numedges_datasetname.json # number of edges per timestep (to create the figures)
- saves timestamps in timestamps.csv
- this can be a bit slow (takes a few hours for all datasets, especially computing seasonality is slow)

## 2. create_edges_figures.py
- makes the *figures with number of edges per timestep* (bins)
- input needed: dataset_name/numedges_datasetname.json and dataset_name/dataset_stats.csv  (output from 1.)
- output figures stored in dataset_name/num_edges_discretized_numbin_datasetname.pdf and png where numbin is the number of bins
- i usually use num_edges_discretized_{num_bars}_{dataset_name}2.pdf

## 3. compute_relation_dataset_stats.py
- compute the statistics for each relation based on the dataset, e.g. number of occurences, recurrency degree 
- outputs: csv file with relationship stats dataset_name/stats/relation_statistics_dataset_name.csv
- comment: for icews and polecat I manually added the strings for the 10 most occuring relations for the plots

## 4. compute_relation_results_df.py
- add the results for selected methods for each relation in head and tail direction to new columns to the dataset_name/stats/relation_statistics_dataset_name.csv
- input needed: relation_statistics_dataset_name.csv (from 4.) and: results_per_relation files e.g. examples/linkpropprediction/tkgl-polecat/saved_results/REGCN_tkgl-polecat_results_per_rel.json

## 5. create_relation_figures.py
- creates the figures for mrr per relation
- outputs: figures (*pie charts*)
- input needed: