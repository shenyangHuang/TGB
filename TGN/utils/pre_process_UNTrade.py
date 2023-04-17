"""
Preprocessing the UN-Trade datasets
In particular, processing the edge weights so that the models can be trained on this dataset.

Date:
    - March 13, 2023
"""
import numpy as np 
import pandas as pd
import networkx as nx 




def gen_src_ts_sum_weight(edgelist_df):
    """
    generates a dictionary where the keys are (src, ts) and 
    the values are the sum of all edge weights at that timestamp with the same source node
    """
    src_ts_sum_w = {}
    for idx, row in edgelist_df.iterrows():
        if (row['src'], row['ts']) not in src_ts_sum_w:
            src_ts_sum_w[(row['src'], row['ts'])] = row['w']
        else:
            src_ts_sum_w[(row['src'], row['ts'])] += row['w']

    return src_ts_sum_w


def normalize_weight_wtd(edgelist_df, src_ts_sum_w):
    """
    Normalize the edge weights by the weighted temporal degrees
    """
    normal_weights = []
    for idx, row in edgelist_df.iterrows():
        sum_weight = src_ts_sum_w[(row['src'], row['ts'])]
        if sum_weight != 0:
            normal_weights.append(row['w']/sum_weight)
        else:
            normal_weights.append(0)

    edgelist_df['w'] = normal_weights

    return edgelist_df

    

def main():
    original_filename = './data/ml_UNtrade_original_weights.csv'
    processed_filename = './data/ml_UNtrade.csv'

    # read the original edgelist
    edgelist_df = pd.read_csv(original_filename)
    print("INFO: original edgelist:\n", edgelist_df.head())
    
    src_ts_sum_w = gen_src_ts_sum_weight(edgelist_df)
    processed_edgelist_df = normalize_weight_wtd(edgelist_df, src_ts_sum_w)
    processed_edgelist_df.to_csv(processed_filename, index=False)
    print("INFO: processed edgelist:\n", processed_edgelist_df.head())






if __name__ == '__main__':
    main()