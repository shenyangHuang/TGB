"""
This file is only required to be executed once for generating the required file for a TGN model

Date:
  - March 04, 2023
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse



def UNTrade_preproc(filename):
  """
  preprocess .csv file of UN-Trade datasets
  NOTE:
    - there is a dummy 'state_label' column that we do not use
    - 'w' column should passed separately: this is the target of prediction for the edge regression tasks
    - this dataset is unattributed; i.e., there is neither node nor edge features
  """
  src_list, dst_list, ts_list, w_list = [], [], [], []
  idx_list = []
  
  with open(filename) as f:
    s = next(f)  # first line is the header
    for idx, line in enumerate(f):
      elements = line.split(',')
      src_list.append(int(elements[0]))
      dst_list.append(int(elements[1]))
      ts_list.append(float(elements[2]))
      w_list.append(float(elements[4]))  # elements[3] is the dummy 'state_label', so we skip it
      idx_list.append(idx)

  # re-index the nodes
  new_src_list = np.array(src_list) + 1
  new_dst_list = np.array(dst_list) + 1
  new_idx_list = np.array(idx_list) + 1
  
  data_df = pd.DataFrame(zip(new_src_list, new_dst_list, ts_list, new_idx_list, w_list),
                    columns=['src', 'dst', 'ts', 'idx', 'w'])
  return data_df


def UNTrade_gen_ml_files(data_df, partial_path, data_name, e_dim=172, n_dim=172):
    """
    generate 'ml' files for the UN-Trade for Edge Regression task
    NOTE:
      - data_df is already re-indexed
      - there is no edge or node features for the UN-Trade dataset, should be generated here (following the convention, they are initialized as all zeros)
    """
    Path(partial_path).mkdir(parents=True, exist_ok=True)
    OUT_DF_PATH = f"{partial_path}/ml_{data_name}.csv"
    OUT_EDGE_FEAT = f"{partial_path}/ml_{data_name}.npy"
    OUT_NODE_FEAT = f"{partial_path}/ml_{data_name}_node.npy"

    # timestamped edge list
    data_df.to_csv(OUT_DF_PATH, index=False)

    # edge features
    e_feat = np.zeros((data_df.shape[0], e_dim))
    np.save(OUT_EDGE_FEAT, e_feat)

    # node features
    max_node_id = max(data_df['src'].max(), data_df['dst'].max())
    n_feat = np.zeros((max_node_id + 1, n_dim))
    np.save(OUT_NODE_FEAT, n_feat)

    print(f"INFO: {data_name}: Dataset Preprocessing is done!")


def main():
  """
  Procedures to pre-process a dataset from its raw format
  """
  parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
  parser.add_argument('--data', type=str, help='Dataset name', default='UNtrade')

  args = parser.parse_args()

  # parameters
  DATA = args.data
  partial_path = f'./data/'
  raw_filename = f'{partial_path}/{DATA}.csv'

  # pre-processing
  if DATA == 'UNtrade':
    data_df = UNTrade_preproc(raw_filename)
    UNTrade_gen_ml_files(data_df, partial_path, DATA)
  else:
    raise ValueError("Dataset Pre-Processing not implemented yet!")



if __name__ == '__main__':
  main()





      


