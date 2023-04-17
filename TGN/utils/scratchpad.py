"""
This file contains some code snippets that from time to time I used to check something...

Date:
    - March 06, 2023
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math





def main():
    """
    to call function(s) of interests...
    """
    DATA = 'UNtrade'
    original_filename = f'./data/ml_{DATA}_original_weights.csv'
    edgelist_df = pd.read_csv(original_filename)
    print("DEBUG: original_df --- original:")
    print(edgelist_df.head(10))

    selected_column = 'w'
    weight_list = np.array(edgelist_df['w'])
    # edgelist_df[selected_column] = MinMaxScaler().fit_transform(np.array(edgelist_df[selected_column]).reshape(-1,1))
    # edgelist_df[selected_column] = np.log(weight_list)
    # edgelist_df[selected_column] = np.where(weight_list != 0, np.log2(weight_list), 0)  # the current version as of March 06, 2023
    # edgelist_df[selected_column] = np.log(weight_list + 1)
    log_weight_list = np.where(weight_list != 0, np.log2(weight_list), 0)
    edgelist_df[selected_column] = MinMaxScaler().fit_transform(np.array(log_weight_list).reshape(-1,1))

    print("DEBUG: edgelist_df --- weight scaled:")
    print(edgelist_df.head(10))

    scaled_filename = f'./data/ml_{DATA}.csv'
    edgelist_df.to_csv(scaled_filename, index=False)


if __name__ == '__main__':
    main()