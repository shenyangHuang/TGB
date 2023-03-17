from typing import Optional, cast, Union, List, overload, Literal
import numpy as np
import pandas as pd


# TODO cleaning the un trade csv with countries with comma in the name, to remove this function
def clean_rows(
        fname: str,
        outname: str,
    ):
    r'''
    clean the rows with comma in the name
    args:
        fname: the path to the raw data
        outname: the path to the cleaned data
    '''

    outf = open(outname, "w")

    with open(fname) as f:
        s = next(f)
        outf.write(s)
        for idx, line in enumerate(f):
            strs = ['China, Taiwan Province of', 'China, mainland'] 
            for str in strs:
                    line = line.replace('China, Taiwan Province of', "Taiwan Province of China")
                    line = line.replace('China, mainland', "China mainland")
                    line = line.replace('China, Hong Kong SAR', "China Hong Kong SAR")
                    line = line.replace('China, Macao SAR', "China Macao SAR")
                    line = line.replace('Saint Helena, Ascension and Tristan da Cunha', 'Saint Helena Ascension and Tristan da Cunha')
                    
            
            e = line.strip().split(',')
            if (len(e) > 4):
                print (e)
                raise ValueError("line has more than 4 elements")
            outf.write(line)

    outf.close()






def _to_pd_data(
        fname: str,
        ):
    r'''
    convert the raw .csv data to pandas dataframe and numpy array
    input .csv file format should be: timestamp, node u, node v, weight w, 
    Args:
        fname: the path to the raw data
    '''
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    w_list = []

    with open(fname) as f:
        s = next(f)
        node_ids = {}
        unique_id = 0
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            if (len(e) > 4):
                print (e)
            #convert to integer node id
            ts = float(e[0])

            if (e[1] not in node_ids):
                node_ids[e[1]] = unique_id
                unique_id += 1
            if (e[2] not in node_ids):
                node_ids[e[2]] = unique_id
                unique_id += 1
            
            u = node_ids[e[1]]
            i = node_ids[e[2]]
            w = float(e[3])

            label = 0
            feat = np.zeros((1))

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            feat_l.append(feat)
            w_list.append(w)
    
    return pd.DataFrame({'u': u_list,
                        'i': i_list,
                        'ts': ts_list,
                        'label': label_list,
                        'idx': idx_list,
                        'w':w_list}), np.array(feat_l)




def reindex(
        df: pd.DataFrame, 
        bipartite: Optional[bool] = False,
        ):
    r'''
    reindex the nodes especially if the node ids are not integers
    Args:
        df: the pandas dataframe containing the graph
        bipartite: whether the graph is bipartite
    '''
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df

# if __name__ == "__main__":
#     fname = "/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/un_trade/un_trade.csv"
#     outname = "/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/un_trade/un_trade_cleaned.csv"
#     clean_rows(fname, outname)