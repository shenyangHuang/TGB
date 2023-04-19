from typing import Optional, cast, Union, List, overload, Literal
from tqdm import tqdm
import numpy as np
import pandas as pd
import os.path as osp
import time
import csv
from datetime import datetime

#! these are helper functions
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



def sort_edgelist(fname, 
                  outname = 'sorted_lastfm_edgelist.csv'):
    r"""
    sort the edgelist by time
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()
    
    with open(outname, 'w') as outf:
        write = csv.writer(outf)
        fields = ["time", "user_id", "genre", "weight"]              
        write.writerow(fields)
        
        rows_dict = {}
        for idx in range(1,len(lines)):
            vals = lines[idx].split(',')
            user_id = vals[0]
            time_ts = vals[1][:-7]
            genre = vals[2]
            w = float(vals[3].strip())
            if (time_ts not in rows_dict):
                rows_dict[time_ts] = [(user_id, genre, w)]
            else:
                rows_dict[time_ts].append((user_id, genre, w))
        
        time_keys = list(rows_dict.keys())
        time_keys.sort()
        
        for ts in time_keys:
            rows = rows_dict[ts]
            for user_id, genre, w in rows:
                write.writerow([ts, user_id, genre, w])
                



def sort_node_labels(fname,
                     outname):
    r"""
    sort the node labels by time
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()
    
    with open(outname, 'w') as outf:
        write = csv.writer(outf)
        fields = ["time", 'user_id', 'genre', 'weight']         
        write.writerow(fields)
        rows_dict = {}
        
        for i in range(1,len(lines)):
            vals = lines[i].split(',')
            user_id = vals[0]
            year = int(vals[1])
            month = int(vals[2])
            day = int(vals[3])
            genre = vals[4]
            w = float(vals[5])
            date_cur = datetime(year,month,day)
            time_ts = date_cur.strftime("%Y-%m-%d")
            if (time_ts not in rows_dict):
                rows_dict[time_ts] = [(user_id, genre, w)]
            else:
                rows_dict[time_ts].append((user_id, genre, w))
                
        time_keys = list(rows_dict.keys())
        time_keys.sort()
        
        for ts in time_keys:
            rows = rows_dict[ts]
            for user_id, genre, w in rows:
                write.writerow([ts, user_id, genre, w])
            
    
    
        
    
    
#! data loading functions
def load_node_labels(fname,
                     genre_index,
                     user_index):
    r"""
    load node labels as weight distribution
    time, user_id, genre, weight
    assume node labels are already sorted by time
    convert all time unit to unix time
    genre_index: a dictionary mapping genre to index
    """
    if not osp.exists(fname):
        raise FileNotFoundError(f"File not found at {fname}")
    #lastfmgenre dataset
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()
    
    TIME_FORMAT = "%Y-%m-%d"

    # day, user_idx, label_vec
    label_size = len(genre_index)
    label_vec = np.zeros(label_size)
    date_prev = 0
    prev_user = 0
    
    ts_list = []
    node_id_list = []
    y_list = []
    

    #user_id,year,month,day,genre,weight
    for i in tqdm(range(1,len(lines))):
        vals = lines[i].split(',')
        user_id = user_index[vals[1]]
        ts = vals[0]
        genre = vals[2]
        weight = float(vals[3])
        date_cur = datetime.strptime(ts, TIME_FORMAT)
        if (i == 1):
            date_prev = date_cur
            prev_user = user_id
        #the next day
        if (date_cur != date_prev):
            ts_list.append(date_prev.timestamp())
            node_id_list.append(prev_user)
            y_list.append(label_vec)
            label_vec = np.zeros(label_size)
            date_prev = date_cur
            prev_user = user_id
        else:
            label_vec[genre_index[genre]] = weight
            
        if (user_id != prev_user):
            ts_list.append(date_prev.timestamp())
            node_id_list.append(prev_user)
            y_list.append(label_vec)
            prev_user = user_id
            label_vec = np.zeros(label_size)
    return pd.DataFrame({'ts': ts_list,
                        'node_id': node_id_list,
                        'y': y_list})


def load_edgelist(fname, genre_index):
    """
    load the edgelist into a pandas dataframe
    assume all edges are already sorted by time
    convert all time unit to unix time
    
    time, user_id, genre, weight
    """
    TIME_FORMAT = "%Y-%m-%d %H:%M:%S"  #2005-02-14 00:00:3
    #lastfmgenre dataset
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    user_index = {} #map user id to index
    unique_id = max(genre_index.values()) + 1
    u_list = []
    i_list = []
    ts_list = []
    label_list = []
    idx_list = []
    feat_l = []
    w_list = []
    for idx in tqdm(range(1,len(lines))):
        vals = lines[idx].split(',')
        time_ts = vals[0]
        user_id = vals[1]
        genre = vals[2]
        w = float(vals[3].strip())
        date_object = datetime.strptime(time_ts, TIME_FORMAT)
        if (user_id not in user_index):
            user_index[user_id] = unique_id
            unique_id += 1

        u = user_index[user_id]
        i = genre_index[genre]
        label = 0
        feat = np.zeros((1))
        u_list.append(u)
        i_list.append(i)
        ts_list.append(date_object.timestamp())
        label_list.append(label)
        idx_list.append(idx)
        feat_l.append(feat)
        w_list.append(w)

    return pd.DataFrame({'u': u_list,
                        'i': i_list,
                        'ts': ts_list,
                        'label': label_list,
                        'idx': idx_list,
                        'w':w_list}), user_index




    



def load_genre_list(fname):
    """
    load the list of genres 
    """
    if not osp.exists(fname):
        raise FileNotFoundError(f"File not found at {fname}")

    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    genre_index = {}
    ctr = 0
    for i in range(1,len(lines)):
        vals = lines[i].split(',')
        genre = vals[0]
        if (genre not in genre_index):
            genre_index[genre] = ctr
            ctr += 1
        else:
            raise ValueError("duplicate in genre_index")
    return genre_index


def csv_to_pd_data(
        fname: str,
        ) -> pd.DataFrame:
    r'''
    currently used by open sky dataset
    convert the raw .csv data to pandas dataframe and numpy array
    input .csv file format should be: timestamp, node u, node v, attributes
    Args:
        fname: the path to the raw data
    '''
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    w_list = []
    TIME_FORMAT = "%Y-%m-%d" #2019-01-01
    node_ids = {}
    unique_id = 0

    with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            idx = 0
            #'day','src','dst','callsign','typecode'
            for row in csv_reader:
                if (idx == 0):
                    idx += 1
                    continue
                else:
                    ts = row[0]
                    date_cur = datetime.strptime(ts, TIME_FORMAT)
                    ts = float(date_cur.timestamp())
                    src = row[1]
                    dst = row[2]
                    if (src not in node_ids):
                        node_ids[src] = unique_id
                        unique_id += 1
                    if (dst not in node_ids):
                        node_ids[dst] = unique_id
                        unique_id += 1
                    u = node_ids[src]
                    i = node_ids[dst]
                    w = float(1)

                    #! padding 
                    label = 0
                    feat = np.zeros((1))
                    u_list.append(u)
                    i_list.append(i)
                    ts_list.append(ts)
                    label_list.append(label)
                    idx_list.append(idx)
                    feat_l.append(feat)
                    w_list.append(w)
                    idx += 1
    return pd.DataFrame({'u': u_list,
                        'i': i_list,
                        'ts': ts_list,
                        'label': label_list,
                        'idx': idx_list,
                        'w':w_list}), np.array(feat_l)

    







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

if __name__ == "__main__":
    # """
    # clean rows for un trade dataset
    # """
    # fname = "/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/un_trade/un_trade.csv"
    # outname = "/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/un_trade/un_trade_cleaned.csv"
    # clean_rows(fname, outname)
    
    # """
    # sort edgelist by time for lastfm dataset
    # """
    # fname = "../datasets/lastfmGenre/lastfm_edgelist_clean.csv"
    # outname = '../datasets/lastfmGenre/sorted_lastfm_edgelist.csv'
    # sort_edgelist(fname, 
    #               outname = outname)
    
    """
    sort node labels by time for lastfm dataset
    """
    fname = "../datasets/lastfmGenre/7days_labels.csv"
    outname = '../datasets/lastfmGenre/sorted_7days_node_labels.csv'
    sort_node_labels(fname,
                     outname)
    
    
    