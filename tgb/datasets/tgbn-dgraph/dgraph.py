import dateutil.parser as dparser
import csv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from os import listdir
from datetime import datetime


"""
# Description of DGraphFin datafile.

#! File **dgraphfin.npz** including below keys:  

#* **x**: 17-dimensional node features.
#* **y**: node label.  
    There four classes. Below are the nodes counts of each class.     
    0: 1210092    
    1: 15509    
    2: 1620851    
    3: 854098    
    Nodes of Class 1 are fraud users and nodes of 0 are normal users, and they the two classes to be predicted.    
    Nodes of Class 2 and Class 3 are background users.    
    
#* **edge_index**: shape (4300999, 2).   
    Each edge is in the form (id_a, id_b), where ids are the indices in x.        

#* **edge_type**: 11 types of edges. 
    
#* **edge_timestamp**: the desensitized timestamp of each edge.
    
#* **train_mask, valid_mask, test_mask**:  
    Nodes of Class 0 and Class 1 are randomly splitted by 70/15/15.  
"""




def main():
    
    #* load the raw data from numpy
    with np.load('dgraphfin.npz') as data:
        
        x = data['x']
        print ("shape of the node feature vectors are")
        print (x.shape)
        
        y = data['y']
        print ("shape of the node labels are")
        print (y.shape)
        
        edge_index = data['edge_index']
        print ("shape of the edge index are")
        print (edge_index.shape)
        
        edge_type = data['edge_type']
        print ("shape of the edge type are")
        print (edge_type.shape)
        
        edge_timestamp = data['edge_timestamp']
        print ("shape of the edge timestamp are")
        print (edge_timestamp.shape)
        
        print ("check if the timestamps are sorted")
        print(np.all(edge_timestamp[:-1] <= edge_timestamp[1:]))

                
    
    


if __name__ == "__main__":
    main()
