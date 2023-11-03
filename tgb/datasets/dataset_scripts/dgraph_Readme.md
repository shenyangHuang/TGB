# Description of DGraphFin datafile.

File **dgraphfin.npz** including below keys:  

- **x**: 17-dimensional node features.
- **y**: node label.  
    There four classes. Below are the nodes counts of each class.     
    0: 1210092    
    1: 15509    
    2: 1620851    
    3: 854098    
    Nodes of Class 1 are fraud users and nodes of 0 are normal users, and they the two classes to be predicted.    
    Nodes of Class 2 and Class 3 are background users.    
    
- **edge_index**: shape (4300999, 2).   
    Each edge is in the form (id_a, id_b), where ids are the indices in x.        

- **edge_type**: 11 types of edges. 
    
- **edge_timestamp**: the desensitized timestamp of each edge.
    
- **train_mask, valid_mask, test_mask**:  
    Nodes of Class 0 and Class 1 are randomly splitted by 70/15/15.  

    


    