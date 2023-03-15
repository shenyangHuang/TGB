import networkx as nx



def load_daily_fav_genres(user_edge_dicts):
    print ("hi")




def generate_node_labels(fname, 
                         fur_k_days=1):
    r"""
    read a temporal edgelist 
    node label = fav genre in the next k days
    generate the node label for each day, summing over the next k days
    
    1. separate the graph into daily snapshots 
    2. for each day, sum over genre weights for the user
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()


    #TODO continue writing here










'''
treat each year as a timestamp 
'''
def load_UNvote_temporarl_edgelist(fname):
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()
    #assume it is a directed graph at each timestamp
    # G = nx.DiGraph()

    #date u  v  w
    #find how many timestamps there are
    max_time = 0
    current_date = ''
    #create one graph for each day
    G_times = []
    G = nx.DiGraph()

    for i in range(0, len(lines)):
        line = lines[i]
        values = line.split(',')
        t = values[0]
        v = values[1]       
        u = values[2]
        w = int(values[3])  #edge weight by number of shared publications in a year
        if current_date != '':
            if t != current_date:
                G_times.append(G)   #append old graph
                G = nx.DiGraph()    #create new graph
                current_date = t
        else:
            current_date = t
        G.add_edge(u, v, weight=w)
    #don't forget to add last one!!!!
    G_times.append(G)

    print ("maximum time stamp is " + str(len(G_times)))
    return G_times
