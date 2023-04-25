import csv

'''
#! analyze statistics from the dataset
#* 1). # of unique nodes, 2). # of edges. 3). # of unique edges, 4). # of timestamps 5). min & max of edge weights, 6). recurrence of nodes
'''
def analyze_csv(fname):
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    node_dict = {}
    edge_dict = {}
    num_edges = 0
    num_time = 0
    prev_t = "none"
    min_w = 100000
    max_w = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                #t,u,v,w
                t = row[0]
                u = row[1]
                v = row[2]
                w = float(row[3].strip())

                # min & max edge weights
                if (w > max_w):
                    max_w = w

                if (w < min_w):
                    min_w = w

                # count unique time
                if (t != prev_t):
                    num_time += 1
                    prev_t = t

                #unique nodes
                if (u not in node_dict):
                    node_dict[u] = 1
                else:
                    node_dict[u] += 1
                
                if (v not in node_dict):
                    node_dict[v] = 1
                else:
                    node_dict[v] += 1

                #unique edges
                num_edges += 1
                if ((u,v) not in edge_dict):
                    edge_dict[(u,v)] = 1
                else:
                    edge_dict[(u,v)] += 1
        
    print ("----------------------high level statistics-------------------------")
    print ("number of total edges are ", num_edges)
    print ("number of nodes are ", len(node_dict))
    print ("number of unique edges are ", len(edge_dict))
    print ("number of unique timestamps are ", num_time)
    print ("maximum edge weight is ", max_w)
    print ("minimum edge weight is ", min_w)

    num_10 = 0
    num_100 = 0
    num_1000 = 0

    for node in node_dict:
        if (node_dict[node] >= 10):
            num_10 += 1
        if (node_dict[node] >= 100):
            num_100 += 1
        if (node_dict[node] >= 1000):
            num_1000 += 1
    print ("number of nodes with # edges >= 10 is ", num_10)
    print ("number of nodes with # edges >= 100 is ", num_100)
    print ("number of nodes with # edges >= 1000 is ", num_1000)
    print ("----------------------high level statistics-------------------------")





def main():
    analyze_csv("coin_edgelistv3.csv")
    


if __name__ == "__main__":
    main()

    



