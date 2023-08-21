import csv

"""
#! analyze statistics from the dataset
#* 1). # of unique nodes, 2). # of edges. 3). # of unique edges, 4). # of timestamps 5). min & max of edge weights, 6). recurrence of nodes
"""


def analyze_csv(fname):
    node_dict = {}
    edge_dict = {}
    num_edges = 0
    num_time = 0
    prev_t = "none"
    min_w = 100000
    max_w = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                # t,u,v,w
                t = row[0]
                u = row[1]
                v = row[2]
                w = float(row[3].strip())

                # min & max edge weights
                if w > max_w:
                    max_w = w

                if w < min_w:
                    min_w = w

                # count unique time
                if t != prev_t:
                    num_time += 1
                    prev_t = t

                # unique nodes
                if u not in node_dict:
                    node_dict[u] = 1
                else:
                    node_dict[u] += 1

                if v not in node_dict:
                    node_dict[v] = 1
                else:
                    node_dict[v] += 1

                # unique edges
                num_edges += 1
                if (u, v) not in edge_dict:
                    edge_dict[(u, v)] = 1
                else:
                    edge_dict[(u, v)] += 1

    print("----------------------high level statistics-------------------------")
    print("number of total edges are ", num_edges)
    print("number of nodes are ", len(node_dict))
    print("number of unique edges are ", len(edge_dict))
    print("number of unique timestamps are ", num_time)
    print("maximum edge weight is ", max_w)
    print("minimum edge weight is ", min_w)

    num_10 = 0
    num_100 = 0
    num_1000 = 0

    for node in node_dict:
        if node_dict[node] >= 10:
            num_10 += 1
        if node_dict[node] >= 100:
            num_100 += 1
        if node_dict[node] >= 1000:
            num_1000 += 1
    print("number of nodes with # edges >= 10 is ", num_10)
    print("number of nodes with # edges >= 100 is ", num_100)
    print("number of nodes with # edges >= 1000 is ", num_1000)
    print("----------------------high level statistics-------------------------")


"""
return a node dict only keeping nodes with > 10 edges
"""


def extract_node_dict(fname, freq=10):
    node_dict = {}
    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                # t,u,v,w
                t = row[0]
                u = row[1]
                v = row[2]
                w = float(row[3].strip())
                if u not in node_dict:
                    node_dict[u] = 1
                else:
                    node_dict[u] += 1

                if v not in node_dict:
                    node_dict[v] = 1
                else:
                    node_dict[v] += 1

    out_dict = {}
    for node in node_dict:
        if node_dict[node] >= freq:
            out_dict[node] = node_dict[node]
    return out_dict


"""
remove any edges do not contain either src or dst not in the node dict
"""


def clean_edgelist(fname, outname, node_dict):
    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["time", "src", "dst", "weight"]
        write.writerow(fields)
        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    # t,u,v,w
                    t = row[0]
                    u = row[1]
                    v = row[2]
                    w = float(row[3].strip())
                    if u in node_dict and v in node_dict:
                        write.writerow([t, u, v, w])



def sort_edgelist(in_file, outname):
    """
    sort the edges by timestamp
    """
    row_dict = {} #{day: {row: row}}
    line_idx = 0
    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["day", "src", "dst", "callsign", "typecode"]
        write.writerow(fields)
        with open(in_file, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if line_idx == 0:  # header
                    line_idx += 1
                    continue
                ts = int(row[0])
                if ts not in row_dict:
                    row_dict[ts] = {}
                    row_dict[ts][line_idx] = row
                else:
                    row_dict[ts][line_idx] = row
                line_idx += 1
        
        for ts in sorted(row_dict.keys()):
            for idx in row_dict[ts].keys():
                row = row_dict[ts][idx]
                write.writerow(row)




def main():
    """
    keeping subgraph of most active nodes
    """
    # freq = 10
    # fname = "stablecoin_edgelist.csv"
    # node_dict = extract_node_dict(fname, freq=freq)

    # outname = "stablecoin_freq10.csv"
    # clean_edgelist(fname, outname, node_dict)

    # fname = "stablecoin_freq10.csv"
    # analyze_csv(fname)

    """
    sort edgelist by time
    """
    in_file = "tgbl-coin_edgelist.csv"
    outname = "tgbl-coin_edgelist_sorted.csv"
    sort_edgelist(in_file, outname)


if __name__ == "__main__":
    main()
