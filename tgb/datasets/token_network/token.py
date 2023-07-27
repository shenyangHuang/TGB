import csv




def store_token_address(token_dict, outname, topk=1000):
    """
    Parameters:
        outname: name of the output csv file
    Output:
        output csv file with topk token addresses
    """
    sorted_tokens = {k: v for k, v in sorted(token_dict.items(), key=lambda item: item[1], reverse=True)}
    ctr = 0
    with open(outname, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(["token_address", "frequency"])
        for key, value in sorted_tokens.items():
            if (ctr <= topk):
                csv_writer.writerow([key, value])
            else:
                break
            ctr += 1

def analyze_token_frequency(fname):
    # ['token_address', 'from_address', 'to_address', 'value', 'block_timestamp']
    token_dict = {}
    node_dict = {}
    time_dict = {}
    max_w = 0
    min_w = 100000

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        ctr = 0
        for row in csv_reader:
            if ctr == 0:
                ctr += 1
                continue
            else:
                token_type = row[0]
                if (token_type not in token_dict):
                    token_dict[token_type] = 1
                else:
                    token_dict[token_type] += 1
                src = row[1]
                if (src not in node_dict):
                    node_dict[src] = 1
                else:
                    node_dict[src] += 1
                dst = row[2]
                if (dst not in node_dict):
                    node_dict[dst] = 1
                else:
                    node_dict[dst] += 1

                w = float(row[3])
                if (w > max_w):
                    max_w = w
                elif (w < min_w):
                    min_w = w
                timestamp = row[4]
                if (timestamp not in time_dict):
                    time_dict[timestamp] = 1
                ctr += 1
    
    print (" number of unique tokens are ", len(token_dict))
    print (" number of unique nodes are ", len(node_dict))
    print (" number of unique timestamps are ", len(time_dict))
    print (" max weight is ", max_w)
    print (" min weight is ", min_w)

    topk = 1000
    store_token_address(token_dict, "token_list.csv", topk=topk)

    




def print_csv(fname):
    # ['token_address', 'from_address', 'to_address', 'value', 'block_timestamp']
    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        ctr = 0
        for row in csv_reader:
            ctr += 1
        print ("there are ", ctr, " rows in the csv file")





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





def main():
    fname = "ERC20_token_network.csv"
    analyze_token_frequency(fname)
    #print_csv(fname)
    #analyze_csv(fname)


if __name__ == "__main__":
    main()