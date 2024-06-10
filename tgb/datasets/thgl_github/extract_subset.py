import csv


def load_edgelist(file_path, freq_threshold=5):
    """
    ts, head, tail, relation_type
    1704085200,/user/34452971,/pr/1660752740,U_SO_C_P
    """
    first_row = True
    edge_dict = {}
    num_nodes = 0
    num_edges = 0
    num_rels = 0
    node_dict = {}
    edge_freq_dict = {}
    num_lines = 0


    #! identify node type with least amount of edges
    node_type_freq = {}

    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter =',')
        for row in reader: 
            if first_row:
                first_row = False
                continue
            ts = int(row[0])
            head = row[1]
            head_type = head.split("/")[1]
            if (head_type not in node_type_freq):
                node_type_freq[head_type] = 1
            else:
                node_type_freq[head_type] += 1
            tail = row[2]
            tail_type = tail.split("/")[1]
            if (tail_type not in node_type_freq):
                node_type_freq[tail_type] = 1
            else:
                node_type_freq[tail_type] += 1
            relation_type = row[3]
            if head not in node_dict:
                node_dict[head] = 1
                num_nodes += 1
            else:
                node_dict[head] += 1

            if tail not in node_dict:
                node_dict[tail] = 1
                num_nodes += 1
            else:
                node_dict[tail] += 1

            if relation_type not in edge_freq_dict:
                edge_freq_dict[relation_type] = 1
                num_rels += 1
            else:
                edge_freq_dict[relation_type] += 1
            num_lines += 1
    print ("there are ", num_lines, " edges")
    print ("there are ", num_nodes, " nodes")
    print ("there are ", num_rels, " relations")

    node_freq5 = 0
    node_freq10 = 0
    node_freq100 = 0
    node_freq1000 = 0
    low_freq_dict = {}
    for k, v in node_dict.items():
        if v <= freq_threshold:
            low_freq_dict[k] = 1
            node_freq5 += 1
        if v >= 10:
            node_freq10 += 1
        if v >= 100:
            node_freq100 += 1
        if v >= 1000:
            node_freq1000 += 1
    print ("there are ", node_freq5, " nodes with frequency <= ", freq_threshold, " (inclusive)")
    print ("there are ", node_freq10, " nodes with frequency >= 10")
    print ("there are ", node_freq100, " nodes with frequency >= 100")
    print ("there are ", node_freq1000, " nodes with frequency >= 1000")
    # return node_freq10_dict
    return low_freq_dict, node_type_freq



def subset_by_node(file_path, low_freq_dict):
    first_row = True
    edge_dict = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter =',')
        for row in reader: 
            if first_row:
                first_row = False
                continue
            ts = int(row[0])
            head = row[1]
            tail = row[2]
            relation_type = row[3]

            #! remove any edges that belongs any node with degree one
            if (head in low_freq_dict) or (tail in low_freq_dict):
                continue

            # if (head in node_dict) or (tail in node_dict):
            if ts not in edge_dict:
                edge_dict[ts] = {}
            if (head,tail,relation_type) not in edge_dict[ts]:
                edge_dict[ts][(head,tail,relation_type)] = 1
            else:
                edge_dict[ts][(head,tail,relation_type)] += 1
    return edge_dict


def subset_by_node_type(file_path, remove_node_type_dict, low_freq_dict=None):
    first_row = True
    edge_dict = {}
    node_dict = {}
    num_edges = 0
    if (low_freq_dict is not None):
        check_low_freq = True

    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter =',')
        for row in reader: 
            if first_row:
                first_row = False
                continue
            ts = int(row[0])
            head = row[1]
            tail = row[2]
            if (head in low_freq_dict) or (tail in low_freq_dict):
                continue

            head_type = head.split("/")[1]
            tail_type = tail.split("/")[1]
            relation_type = row[3]



            if (head_type in remove_node_type_dict) or (tail_type in remove_node_type_dict):
                continue

            if (head not in node_dict):
                node_dict[head] = 1
            if (tail not in node_dict):
                node_dict[tail] = 1
            num_edges += 1
            if ts not in edge_dict:
                edge_dict[ts] = {}
            if (head,tail,relation_type) not in edge_dict[ts]:
                edge_dict[ts][(head,tail,relation_type)] = 1
            else:
                edge_dict[ts][(head,tail,relation_type)] += 1

    print ("there are ", num_edges, " edges in the output file")
    print ("there are ", len(node_dict), " nodes in the output file")
    return edge_dict




def write2csv(outname, out_dict):
    num_edges = 0
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['ts', 'head', 'tail', 'relation_type'])
        ts_list = list(out_dict.keys())
        ts_list.sort()

        for ts in ts_list:
            for edge in out_dict[ts]:
                head = edge[0]
                tail = edge[1]
                relation_type = edge[2]
                row = [ts, head, tail, relation_type]
                writer.writerow(row)
                num_edges += 1
    print ("there are ", num_edges, " edges in the output file")





def combine_edgelist(file_paths, outname):
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['ts', 'head', 'tail', 'relation_type'])
        for file_path in file_paths:
            first_row = True
            with open(file_path, 'r') as f:
                reader = csv.reader(f, delimiter =',')
                for row in reader: 
                    if first_row:
                        first_row = False
                        continue
                    ts = int(row[0])
                    head = row[1]
                    tail = row[2]
                    relation_type = row[3]
                    writer.writerow([ts, head, tail, relation_type])




def main():
    file_path = "github_03_2024.csv"
    freq_threshold = 2
    low_freq_dict, node_type_dict = load_edgelist(file_path, freq_threshold=freq_threshold)

    remove_node_type_dict = {'issue_comment':1, 'pr_review_comment':1} #{'issue_comment':1, 'pr_review_comment':1, 'issue':1} 
    edge_dict = subset_by_node_type(file_path, remove_node_type_dict, low_freq_dict=low_freq_dict)
    # edge_dict = subset_by_node(file_path, low_freq_dict=low_freq_dict)
    outname = "github_03_2024_subset.csv"
    write2csv(outname, edge_dict)

    # file_paths = ["github_01_2024_subset.csv", "github_02_2024_subset.csv", "github_03_2024_subset.csv"]
    # outname = "thgl-github_edges.csv"
    # combine_edgelist(file_paths, outname)





    
if __name__ == "__main__":
    main()