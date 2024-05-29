import csv
import datetime
import glob, os

def load_csv_raw(fname):
    """
    load the raw csv file and merge them into one
    """
    out_dict = {}
    num_lines = 0
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter ='\t')
        #* /user/10746682	U_SO_C_IC	/issue_comment/455195715	1547754198
        for row in reader: 
            head = row[0]
            relation_type = row[1]
            tail = row[2]
            ts = int(row[3])
            if (ts in out_dict):
                if (head, tail, relation_type) in out_dict[ts]:
                    out_dict[ts][(head, tail, relation_type)] += 1
                else:
                    out_dict[ts][(head, tail, relation_type)] = 1
            else:
                out_dict[ts] = {}
                out_dict[ts][(head, tail, relation_type)] = 1
            num_lines += 1
    return out_dict, num_lines


def write2csv(outname, out_dict):
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


def load_edgelist(fname):
    """
    load the edgelist
    """
    node_dict = {} # {node_name: node_id}
    node_type_dict = {} # {node_id: node_type}
    rel_type_dict = {}
    edge_dict = {} # {edge: edge_type}
    node_type_mapping = {}
    num_edges = 0
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter =',')
        first_row = True
        for row in reader:
            if first_row:
                first_row = False
                continue
            ts = int(row[0])
            head = row[1]
            tail = row[2]
            relation_type = row[3]

            head_strs = head.split('/')
            tail_strs = tail.split('/')
            head_type = head_strs[1]
            tail_type = tail_strs[1]

            if head_type not in node_type_mapping:
                node_type_mapping[head_type] = len(node_type_mapping)
            if tail_type not in node_type_mapping:
                node_type_mapping[tail_type] = len(node_type_mapping)

            if head not in node_dict:
                node_dict[head] = len(node_dict)
                node_type_dict[node_dict[head]] = node_type_mapping[head_type]
            if tail not in node_dict:
                node_dict[tail] = len(node_dict)
                node_type_dict[node_dict[tail]] = node_type_mapping[tail_type]
            if relation_type not in rel_type_dict:
                rel_type_dict[relation_type] = len(rel_type_dict)
            if ts not in edge_dict:
                edge_dict[ts] = {}
            edge_dict[ts][(node_dict[head], node_dict[tail], rel_type_dict[relation_type])] = 1
            num_edges += 1
    print ("there are {} nodes".format(len(node_dict)))
    print ("there are {} edges".format(num_edges))

    return node_dict, node_type_dict, edge_dict, rel_type_dict, node_type_mapping



def writeNodeType(node_type_dict, outname):
    r"""
    write the node type mapping to a file
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['node_id', 'type'])
        for key in node_type_dict:
            writer.writerow([key, node_type_dict[key]])


def writeEdgeTypeMapping(edge_type_dict, outname):
    r"""
    write the edge type mapping to a file
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['edge_id', 'type'])
        for key in edge_type_dict:
            writer.writerow([key, edge_type_dict[key]])


def writeNodeTypeMapping(node_type_dict, outname):
    r"""
    write the edge type mapping to a file
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['node_type_id', 'type'])
        for key in node_type_dict:
            writer.writerow([key, node_type_dict[key]])


def write2edgelist(out_dict, outname):
    r"""
    Write the dictionary to a csv file
    """
    num_lines = 0
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['timestamp', 'head', 'tail', 'relation_type'])
        dates = list(out_dict.keys())
        dates.sort()
        for date in dates:
            for edge in out_dict[date]:
                head = edge[0]
                tail = edge[1]
                relation_type = int(edge[2])
                row = [date, head, tail, relation_type]
                writer.writerow(row)
                num_lines += 1
    print ("there are {} lines in the file".format(num_lines))



def main():
    # """
    # concatenate edgelists
    # """
    # total_lines = 0
    # total_edge_dict = {} 
    # #1. find all files with .txt in the folder
    # for file in glob.glob("*.txt"):
    #     # outname = file[7:11] + "_edgelist.csv"
    #     print ("processing", file)
    #     edge_dict, num_lines = load_csv_raw(file)
    #     total_lines += num_lines
    #     print ("-----------------------------------")
    #     print ("file, ", file)
    #     print ("number of lines, ", num_lines)
    #     print ("number of ts, ", len(edge_dict))
    #     print ("-----------------------------------")
    #     total_edge_dict.update(edge_dict)
    # outname = "all_edgelist.csv"
    # write2csv(outname, total_edge_dict)


    fname ="github_03_2024_subset.csv"#"github_01_2024_subset.csv" #"thgl-github_edges.csv" #"all_edgelist.csv" 
    node_dict, node_type_dict, edge_dict, edge_type_dict, node_type_mapping = load_edgelist(fname)
    write2edgelist (edge_dict, "thgl-github_edgelist.csv")
    writeNodeType(node_type_dict, "thgl-github_nodetype.csv")
    writeEdgeTypeMapping(edge_type_dict, "thgl-github_edgemapping.csv")
    writeNodeTypeMapping(node_type_mapping, "thgl-github_nodemapping.csv")
    


if __name__ == "__main__":
    main()