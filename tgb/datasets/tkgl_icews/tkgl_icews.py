import csv
import datetime
import glob, os


def load_csv_raw(fname):
    r"""
    load from the raw data and retrieve, timestamp, head, tail, relation 
    convert all dates into unix timestamps
    #! Event ID	Event Date	Source Name	Source Sectors	Source Country	Event Text	CAMEO Code	Intensity	Target Name	Target Sectors	Target Country	Story ID	Sentence Number	Publisher	City	District	Province	Country	Latitude	Longitude
    """
    out_dict = {}
    first_row = True
    num_lines = 0
    with open(fname, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f, delimiter ='\t')
        for row in reader: 
            if first_row:
                first_row = False
                continue
            date = row[1] #1995-01-01
            head = row[2]
            tail = row[8]
            relation_type = row[6] #CAMEO code  #! not always integer in 2017 for some reason there is 13y
            if (len(date) == 0):
                continue
            
            if ("None" in date or "None" in head or "None" in tail or "None" in relation_type):
                continue
            else:
                 #! remove redundant edges with same timestamps
                TIME_FORMAT = "%Y-%m-%d" #2018-01-01
                date_cur = datetime.datetime.strptime(date, TIME_FORMAT)
                ts = int(date_cur.timestamp())
                num_lines += 1
                if (ts in out_dict):
                    if (head, tail, relation_type) in out_dict[ts]:
                        out_dict[ts][(head, tail, relation_type)] += 1
                    else:
                        out_dict[ts][(head, tail, relation_type)] = 1
                else:
                    out_dict[ts] = {}
                    out_dict[ts][(head, tail, relation_type)] = 1
    return out_dict, num_lines


def write2csv(outname, out_dict):

    node_dict = {}
    max_node_id = 0
    edge_type_dict = {}
    max_edge_type_id = 0

    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['date', 'head', 'tail', 'relation_type'])

        for date in out_dict:
            for edge in out_dict[date]:
                head = edge[0]
                tail = edge[1]
                relation_type = edge[2]
                if head not in node_dict:
                    node_dict[head] = max_node_id
                    max_node_id += 1
                if tail not in node_dict:
                    node_dict[tail] = max_node_id
                    max_node_id += 1
                if relation_type not in edge_type_dict:
                    edge_type_dict[relation_type] = max_edge_type_id
                    max_edge_type_id += 1
                row = [date, node_dict[head], node_dict[tail], edge_type_dict[relation_type]]
                writer.writerow(row)
    return node_dict, edge_type_dict


def writeEdgeTypeMapping(edge_type_dict, outname):
    r"""
    write the edge type mapping to a file
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['edge_id', 'type'])
        for key in edge_type_dict:
            writer.writerow([key, edge_type_dict[key]])




def main():

    total_lines = 0
    total_edge_dict = {} 
    #1. find all files with .txt in the folder
    for file in glob.glob("*.tab"):
        # outname = file[7:11] + "_edgelist.csv"
        print ("processing", file)
        edge_dict, num_lines = load_csv_raw(file)
        total_lines += num_lines
        print ("-----------------------------------")
        print ("file, ", file)
        print ("number of lines, ", num_lines)
        print ("number of days, ", len(edge_dict))
        print ("-----------------------------------")
        total_edge_dict.update(edge_dict)
    outname = "tkgl-icews_edgelist_tiny.csv"
    print ("total number of lines", total_lines)
    print ("total number of days", len(total_edge_dict))    
    node_dict, edge_type_dict = write2csv(outname, total_edge_dict)
    writeEdgeTypeMapping(edge_type_dict, "tkgl-icews_edgemapping.csv")




if __name__ == "__main__":
    main()