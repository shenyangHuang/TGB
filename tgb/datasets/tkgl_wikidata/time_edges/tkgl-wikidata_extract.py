import csv
import datetime
import glob, os


def load_time_csv_raw(fname):
    r"""
    load from the raw data and retrieve, timestamp, head, tail, relation, time_rel
    convert all dates into unix timestamps
    """
    out_dict = {}
    first_row = True
    num_lines = 0
    #? timestamp,head,tail,relation_type,time_rel_type
    #* +1999-01-01T00:00:00Z,Q31,Q4916,P38,P580
    error_ctr = 0
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter =',')

        for row in reader: 
            if first_row:
                first_row = False
                continue
            date = row[0][0:11]
            head = row[1]
            tail = row[2]
            relation_type = row[3]
            time_rel = row[4]

            if (len(date) == 0):
                continue
            
            if ("None" in date or "None" in head or "None" in tail or "None" in relation_type):
                continue
            else:
                TIME_FORMAT = "%Y"
                #* only keep track of year in positive BC
                if (date[0] == "+"):
                    ts = int(date[1:5])
                else:
                    continue

                #* no scifi for knowledge graphs 
                if (ts > 2024):
                    continue

                num_lines += 1
                if (ts in out_dict):
                    if (head, tail, relation_type, time_rel) in out_dict[ts]:
                        out_dict[ts][(head, tail, relation_type, time_rel)] += 1
                    else:
                        out_dict[ts][(head, tail, relation_type, time_rel)] = 1
                else:
                    out_dict[ts] = {}
                    out_dict[ts][(head, tail, relation_type, time_rel)] = 1
    return out_dict, num_lines


def write2csv(outname: str, 
              out_dict: dict,):
    r"""
    Write the dictionary to a csv file
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['ts', 'head', 'tail', 'relation_type','time_rel_type'])
        dates = list(out_dict.keys())
        dates.sort()
        for date in dates:
            for edge in out_dict[date]:
                head = edge[0]
                tail = edge[1]
                relation_type = edge[2]
                time_rel = edge[3]
                row = [date, head, tail, relation_type, time_rel]
                writer.writerow(row)


def update_dict(total_dict, new_dict):
    r"""
    Update the total_dict with new_dict
    """
    for key in new_dict:
        if key in total_dict:
            for edge in new_dict[key]:
                if edge in total_dict[key]:
                    total_dict[key][edge] += new_dict[key][edge]
                else:
                    total_dict[key][edge] = new_dict[key][edge]
        else:
            total_dict[key] = new_dict[key]
    return total_dict


def retrieve_all_entities(total_dict):
    r"""
    retrieve the entities from all edges of the total dictionary

    Parameters:
        total_dict: dictionary of all edges, {ts: {edge: count}}
    """
    node_dict = {}
    for key in total_dict:
        for edge in total_dict[key]:
            head = edge[0]
            tail = edge[1]
            if head not in node_dict:
                node_dict[head] = 1
            else:
                node_dict[head] += 1
            if tail not in node_dict:
                node_dict[tail] = 1
            else:
                node_dict[tail] += 1
    return node_dict


def writenode2csv(outname: str, 
              out_dict: dict,):
    r"""
    Write the dictionary to a csv file
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['entity', 'occurrences'])
        for node in out_dict:
            row = [node, out_dict[node]]
            writer.writerow(row) 


def main():

    #! when timestamps overlap can't update dictionary

    total_lines = 0
    total_edge_dict = {}

    #1. find all files with .txt in the folder
    total_edge_file = "tkgl-wikidata_edgelist.csv"
    for file in glob.glob("*.csv"):
        print (file)
        edge_dict, num_lines = load_time_csv_raw(file)
        print ("processed ", num_lines, " lines")
        total_lines += num_lines
        update_dict(total_edge_dict, edge_dict)
    print ("processed a total of ", total_lines, " lines")
    node_dict = retrieve_all_entities(total_edge_dict)
    writenode2csv("wiki_entities.csv", node_dict)
    write2csv(total_edge_file, total_edge_dict)



if __name__ == "__main__":
    main()