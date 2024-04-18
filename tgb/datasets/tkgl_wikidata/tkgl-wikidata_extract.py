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
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter =',')

        for row in reader: 
            if first_row:
                first_row = False
                continue
            date = row[0][1:10]
            head = row[1]
            tail = row[2]
            relation_type = row[3]
            time_rel = row[4]

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
        writer.writerow(['date', 'head', 'tail', 'relation_type','time_rel_type'])
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



def main():

    total_lines = 0
    total_edge_dict = {}

    #1. find all files with .txt in the folder
    total_edge_file = "tkgl-wikidata_edgelist.csv"
    for file in glob.glob("*.csv"):
        print (file)
        edge_dict, num_lines = load_time_csv_raw(file)
        total_lines += num_lines
        total_edge_dict.update(edge_dict)
    edge_type_dict, node_dict = write2csv(total_edge_file, total_edge_dict)
