import csv
import datetime

def load_csv_raw(fname):
    r"""
    load from the raw data and retrieve, timestamp, head, tail, relation 
    convert all dates into unix timestamps
    #!Event ID	Event Date	Event Type	Event Mode	Intensity	Quad Code	Contexts	Actor Name	Actor Country	Actor COW	Primary Actor Sector	Actor Sectors	Actor Title	Actor Name Raw	Wikipedia Actor ID	Recipient Name	Recipient Country	Recipient COW	Primary Recipient Sector	Recipient Sectors	Recipient Title	Recipient Name Raw	Wikipedia Recipient ID	Placename	City	District	Province	Country	Latitude	Longitude	GeoNames ID	Raw Placename	Feature Type	Source	Publication Date	Story People	Story Organizations	Story Locations	Language	Version
    """
    out_dict = {}
    first_row = True
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter ='\t')
        for row in reader: 
            if first_row:
                first_row = False
                continue
            date = row[1]
            relation_type = row[2]
            head = row[7]
            tail = row[15]
            
            if ("None" in date or "None" in head or "None" in tail or "None" in relation_type):
                continue
            else:
                 #! remove redundant edges with same timestamps
                TIME_FORMAT = "%Y-%m-%d" #2018-01-01
                date_cur = datetime.datetime.strptime(date, TIME_FORMAT)
                ts = int(date_cur.timestamp())
                if (ts in out_dict):
                    if (head, tail, relation_type) in out_dict[ts]:
                        out_dict[ts][(head, tail, relation_type)] += 1
                    else:
                        out_dict[ts][(head, tail, relation_type)] = 1
                else:
                    out_dict[ts] = {}
                    out_dict[ts][(head, tail, relation_type)] = 1
    return out_dict

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



def main():
    fname = "2018-Jan.txt"
    print ("hi")
    lines = load_csv_raw(fname)
    outname = "tkgl-polecat_edgelist.csv"
    write2csv(outname, lines)


if __name__ == "__main__":
    main()