import csv
import datetime
import glob, os




def load_csv_raw(fname):
    r"""
    load from the raw data and retrieve, timestamp, head, tail, relation 
    convert all dates into unix timestamps
    #!Event ID	Event Date	Event Type	Event Mode	Intensity	Quad Code	Contexts	Actor Name	Actor Country	Actor COW	Primary Actor Sector	Actor Sectors	Actor Title	Actor Name Raw	Wikipedia Actor ID	Recipient Name	Recipient Country	Recipient COW	Primary Recipient Sector	Recipient Sectors	Recipient Title	Recipient Name Raw	Wikipedia Recipient ID	Placename	City	District	Province	Country	Latitude	Longitude	GeoNames ID	Raw Placename	Feature Type	Source	Publication Date	Story People	Story Organizations	Story Locations	Language	Version
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
            date = row[1]
            relation_type = row[2]
            head = row[7]
            tail = row[15]

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


#! fill in node and edge type dictionaries
def write2csv(outname: str, 
              out_dict: dict,
              edge_type_dict: dict = None,
              node_dict: dict = None,):
    r"""
    Write the dictionary to a csv file
    also keep track of edge_type or node_dict, update the provided one too
    """
    if (edge_type_dict is None):
        edge_type_dict = {}
    if (node_dict is None):
        node_dict = {}

    max_edge_type_id = len(edge_type_dict)
    max_node_id = len(node_dict)    
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['date', 'head', 'tail', 'relation_type'])

        dates = list(out_dict.keys())
        dates.sort()
        for date in dates:
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
    
    return edge_type_dict, node_dict


# def write2csv(outname, out_dict):

#     node_dict = {}
#     max_node_id = 0
#     edge_type_dict = {}
#     max_edge_type_id = 0

#     with open(outname, 'w') as f:
#         writer = csv.writer(f, delimiter =',')
#         writer.writerow(['date', 'head', 'tail', 'relation_type'])

#         for date in out_dict:
#             for edge in out_dict[date]:
#                 head = edge[0]
#                 tail = edge[1]
#                 relation_type = edge[2]
#                 if head not in node_dict:
#                     node_dict[head] = max_node_id
#                     max_node_id += 1
#                 if tail not in node_dict:
#                     node_dict[tail] = max_node_id
#                     max_node_id += 1
#                 if relation_type not in edge_type_dict:
#                     edge_type_dict[relation_type] = max_edge_type_id
#                     max_edge_type_id += 1
#                 row = [date, node_dict[head], node_dict[tail], edge_type_dict[relation_type]]
#                 writer.writerow(row)


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

    #example
    # fname = "2018-Jan.txt"
    # print ("hi")
    # lines = load_csv_raw(fname)
    # outname = "tkgl-polecat_edgelist.csv"
    # write2csv(outname, lines)

    total_lines = 0
    num_days = 0
    total_edge_dict = {}

    #1. find all files with .txt in the folder
    for file in glob.glob("*.csv"):
        outname = file[0:7] + "_edgelist.csv"
        print ("processing", file, "to", outname)
        edge_dict, num_lines = load_csv_raw(file)
        total_lines += num_lines
        num_days += len(edge_dict)
        total_edge_dict.update(edge_dict)
    edge_type_dict, node_dict = write2csv("tkgl-polecat_edgelist.csv", total_edge_dict)
    print ("-----------------------------------")
    print ("total number of lines", total_lines)
    print ("total number of days", num_days)
    print ("there are", len(edge_type_dict), "unique edge types")
    print ("there are", len(node_dict), "unique nodes")
    writeEdgeTypeMapping(edge_type_dict, "tkgl-polecat_edgemapping.csv")



if __name__ == "__main__":
    main()

    #* rename functions
    # renames = []
    # for file in glob.glob("*.txt"):
    #     outname = file[-12:-4] + "_edgelist.csv"
    #     file_rename = file[-12:-4] + "_raw.csv"
    #     if ("Jan" in outname):
    #         outname = outname.replace("Jan", "01")
    #         renames.append((file, file_rename.replace("Jan", "01")))
    #     elif ("Feb" in outname):
    #         outname = outname.replace("Feb", "02")
    #         renames.append((file, file_rename.replace("Feb", "02")))
    #     elif ("Mar" in outname):
    #         outname = outname.replace("Mar", "03")
    #         renames.append((file, file_rename.replace("Mar", "03")))
    #     elif ("Apr" in outname):
    #         outname = outname.replace("Apr", "04")
    #         renames.append((file, file_rename.replace("Apr", "04")))
    #     elif ("May" in outname):
    #         outname = outname.replace("May", "05")
    #         renames.append((file, file_rename.replace("May", "05")))
    #     elif ("Jun" in outname):
    #         outname = outname.replace("Jun", "06")
    #         renames.append((file, file_rename.replace("Jun", "06")))
    #     elif ("Jul" in outname):
    #         outname = outname.replace("Jul", "07")
    #         renames.append((file, file_rename.replace("Jul", "07")))
    #     elif ("Aug" in outname):
    #         outname = outname.replace("Aug", "08")
    #         renames.append((file, file_rename.replace("Aug", "08")))
    #     elif ("Sep" in outname):
    #         outname = outname.replace("Sep", "09")
    #         renames.append((file, file_rename.replace("Sep", "09")))
    #     elif ("Oct" in outname):
    #         outname = outname.replace("Oct", "10")
    #         renames.append((file, file_rename.replace("Oct", "10")))
    #     elif ("Nov" in outname):
    #         outname = outname.replace("Nov", "11")
    #         renames.append((file, file_rename.replace("Nov", "11")))
    #     elif ("Dec" in outname):
    #         outname = outname.replace("Dec", "12")
    #         renames.append((file, file_rename.replace("Dec", "12")))
    # for file, file_rename in renames:
    #     os.rename(file, file_rename)