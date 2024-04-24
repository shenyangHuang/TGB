import csv
import datetime
import glob, os


def load_csv_raw(fname):
    r"""
    load from the raw data and retrieve, timestamp, head, tail, relation, time_rel
    convert all dates into unix timestamps
    """
    out_dict = {}
    first_row = True
    num_lines = 0
    #? head,tail,relation_type
    #* Q31,Q1088364,P1344
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter =',')

        for row in reader: 
            if first_row:
                first_row = False
                continue
            head = row[0]
            tail = row[1]
            relation_type = row[2]

            if ( "None" in head or "None" in tail or "None" in relation_type):
                continue
            else:
                if ((head, tail, relation_type) in out_dict):
                    out_dict[(head, tail, relation_type)] += 1
                else:
                    out_dict[(head, tail, relation_type)] = 1 
    return out_dict, num_lines


def write2csv(outname: str, 
              out_dict: dict,):
    r"""
    Write the dictionary to a csv file
    Parameters:
        outname: str: name of the output file
        out_dict: dictionary to be written
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['head', 'tail', 'relation_type'])
        for edge in out_dict:
            head = edge[0]
            tail = edge[1]
            relation_type = edge[2]
            row = [head, tail, relation_type]
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




def main():

    #! when timestamps overlap can't update dictionary

    total_lines = 0
    total_edge_dict = {}

    #1. find all files with .txt in the folder
    total_edge_file = "tkgl-wikidata_static_edgelist.csv"
    for file in glob.glob("*.csv"):
        print (file)
        edge_dict, num_lines = load_csv_raw(file)
        print ("processed ", num_lines, " lines")
        total_lines += num_lines
        update_dict(total_edge_dict, edge_dict)
    print ("processed a total of ", total_lines, " lines")
    write2csv(total_edge_file, total_edge_dict)



if __name__ == "__main__":
    main()