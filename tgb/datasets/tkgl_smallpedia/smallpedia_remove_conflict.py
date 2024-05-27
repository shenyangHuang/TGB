import csv

def load_static_edgelist(file_path):
    r"""
    Load the static edgelist from the file_path
    Args:
        file_path: str, The path to the file
    """
    static_dict = {}
    first_row = True
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter =',')
        for row in reader: 
            if first_row:
                first_row = False
                continue
            head = row[0]
            tail = row[1]
            relation_type = row[2]
            static_dict[(head, tail, relation_type)] = 1
    return static_dict


def load_temporal_edgelist(file_path):
    r"""
    Load the temporal edgelist from the file_path
    Args:
        file_path: str, The path to the file
    """
    temporal_dict = {}
    first_row = True
    with open(file_path, 'r') as f:
        """
        ts,head,tail,relation_type
        0,Q331755,Q1294765,P39
        """
        reader = csv.reader(f, delimiter =',')
        for row in reader: 
            if first_row:
                first_row = False
                continue
            ts = int(row[0])
            head = row[1]
            tail = row[2]
            relation_type = row[3]
            if ts not in temporal_dict:
                temporal_dict[ts] = {}
                temporal_dict[ts][(head, tail, relation_type)] = 1
            else:
                if (head, tail, relation_type) in temporal_dict[ts]:
                    temporal_dict[ts][(head, tail, relation_type)] += 1
                else:
                    temporal_dict[ts][(head, tail, relation_type)] = 1
    return temporal_dict


def remove_conflict(static_dict, temporal_dict):
    r"""
    Remove the conflict between the static and temporal edgelist
    Args:
        static_dict: dict, The static edgelist
        temporal_dict: dict, The temporal edgelist
    """
    num_conflicts = 0
    for ts in temporal_dict:
        for edge in temporal_dict[ts]:
            head = edge[0]
            tail = edge[1]
            relation_type = edge[2]
            if (head, tail, relation_type) in static_dict:
                num_conflicts += 1
                static_dict.pop((head, tail, relation_type))
    print("Removed {} conflicts".format(num_conflicts))
    return static_dict


def write2csv(outname: str, 
              out_dict: dict,):
    r"""
    Write the dictionary to a csv file
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        #head,tail,relation_type
        writer.writerow(['head', 'tail', 'relation_type'])
        for edge in out_dict:
            head = edge[0]
            tail = edge[1]
            relation_type = edge[2]
            row = [head, tail, relation_type]
            writer.writerow(row)




def main():
    #! remove conflict: remove all edges with the same head, tail, relation_type from the static edgelist
    static_file = "tkgl-smallpedia_static_edgelist.csv"
    temporal_file = "tkgl-smallpedia_edgelist.csv"
    static_dict = load_static_edgelist(static_file)
    print("constructed static dictionary")
    temporal_dict = load_temporal_edgelist(temporal_file)
    print("constructed temporal dictionary")
    static_dict = remove_conflict(static_dict, temporal_dict)
    out_name = "tkgl-smallpedia_static_edgelist_no_conflict.csv"
    write2csv(out_name, static_dict)




if __name__ == "__main__":
    main()