import csv
import datetime
import glob, os


def load_time_csv(fname):
    r"""
    load from data and retrieve, ts,head,tail,relation_type,time_rel_type
    """
    out_dict = {} #only contain edges {ts: {(head, tail, rel_type):count}}
    start_end_dict = {} #{(head, tail, rel_type): {start:year, end:year}}

    first_row = True
    point_in_time_lines = 0
    start_end_lines = 0

    #? ts,head,tail,relation_type,time_rel_type
    #* 0,Q331755,Q1294765,P39,P580
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter =',')
        for row in reader: 
            if first_row:
                first_row = False
                continue
            ts = int(row[0])
            head = row[1]
            tail = row[2]
            relation_type = row[3]
            time_rel = row[4]
            if (time_rel in ['P585', 'P577', 'P574']):
                if (ts in out_dict):
                    if (head, tail, relation_type) in out_dict[ts]:
                        out_dict[ts][(head, tail, relation_type)] += 1
                    else:
                        out_dict[ts][(head, tail, relation_type)] = 1
                else:
                    out_dict[ts] = {}
                    out_dict[ts][(head, tail, relation_type)] = 1
                point_in_time_lines += 1
            else: # time_rel in ['P580', 'P582']
                if (head, tail, relation_type) in start_end_dict:
                    if (time_rel in ['P580']):
                        start_end_dict[(head, tail, relation_type)]['start'] = ts
                    elif (time_rel in ['P582']):
                        start_end_dict[(head, tail, relation_type)]['end'] = ts
                    else:
                        raise ValueError(f"Unknown time_rel: {time_rel}")
                else:
                    start_end_dict[(head, tail, relation_type)] = {}
                    if (time_rel in ['P580']):
                        start_end_dict[(head, tail, relation_type)]['start'] = ts
                    else:
                        start_end_dict[(head, tail, relation_type)]['end'] = ts
                start_end_lines += 1

    print ("-----------------------------------")
    print ("for this edgelist:")
    print (f"point_in_time_lines: {point_in_time_lines}")
    print (f"start_end_lines: {start_end_lines}")
    print ("-----------------------------------")
    
    

    repeated_lines = 0
    no_duration_lines = 0
    
    #* now, repeat edges from start_end_dict
    for edge in start_end_dict.keys():
        if 'start' not in start_end_dict[edge]:
            #start_end_dict[edge]['start'] = 0 #start at year 0
            #start_end_dict[edge]['start'] = start_end_dict[edge]['end']
            no_duration_lines += 1
            continue
        if 'end' not in start_end_dict[edge]:
            # start_end_dict[edge]['end'] = 2024 #end at year 2024
            start_end_dict[edge]['end'] = start_end_dict[edge]['start'] #end at year 2024
            no_duration_lines += 1
            continue
        for year in range(start_end_dict[edge]['start'], start_end_dict[edge]['end']+1):
            if year not in out_dict:
                out_dict[year] = {}
            out_dict[year][edge] = 1
            repeated_lines += 1

    print ("-----------------------------------")
    print ("for this edgelist:")
    print (f"point_in_time_lines: {point_in_time_lines}")
    print (f"start_end_lines: {start_end_lines} resulting in")
    print (f"repeated_lines: {repeated_lines}")
    print (f"no_duration_lines: {no_duration_lines}")
    print ("-----------------------------------")
    print ("total lines: ", point_in_time_lines + repeated_lines)
    num_lines = point_in_time_lines + repeated_lines
    return out_dict, num_lines



def write2csv(outname: str, 
              out_dict: dict,):
    r"""
    Write the dictionary to a csv file
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['ts', 'head', 'tail', 'relation_type'])
        dates = list(out_dict.keys())
        dates.sort()
        for date in dates:
            for edge in out_dict[date]:
                head = edge[0]
                tail = edge[1]
                relation_type = edge[2]
                row = [date, head, tail, relation_type]
                writer.writerow(row)


def extract_subset(fname, outname, start_year=2000):
    node_dict = {}
    first_row = True
    r"""
    ts,head,tail,relation_type
    0,Q331755,Q1294765,P39
    0,Q116233388,Q2566630,P2348
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['ts', 'head', 'tail', 'relation_type'])
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter =',')
            for row in reader: 
                if first_row:
                    first_row = False
                    continue
                ts = int(row[0])
                head = row[1]
                tail = row[2]
                relation_type = row[3]
                if (ts >= start_year):
                    if head not in node_dict:
                        node_dict[head] = 1
                    if tail not in node_dict:
                        node_dict[tail] = 1
                    row = [ts, head, tail, relation_type]
                    writer.writerow(row)
    return node_dict



def extract_static_subset(fname, outname, node_dict):
    r"""
    extract static edges based a given node dict
    """
    first_row = True
    r"""
    head,tail,relation_type
    Q31,Q1088364,P1344
    Q31,Q3247091,P1151
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['head', 'tail', 'relation_type'])
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter =',')
            for row in reader: 
                if first_row:
                    first_row = False
                    continue
                head = row[0]
                tail = row[1]
                relation_type = row[2]
                if (head in node_dict) or (tail in node_dict):
                    row = [head, tail, relation_type]
                    writer.writerow(row)

def main():

    # #* repeat the edges of start and end dates
    # """
    # P580: start time
    # P582: end time
    # P585: point in time
    # P577: publication date
    # P574: year of publication of scientific name for taxon

    # we need to:
    # 1. get all edges with P585, P577 and P574 
    # 2. find out which edges has both start and end time
    # 3. for those without start time, start at year 0, without end time, end at year 2024
    # """
    # fname = "tkgl-wikidata_edgelist_raw.csv"
    # out_dict, num_lines = load_time_csv(fname)

    # outname = "tkgl-wikidata_edgelist.csv"
    # write2csv(outname, out_dict)

    inputfile = "tkgl-wikidata_edgelist.csv"
    outname = "tkgl-smallpedia_edgelist.csv"
    start_year = 2015
    node_dict = extract_subset(inputfile, outname, start_year=start_year)

    inputfile = "tkgl-wikidata_static_edgelist.csv"
    outname = "tkgl-smallpedia_static_edgelist.csv"
    extract_static_subset(inputfile, outname, node_dict)





if __name__ == "__main__":
    main()