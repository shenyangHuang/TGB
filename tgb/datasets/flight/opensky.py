import dateutil.parser as dparser
import csv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from os import listdir


def find_csv_filenames(path_to_dir, 
                       suffix=".csv" ):
    r"""
    find all csv files in a directory
    Parameters:
    	path_to_dir (str): path to the directory
		suffix (str): suffix of the file
    """
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]
	




def flight2edgelist(fname, 
                outname,
                node_dict = None,):
    '''
    process all rows into 
    Day, src, dst, callsign, number, icao24, registration, typecode
    and save it as an edgelist file
    '''
    miss_node_lines = 0

    skip_lines = 0
    print ("processing ", outname)
    with open(outname, 'w') as outf:
        write = csv.writer(outf)
        fields = ['day','src','dst','callsign','number','icao24','registration','typecode']
        write.writerow(fields)

        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            #callsign,number,icao24,registration,typecode,origin,destination,firstseen,lastseen,day,latitude_1,longitude_1,altitude_1,latitude_2,longitude_2,altitude_2
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    out = []
                    callsign = row[0]
                    number = row[1]
                    icao24 = row[2]
                    registration = row[3]
                    typecode = row[4]
                    src = row[5]
                    if (src == ''):
                        skip_lines += 1
                        continue
                    dst = row[6]
                    if (dst == ''):
                        skip_lines += 1 
                        continue

                    if (node_dict is not None):
                        if (src not in node_dict):
                            miss_node_lines += 1
                            continue
                        if (dst not in node_dict):
                            miss_node_lines += 1
                            continue
                    day = row[9]
                    day = day[0:10]

                    out.append(day)
                    out.append(src)
                    out.append(dst)
                    out.append(callsign)
                    out.append(number)
                    out.append(icao24)
                    out.append(registration)
                    out.append(typecode)
                    write.writerow(out)
                    line_count += 1
        print(f'Processed {line_count} lines.')
        print(f'Skipped {skip_lines} lines.')
        print(f'missing node {miss_node_lines} lines.')
    return line_count, skip_lines, miss_node_lines


def load_icao_airports(fname="airport_codes.csv"):
    airports_continent = {}
    airports_country = {}

    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()
    #date u  v  w
    #find how many timestamps there are
    
    for i in range(0, len(lines)):
        line = lines[i]
        values = line.split(',')
        icao = values[0]
        continent = values[4]
        country = values[5]
        airports_continent[icao] = continent
        airports_country[icao] = country
    return airports_continent, airports_country

def main():

    """
    instructions for recompiling the dataset from 
    https://zenodo.org/record/7323875#.ZD1-43ZKguX

    1. download all datasets into a folder specified by in_dir (such as full_dataset)
    2. run the following code to extract the needed information 
    """
    
    _, airports_country = load_icao_airports(fname="airport_codes.csv")
    
    in_dir = "full_dataset/"
    out_dir = "edgelists/"

    csv_name = "flightlist_20190101_20190131.csv"

    csv_names = find_csv_filenames(in_dir)
    processed_lines = 0
    skipped_lines = 0
    miss_node_lines = 0

    for csv_name in tqdm(csv_names):
        fname = in_dir + csv_name
        outname = out_dir + csv_name[11:-4] + "edgelist"+".csv"
        line_count, skip_lines, miss_node = flight2edgelist(fname, outname, node_dict=airports_country)
        processed_lines += line_count
        skipped_lines += skip_lines
        miss_node_lines += miss_node
    print(f'Processed {processed_lines} lines.')
    print(f'Skipped {skipped_lines} lines.')
    print(f'missing node {miss_node_lines} lines.')



    


if __name__ == "__main__":
    main()
