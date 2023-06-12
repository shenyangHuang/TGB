import networkx as nx
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import glob


"""
process all rows with origin, destination and day
and save it as an edgelist file
ignore lines without either destination or origin
"""


def flight2edgelist(fname, outname="None"):
    # fout = open(outname, "w")
    edge_ctr = 0
    time_ctr = 0
    prev_time = 0

    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                origin = row[5]
                destination = row[6]
                day = row[9]
                day = day[0:10]
                lastseen = row[8]
                line_count += 1
                if origin == "" or destination == "":
                    continue
                else:
                    edge_ctr += 1
                    if lastseen != prev_time:
                        time_ctr += 1
                    # fout.write(str(day) + "," + str(origin) + "," + str(destination) + "\n")
        print(f"Processed {line_count} lines.")
    # fout.close()
    return edge_ctr, time_ctr


# def load_covidflight_temporarl_edgelist(fname):
#     edgelist = open(fname, "r")
#     lines = list(edgelist.readlines())
#     edgelist.close()

#     for i in range(1, len(lines)):
#         line = lines[i]
#         values = line.split(',')


def main():
    # folder = "datasets/flight/"
    # suffix = ".csv"
    # fname = "flightlist_20190101_20190131"
    # outname = "20190101_20190131"

    folder = "/mnt/c/Users/sheny/Desktop/flight_network/"
    files = glob.glob(folder + "*.csv")
    total_edges = 0
    total_time = 0

    for file in files:
        edge_ctr, time_ctr = flight2edgelist(file)
        total_edges += edge_ctr
        total_time += time_ctr

    print("there are ", total_edges, " total edges")
    print("there are ", total_time, " total time")


if __name__ == "__main__":
    main()
