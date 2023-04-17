import dateutil.parser as dparser
import re
import networkx as nx
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np




'''
process all rows with origin, destination and day
and save it as an edgelist file
'''
def flight2edgelist(fname, outname):
    fout = open(outname, "w")

    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
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
                fout.write(str(day) + "," + str(origin) + "," + str(destination) + "\n")
                line_count += 1
        print(f'Processed {line_count} lines.')
    fout.close()


