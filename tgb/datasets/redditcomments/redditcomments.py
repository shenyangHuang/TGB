import dateutil.parser as dparser
import csv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from os import listdir


def find_filenames(path_to_dir):
    r"""
    find all files in a folder
    Parameters:
    	path_to_dir (str): path to the directory
    """
    filenames = listdir(path_to_dir)
    return filenames

def read_edgelist(fname, 
                  outfname):
    """
    read a space separated edgelist
    comment’s author, author of the parent (the post that the comment is replied to), comment’s creation time, comment’s edge id
    u,v,t,edge_id
    3746738	1637382	1551398391	31534079835
    Parameters:
        fname (str): path to the edgelist
        outfname (str): path to the output file
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    with open(outfname, 'a') as outf:
        write = csv.writer(outf)
        fields = ['ts', 'src', 'dst', 'edge_id']
        write.writerow(fields)
        for line in tqdm(lines):
            line = line.split()
            src = line[0]
            dst = line[1]
            ts = line[2]
            edge_id = line[3]
            write.writerow([ts, src, dst, edge_id])



def main():
    f_dir = "raw/"
    fnames = find_filenames(f_dir)
    outname = "output.csv"


if __name__ == "__main__":
    main()

