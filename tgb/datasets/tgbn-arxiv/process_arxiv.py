import json
import networkx as nx 
import numpy as np
import csv
from datetime import date


def load_full_json(fname):
    json_str = ""
    ctr = 0
    with open(fname, "r", encoding='utf-8') as f:

        #TODO need to determine how many lines form a json object 
        for line in f:
            data = json.loads(line)
            print (data)
            quit() #remove this when you write the code


def main():
    fname = "nodes.json"
    load_full_json(fname)


if __name__ == "__main__":
    main()
