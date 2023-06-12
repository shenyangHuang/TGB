import dateutil.parser as dparser
import csv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from os import listdir


def count_unique_countries(fname):
    node_dict = {}

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        # year,u,v,w
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                year = int(row[0])
                u = row[1]
                v = row[2]
                w = float(row[3])
                if u not in node_dict:
                    node_dict[u] = 1
                if v not in node_dict:
                    node_dict[v] = 1

    print("there are {} unique countries".format(len(node_dict)))



#! incorrect, do not use
def normalize_edgelist(fname: str, outname: str):
    """
    need to track id for nodes
    normalize the edgelist by row for each year
    """
    prev_t = 0
    uid = 0
    node_dict = {}
    year_dict = {}

    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["year", "nation", "trading nation", "weight"]
        write.writerow(fields)
        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    ts = int(row[0])
                    u = row[1]
                    v = row[2]
                    w = float(row[3])
                    if line_count == 1:
                        prev_t = ts
                    if u not in node_dict:
                        node_dict[u] = uid
                        uid += 1
                    if v not in node_dict:
                        node_dict[v] = uid
                        uid += 1
                    if w == 0:
                        line_count += 1
                        continue

                    if ts != prev_t:  # a new year now, write everything
                        # normalize the counts
                        for u in year_dict:
                            if np.sum(year_dict[u]) == 0:
                                continue
                            year_dict[u] = year_dict[u] / np.sum(year_dict[u])
                            invert_dict = {v: k for k, v in node_dict.items()}
                            for v in range(len(year_dict[u])):
                                if year_dict[u][v] > 0:
                                    write.writerow(
                                        [prev_t, u, invert_dict[v], year_dict[u][v]]
                                    )
                        year_dict = {}
                        prev_t = ts
                    else:
                        if u not in year_dict:
                            year_dict[u] = np.zeros(255)
                            year_dict[u][node_dict[v]] = w
                        else:
                            year_dict[u][node_dict[v]] = w
                    line_count += 1


def generate_aggregate_labels(fname: str, outname: str):
    """
    aggregate the node label for next year
    """

    ts_init = 1986

    # ts, src, subreddit, num_words, score
    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["year", "nation", "trading nation", "weight"]
        write.writerow(fields)

        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            # ts, src, subreddit, num_words, score
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    ts = int(row[0])
                    u = row[1]
                    v = row[2]
                    w = float(row[3])
                    if (ts > ts_init):
                        write.writerow([ts, u, v, w])
                    line_count += 1


def check_sum_to_one(fname: str):
    """
    just to check if weights sum to 1 in a year
    """
    u_dict = {}
    ts_prev = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        # ts, src, subreddit, num_words, score
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                ts = int(row[0])
                if (line_count == 1):
                    ts_prev = ts
                if (ts != ts_prev):
                    ts_prev = ts
                    for u in u_dict:
                        print (u_dict[u])
                    u_dict = {}
                u = row[1]
                v = row[2]
                w = float(row[3])
                if (u not in u_dict):
                    u_dict[u] = w
                else:
                    u_dict[u] += w
                line_count += 1




def main():
    #! should have the normalized version on the edgelist

    # #find the number of unique countries
    # fname = "un_trade_edgelist.csv"
    # count_unique_countries(fname)

    #! normalize edgelist by row for each year
    # fname = "un_trade_edgelist.csv"
    # outname = "un_trade_edgelist_normalized.csv"
    # normalize_edgelist(fname, outname)

    #! find the node label for next year
    # * the node labels are simply the edgelist in this case
    # fname = "un_trade_edgelist.csv"
    # outname = "un_trade_node_labels.csv"
    # generate_aggregate_labels(fname, outname)


    # #! check if all sums are correct
    # fname = "un_trade_node_labels.csv"
    # check_sum_to_one(fname)



if __name__ == "__main__":
    main()
