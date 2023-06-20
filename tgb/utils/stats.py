"""
script for generating statistics from the dataset
"""
import csv
import numpy as np
# import matplotlib.pyplot as plt


"""
#! analyze statistics from the dataset
#* 1). # of unique nodes, 2). # of edges. 3). # of unique edges, 4). # of timestamps 5). recurrence of nodes
"""


def analyze_csv(fname):
    node_dict = {}
    edge_dict = {}
    num_edges = 0
    num_time = 0
    time_dict = {}

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                # t,u,v,w
                t = row[0]
                u = row[1]
                v = row[2]

                # count unique time
                if t not in time_dict:
                    time_dict[t] = 1
                    num_time += 1

                # unique nodes
                if u not in node_dict:
                    node_dict[u] = 1
                else:
                    node_dict[u] += 1

                if v not in node_dict:
                    node_dict[v] = 1
                else:
                    node_dict[v] += 1

                # unique edges
                num_edges += 1
                if (u, v) not in edge_dict:
                    edge_dict[(u, v)] = 1
                else:
                    edge_dict[(u, v)] += 1

    print("----------------------high level statistics-------------------------")
    print("number of total edges are ", num_edges)
    print("number of nodes are ", len(node_dict))
    print("number of unique edges are ", len(edge_dict))
    print("number of unique timestamps are ", num_time)

    num_10 = 0
    num_100 = 0
    num_1000 = 0

    for node in node_dict:
        if node_dict[node] >= 10:
            num_10 += 1
        if node_dict[node] >= 100:
            num_100 += 1
        if node_dict[node] >= 1000:
            num_1000 += 1
    print("number of nodes with # edges >= 10 is ", num_10)
    print("number of nodes with # edges >= 100 is ", num_100)
    print("number of nodes with # edges >= 1000 is ", num_1000)
    print("----------------------high level statistics-------------------------")


def plot_curve(y: np.ndarray, outname: str) -> None:
    """
    plot the training curve given y
    Parameters:
        y: np.ndarray, the training curve
        outname: str, the output name
    """
    plt.plot(y, color="#fc4e2a")
    plt.savefig(outname + ".pdf")
    plt.close()


def main():
    fname = "tgb/datasets/tgbl-wiki/tgbl-wiki_edgelist.csv"
    analyze_csv(fname)


if __name__ == "__main__":
    main()
