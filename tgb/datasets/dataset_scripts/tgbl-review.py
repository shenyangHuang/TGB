import pyarrow.dataset as ds
import csv
import numpy as np
from tgb.utils.stats import analyze_csv
import pandas as pd
from tqdm import tqdm


def collect_csv(dir_name="software"):
    dataset = ds.dataset(dir_name, format="csv")
    df = dataset.to_table().to_pandas()
    df.to_csv(dir_name + ".csv", index=True)


def reorder_column(fname: str, outname: str):
    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["ts", "source", "target", "weight"]
        write.writerow(fields)
        line_count = 0
        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if line_count == 0:  # header
                    line_count += 1
                else:
                    # edgeid, SourceId,TargetId,Weight,Timestamp
                    src = row[1]
                    dst = row[2]
                    w = row[3]
                    ts = row[4]
                    write.writerow([ts, src, dst, w])
                    line_count += 1


def sort_edgelist(fname: str, outname: str):
    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["ts", "source", "target", "weight"]
        write.writerow(fields)
        line_count = 0
        ts_list = []
        line_list = []

        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if line_count == 0:  # header
                    line_count += 1
                else:
                    ts = int(row[0])
                    src = row[1]
                    dst = row[2]
                    w = row[3]
                    ts_list.append(ts)
                    line_list.append([ts, src, dst, w])
                    # write.writerow([ts, src, dst, w])
                    line_count += 1

        ts_list = np.array(ts_list)
        idx = np.argsort(ts_list)
        idx = idx.tolist()

        line_list_out = []
        for i in idx:
            line_list_out.append(line_list[i])
        for line in line_list_out:
            write.writerow(line)


def count_degree(fname: str):
    node_counts = {}
    line_count = 0
    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if line_count == 0:  # header
                line_count += 1
            else:
                ts = int(row[0])
                src = row[1]
                dst = row[2]
                w = row[3]
                if src not in node_counts:
                    node_counts[src] = 1
                else:
                    node_counts[src] += 1

                if dst not in node_counts:
                    node_counts[dst] = 1
                else:
                    node_counts[dst] += 1
                line_count += 1
    return node_counts


def reduce_edgelist(fname: str, outname: str, node10_id: dict):
    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["ts", "source", "target", "weight"]
        write.writerow(fields)
        line_count = 0

        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if line_count == 0:  # header
                    line_count += 1
                else:
                    ts = int(row[0])
                    src = row[1]
                    dst = row[2]
                    w = row[3]
                    if (src in node10_id) and (dst in node10_id):
                        write.writerow([ts, src, dst, w])
                    line_count += 1

"""
function for review
"""
def csv_process_review(
    fname: str,
    outname: str = "review.csv",
) -> pd.DataFrame:
    r"""
    used for processing review dataset, helper function, not used in actual dataloading
    input .csv file format should be: timestamp, node u, node v, attributes
    Parameters:
        fname: the path to the raw data
    Returns:
        df: a pandas dataframe containing the edgelist data
        feat_l: a numpy array containing the node features
        node_ids: a dictionary mapping node id to integer
    """
    src_ids = {}
    dst_ids = {}
    src_ctr = 0
    dst_ctr = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        """
        ts,source,target,weight
        929232000,137139,30122,5.0
        930787200,129185,175070,2.0
        931824000,246213,30122,2.0
        """
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
                continue
            else:
                ts = int(row[0])
                src = row[1]
                dst = row[2]
                if src not in src_ids:
                    src_ids[src] = src_ctr
                    src_ctr += 1
                if dst not in dst_ids:
                    dst_ids[dst] = dst_ctr
                    dst_ctr += 1
                w = float(row[3])
    
    #! ensure that source and destination nodes are unique and non-overlapping
    src_ctr += 1
    dst_ids = {k:v+src_ctr for k,v in dst_ids.items()}

    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["ts","source","target","weight"]
        write.writerow(fields)

        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            idx = 0
            """
            ts,source,target,weight
            929232000,137139,30122,5.0
            930787200,129185,175070,2.0
            931824000,246213,30122,2.0
            """
            for row in tqdm(csv_reader):
                if idx == 0:
                    idx += 1
                    continue
                else:
                    ts = int(row[0])
                    src = src_ids[row[1]]
                    dst = dst_ids[row[2]]
                    w = float(row[3])
                write.writerow([ts,src,dst,w])


def main():
    # # collect csv
    # # collect_csv(dir_name = "software")
    # collect_csv(dir_name="books")
    # # collect_csv(dir_name = "electronics")

    # # #* reorder column
    # # fname = "electronics.csv"
    # # outname = "amazonreview_edgelist.csv"
    # # reorder_column(fname,
    # #                outname)

    # # #* sort edgelist
    # # fname = "amazonreview_edgelist.csv"
    # # outname = "amazonreview_edgelist_sort.csv"
    # # sort_edgelist(fname,
    # #               outname)

    # fname = "amazonreview_edgelist_reduce.csv"
    # analyze_csv(fname)

    # # fname = "amazonreview_edgelist.csv"
    # # node_counts = count_degree(fname)
    # # node10_id = {}
    # # for node in node_counts:
    # #     if node_counts[node] > 10:
    # #         node10_id[node] = node_counts[node]

    # # outname = "amazonreview_edgelist_reduce.csv"
    # # reduce_edgelist(fname,
    # #                 outname,
    # #                 node10_id)

    csv_process_review("tgbl-review_edgelist_v2.csv", "review.csv")

    


if __name__ == "__main__":
    main()
