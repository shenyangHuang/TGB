import csv
from tqdm import tqdm
from os import listdir
from tgb.utils.stats import analyze_csv


def find_filenames(path_to_dir):
    r"""
    find all files in a folder
    Parameters:
        path_to_dir (str): path to the directory
    """
    filenames = listdir(path_to_dir)
    return filenames


def combine_edgelist_edgefeat2subreddits(edgefname, featfname, outname):
    """
    combine edgelist and edge features
    'ts', 'src', 'subreddit', 'num_words', 'score'
    """
    line_idx = 0

    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["ts", "src", "subreddit", "num_words", "score"]
        write.writerow(fields)
        sub_id = 0
        edgelist = open(edgefname, "r")
        edgefeat = open(featfname, "r")
        edgelist.readline()
        edgefeat.readline()

        while True:
            #'ts', 'src', 'dst', 'edge_id'
            edge_line = edgelist.readline()
            edge_line = edge_line.split(",")
            if len(edge_line) < 4:
                break
            edge_id = int(edge_line[3])
            ts = int(edge_line[0])
            src = int(edge_line[1])

            #'edge_id', 'subreddit', 'num_characters', 'num_words', 'score', 'edited_flag'
            feat_line = edgefeat.readline()
            feat_line = feat_line.split(",")
            edge_id_feat = int(feat_line[0])
            subreddit = feat_line[1]
            num_words = int(feat_line[3])
            score = int(feat_line[4])

            if edge_id != edge_id_feat:
                print("edge_id != edge_id_feat")
                print(edge_id)
                print(edge_id_feat)
                break

            write.writerow([ts, src, subreddit, num_words, score])
            line_idx += 1
    print("processed", line_idx, "lines")


def filter_subreddits(fname):
    """
    check the frequency of subreddits
    """
    subreddit_count = {}
    node_count = {}
    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        # ts, src, subreddit, num_words, score
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                ts = row[0]
                src = row[1]
                subreddit = row[2]
                if subreddit not in subreddit_count:
                    subreddit_count[subreddit] = 1
                else:
                    subreddit_count[subreddit] += 1
                if src not in node_count:
                    node_count[src] = 1
                else:
                    node_count[src] += 1
    return subreddit_count, node_count


def clean_edgelist(fname, node_counts, outname, threshold=1000):
    """
    helper function for filtering out low frequency nodes
    """
    node_dict = {}

    for node in node_counts:
        if node_counts[node] >= threshold:
            node_dict[node] = 1

    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["ts", "user", "subreddit", "num_words", "score"]
        write.writerow(fields)

        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            # ts, src, subreddit, num_words, score
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    ts = row[0]
                    src = row[1]
                    subreddit = row[2]
                    num_words = int(row[3])
                    score = int(row[4])
                    if src in node_dict:
                        write.writerow([ts, src, subreddit, num_words, score])


def clean_edgelist_reddits(fname, reddit_counts, outname, threshold=50):
    """
    helper function for filtering out low frequency subreddits
    """
    reddit_dict = {}

    for reddit in reddit_counts:
        if reddit_counts[reddit] >= threshold:
            reddit_dict[reddit] = 1
    print ("there remains, ", len(reddit_dict), " subreddits")

    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["ts", "user", "subreddit", "num_words", "score"]
        write.writerow(fields)
        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            # ts, src, subreddit, num_words, score
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    ts = row[0]
                    src = row[1]
                    subreddit = row[2]
                    num_words = int(row[3])
                    score = int(row[4])
                    if subreddit in reddit_dict:
                        write.writerow([ts, src, subreddit, num_words, score])


def remove_missing_user(fname, outname):
    """
    remove all lines that are missing the user
    """

    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["ts", "user", "subreddit", "num_words", "score"]
        write.writerow(fields)
        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            # ts, src, subreddit, num_words, score
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    ts = row[0]
                    src = int(row[1])
                    subreddit = row[2]
                    num_words = int(row[3])
                    score = int(row[4])
                    if src != -1:
                        write.writerow([ts, src, subreddit, num_words, score])


def generate_daily_node_labels(
    fname: str,
    outname: str,
):
    r"""
    function for generating daily node labels then can be used for aggregation
    """

    day_dict = {}  # store the weights of genres on this day
    prev_t = -1
    DAY_IN_SEC = 86400
    # WEEK_IN_SEC = 604800
    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["ts", "user", "subreddit", "weight"]
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
                    user_id = row[1]
                    subreddit = row[2]

                    if line_count == 1:
                        prev_t = ts

                    if (prev_t + DAY_IN_SEC) < ts:
                        #! user,time,genre,weight  # genres = # of weights
                        out = [user_id, ts]
                        for subreddit, w in day_dict.items():
                            write.writerow(out + [subreddit] + [w])
                        prev_t = ts
                        day_dict = {}
                    else:
                        if subreddit not in day_dict:
                            day_dict[subreddit] = 1
                        else:
                            day_dict[subreddit] += 1
                    line_count += 1


#! note that the edgelist are not sorted by users then by time, should keep multiple users when aggregating
def generate_aggregate_labels(fname: str, outname: str, days: int = 7):
    """
    aggregate the genres over a number of days,  as specified by days
    prediction should always be at the first second of the day
    #! daily labels are always shifted by 1 day
    """

    ts_prev = 0

    DAY_IN_SEC = 86400
    timespan = days * DAY_IN_SEC

    user_dict = {}

    # ts, src, subreddit, num_words, score
    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["ts", "user", "subreddit", "weight"]
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
                    user = row[1]
                    subreddit = row[2]
                    w = int(row[3])
                    if line_count == 1:
                        ts_prev = ts

                    if (ts - ts_prev) > timespan:
                        for user in user_dict:
                            total = sum(user_dict[user].values())
                            subreddit_dict = {
                                k: v / total for k, v in user_dict[user].items()
                            }
                            for subreddit, w in subreddit_dict.items():
                                write.writerow(
                                    [ts_prev + DAY_IN_SEC, user, subreddit, w]
                                )
                        user_dict = {}
                        ts_prev = ts_prev + DAY_IN_SEC  #! move label to the next day
                    else:
                        if user in user_dict:
                            if subreddit in user_dict[user]:
                                user_dict[user][subreddit] += w
                            else:
                                user_dict[user][subreddit] = w
                        else:
                            user_dict[user] = {}
                            user_dict[user][subreddit] = w
                    line_count += 1


def main():
    # #? see redditcomments.py for the extraction from the raw files

    #! combine edgelist and edge feat file check if the edge_id matches
    # edgefname = "redditcomments_edgelist_2008_2010.csv"
    # featfname = "redditcomments_edgefeat_2008_2010.csv"
    # outname = "subreddits_edgelist.csv"
    # combine_edgelist_edgefeat2subreddits(edgefname, featfname, outname)

    #! remove all edges missing user
    # fname = "subreddits_edgelist.csv"
    # outname = "subreddits_edgelist_filtered.csv"
    # remove_missing_user(fname,
    #                     outname)

    #! should clean subreddits first, frequency count of reddits
    # fname = "subreddits_edgelist.csv"
    # outname = "subreddits_edgelist_filter.csv"
    # subreddit_count, node_count = filter_subreddits(fname)
    # threshold = 1000 #200 #100
    # clean_edgelist_reddits(fname, subreddit_count, outname, threshold=threshold)


    #! filter out nodes with low frequency frequency count of nodes
    # fname = "subreddits_edgelist.csv"
    # outname = "subreddits_edgelist_clean.csv"
    # subreddit_count, node_count = filter_subreddits(fname)
    # threshold = 1000
    # clean_edgelist(fname, node_count, outname, threshold=threshold)
    # print ("finish cleaning")

    #! generate aggregate labels, the label for each day is shifted by 1 day as it uses the edges from today
    # fname = "subreddits_edgelist.csv"
    # outname = "subreddits_node_labels.csv"
    # generate_aggregate_labels(fname, outname, days=7)

    #! analyze the extracted csv
    fname = "subreddits_edgelist.csv"
    analyze_csv(fname)


    
    # #! generate daily node labels
    # outname = 'subreddits_daily_labels.csv'
    # fname = "subreddits_edgelist.csv"
    # generate_daily_node_labels(fname,outname)


if __name__ == "__main__":
    main()
