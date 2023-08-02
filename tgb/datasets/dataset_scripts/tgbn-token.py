import csv
import datetime

def count_node_freq(fname, filter_size=100):

    node_dict = {}
    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        ctr = 0
        for row in csv_reader:
            if ctr == 0:
                ctr += 1
                continue
            else:
                token_type = row[0]
                src = row[1]
                if (src not in node_dict):
                    node_dict[src] = 1
                else:
                    node_dict[src] += 1
                dst = row[2]
                if (dst not in node_dict):
                    node_dict[dst] = 1
                else:
                    node_dict[dst] += 1
                ctr += 1

    num_10 = 0
    num_100 = 0
    num_1000 = 0
    num_2000 = 0
    num_5000 = 0 

    for node in node_dict:
        if node_dict[node] >= 10:
            num_10 += 1
        if node_dict[node] >= 100:
            num_100 += 1
        if node_dict[node] >= 1000:
            num_1000 += 1
        if node_dict[node] >= 2000:
            num_2000 += 1
        if node_dict[node] >= 5000:
            num_5000 += 1

    print("number of nodes with # edges >= 10 is ", num_10)
    print("number of nodes with # edges >= 100 is ", num_100)
    print("number of nodes with # edges >= 1000 is ", num_1000)
    print("number of nodes with # edges >= 2000 is ", num_2000)
    print("number of nodes with # edges >= 5000 is ", num_5000)
    print("----------------------high level statistics-------------------------")


    #! keep nodes with at least 100 edges
    node_dict_filtered = {}
    for node in node_dict:
        if node_dict[node] >= filter_size:
            node_dict_filtered[node] = node_dict[node]
    return node_dict_filtered






def filter_edgelist(token_fname, edgefile, outname):
    """
    preserve only the tokens in the token file
    Parameters:
        token_fname: the file of the token file
        edgefile: the edgelist file name
        outname: the output filtered edgelistname
    """
    #* read tokens from the file
    token_dict = {}
    with open(token_fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        ctr = 0
        for row in csv_reader:
            if ctr == 0:
                ctr += 1
                continue
            else:
                token_type = row[0]
                token_dict[token_type] = 1
    
    with open(edgefile, "r") as in_file:
        with open(outname, "w") as out_file:
            csv_reader = csv.reader(in_file, delimiter=",")
            csv_writer = csv.writer(out_file, delimiter=",")
            csv_writer.writerow(["token_address", "from_address", "to_address", "value", "block_timestamp"])
            ctr = 0
            for row in csv_reader:
                if ctr == 0:
                    ctr += 1
                    continue
                else:
                    token_type = row[0]
                    if token_type in token_dict:
                        csv_writer.writerow(row)
                    ctr += 1


def filter_by_node(node_dict, edgefile, outname):
    with open(edgefile, "r") as in_file:
        with open(outname, "w") as out_file:
            csv_reader = csv.reader(in_file, delimiter=",")
            csv_writer = csv.writer(out_file, delimiter=",")
            csv_writer.writerow(["token_address", "from_address", "to_address", "value", "block_timestamp"])
            ctr = 0
            for row in csv_reader:
                if ctr == 0:
                    ctr += 1
                    continue
                else:
                    token_type = row[0]
                    src = row[1]
                    dst = row[2]
                    if (src in node_dict) or (dst in node_dict):
                        csv_writer.writerow(row)
                    ctr += 1



def store_node_list(node_dict, outname):
    """
    Parameters:
        outname: name of the output csv file
    Output:
        output csv file with node list
    """
    with open(outname, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(["node_list", "frequency"])
        for key, value in node_dict.items():
            csv_writer.writerow([key, value])


def load_node_dict(fname):
    node_dict = {}
    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        ctr = 0
        for row in csv_reader:
            if ctr == 0:
                ctr += 1
                continue
            else:
                node = row[0]
                freq = int(row[1])
                node_dict[node] = freq
                ctr += 1
    return node_dict







def store_token_address(token_dict, outname, topk=1000):
    """
    Parameters:
        outname: name of the output csv file
    Output:
        output csv file with topk token addresses
    """
    sorted_tokens = {k: v for k, v in sorted(token_dict.items(), key=lambda item: item[1], reverse=True)}
    ctr = 0
    with open(outname, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(["token_address", "frequency"])
        for key, value in sorted_tokens.items():
            if (ctr <= topk):
                csv_writer.writerow([key, value])
            else:
                break
            ctr += 1

def analyze_token_frequency(fname):
    # ['token_address', 'from_address', 'to_address', 'value', 'block_timestamp']
    token_dict = {}
    node_dict = {}
    time_dict = {}
    max_w = 0
    min_w = 100000
    num_edges = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        ctr = 0
        for row in csv_reader:
            if ctr == 0:
                ctr += 1
                continue
            else:
                token_type = row[0]
                if (token_type not in token_dict):
                    token_dict[token_type] = 1
                else:
                    token_dict[token_type] += 1
                src = row[1]
                if (src not in node_dict):
                    node_dict[src] = 1
                else:
                    node_dict[src] += 1
                dst = row[2]
                if (dst not in node_dict):
                    node_dict[dst] = 1
                else:
                    node_dict[dst] += 1

                w = float(row[3])
                if (w > max_w):
                    max_w = w
                elif (w < min_w):
                    min_w = w
                timestamp = row[4]
                if (timestamp not in time_dict):
                    time_dict[timestamp] = 1
                ctr += 1
                num_edges += 1

    print ( "number of edges are ", num_edges)
    print (" number of unique tokens are ", len(token_dict))
    print (" number of unique nodes are ", len(node_dict))
    print (" number of unique timestamps are ", len(time_dict))
    print (" max weight is ", max_w)
    print (" min weight is ", min_w)

    # topk = 1000
    # store_token_address(token_dict, "token_list.csv", topk=topk)

def to_bipartite(in_name, out_name, node_dict):
    """
    load and convert a user-user graph into a user-token bipartite graph
    """
    with open(in_name, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        with open(out_name, "w") as out_file:
            csv_writer = csv.writer(out_file, delimiter=",")
            csv_writer.writerow(["timestamp", "user_address", "token_address", "value", "IsSender"])
            ctr = 0
            for row in csv_reader:
                if ctr == 0:
                    ctr += 1
                    continue
                else:
                    token_type = row[0]
                    src = row[1]
                    dst = row[2]
                    w = float(row[3])
                    timestamp = row[4]
                    if (src in node_dict):
                        csv_writer.writerow([timestamp, src, token_type, w, 1])
                    if (dst in node_dict):
                        csv_writer.writerow([timestamp, dst, token_type, w, 0])
                    

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



def convert_2_sec(fname, outname):
    """
    convert datetime object format = "%Y-%m-%d %H:%M:%S" to seconds
    #2017-07-24 17:48:15+00:00
    """
    format = "%Y-%m-%d %H:%M:%S"
    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        with open(outname, "w") as out_file:
            csv_writer = csv.writer(out_file, delimiter=",")
            csv_writer.writerow(["timestamp", "user_address", "token_address", "value", "IsSender"])
            ctr = 0
            for row in csv_reader:
                if ctr == 0:
                    ctr += 1
                    continue
                else:
                    timestamp = row[0][:19]
                    date_object = datetime.datetime.strptime(timestamp, format)
                    timestamp_sec = int(date_object.timestamp())
                    src = row[1]
                    dst = row[2]
                    w = float(row[3])
                    IsSender = int(row[4])
                    if (w != 0):
                        csv_writer.writerow([timestamp_sec, src, dst, w, IsSender])

    



def print_csv(fname):
    # ['token_address', 'from_address', 'to_address', 'value', 'block_timestamp']
    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        ctr = 0
        for row in csv_reader:
            ctr += 1
    print ("there are ", ctr, " rows in the csv file")



def sort_edgelist_by_time(fname, outname):
    row_dict = {}
    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        with open(outname, "w") as out_file:
            csv_writer = csv.writer(out_file, delimiter=",")
            csv_writer.writerow(["timestamp", "user_address", "token_address", "value", "IsSender"])
            ctr = 0
            for row in csv_reader:
                if ctr == 0:
                    ctr += 1
                    continue
                else:
                    timestamp =int(row[0])
                    if (timestamp not in row_dict):
                        row_dict[timestamp] = [row]
                    else:
                        row_dict[timestamp].append(row)
            for i in sorted(row_dict.keys()):
                rows = row_dict[i]
                for row in rows:
                    csv_writer.writerow(row)





#! aggregate node labels
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
        fields = ["ts", "user_address", "token_address", "weight"] #["ts", "user", "subreddit", "weight"]
        write.writerow(fields)

        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            # ts, src, subreddit, num_words, score
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    ts = float(row[0])
                    ts = int(ts)
                    user = row[1]
                    item = row[2]
                    w = float(row[3])
                    if (w == 0):
                        print (row)

                    if line_count == 1:
                        ts_prev = ts

                    if (ts - ts_prev) > timespan:
                        for user in user_dict:
                            total = sum(user_dict[user].values())
                            item_dict = {
                                k: v / total for k, v in user_dict[user].items()
                            }
                            for item, w in item_dict.items():
                                write.writerow(
                                    [ts_prev + DAY_IN_SEC, user, item, w]
                                )
                        user_dict = {}
                        ts_prev = ts_prev + DAY_IN_SEC  #! move label to the next day
                    else:
                        if user in user_dict:
                            if item in user_dict[user]:
                                user_dict[user][item] += w
                            else:
                                user_dict[user][item] = w
                        else:
                            user_dict[user] = {}
                            user_dict[user][item] = w
                    line_count += 1




def main():

    """
    processing token types
    """
    # fname = "ERC20_token_network.csv"
    # #analyze_token_frequency(fname)

    # token_file = "token_list.csv"
    # outname = "filtered_token_edgelist.csv"

    #! filter by token frequency
    # filter_edgelist(token_file, fname, outname)
    # #print_csv(fname)
    # #analyze_csv(fname)

    """
    processing node dict
    """
    # fname = "filtered_token_edgelist.csv"
    # #! filter by node frequency
    # node_dict = count_node_freq(fname, filter_size=100)
    # store_node_list(node_dict, "node_list.csv")
    # #store_token_address(node_dict, "node_list.csv", topk=0)

    # outname = "tgbl-token-edgelist_100.csv"
    # filter_by_node(node_dict, fname, outname)
    # analyze_token_frequency('tgbl-token-edgelist_100.csv')


    #! converting user-user graph to user-token bipartite graph
    # out_name = "tgbl-token_edgelist.csv"
    # node_dict = load_node_dict("node_list.csv")
    # to_bipartite('tgbl-token-edgelist_100.csv', out_name, node_dict)
    # analyze_csv(out_name)


    #! convert datetime to seconds
    #convert_2_sec("tgbl-token_edgelist_old.csv", "tgbn-token_edgelist.csv")


    #! sort the timestamps in the edgelist
    # fname = "tgbn-token_edgelist.csv"
    # outname = "tgbn-token_edgelist_sorted.csv"
    # sort_edgelist_by_time(fname, outname)



    #! generate node labels
    edgefile = "tgbn-token_edgelist.csv"
    outfile = "tgbn-token_node_labels.csv"
    days = 7
    generate_aggregate_labels(edgefile, outfile, days=days)





    




if __name__ == "__main__":
    main()