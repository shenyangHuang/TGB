
"""
Data source: 
# https://surfdrive.surf.nl/files/index.php/s/M09RDerAMZrQy8q#editor

# https://dl.acm.org/doi/abs/10.1145/3487553.3524699

also see here: https://arxiv.org/pdf/1803.03697

#  Temporal Social Network Dataset of Reddit
### Dataset to accompany “A large-scale temporal analysis of user lifespan durability on the Reddit social media platform” (WWW 2022).

## Overview

This dataset consists of more than 6.7 billion Reddit comment interactions made from the beginning of Reddit in 2005 until the end of 2019. 

### Nodes

Nodes in the network represent users who posted at least one comment or one submission until the end of 2019 and have not deleted their accounts by the time of the data ingestion. 

Each user is assigned a unique identifier starting from 0, and -1 is the identifier of the node representing deleted users. The `nodes` file maintains the node-identifier assignment to each user’s username.

### Edges

For each month of our data, we maintain two separate files, an edge file that consists of temporal edges data and an attribute file that consists of attributes of each interaction. All these files are in a tab-separated format. The compressed edge files and compressed attribute files are available in the `edges` and the `attributes` directory, respectively. The name of files indicates the timeframe they belong to.

Each line in an edge file corresponds to a comment and includes:

 - comment’s author
 - author of the parent (the post that the comment is replied to)
 - comment’s creation time
 - comment’s edge id

Each line in an attribute file corresponds to the line with the same line number in the corresponding edge file and includes:

 - comment’s edge id
 - Reddit’s identifier of the comment
 - Reddit’s identifier of the parent (the post that the comment is replied to)
 - Reddit’s identifier of the submission that the comment is in
 - name of the subreddit that the comment is in
 - number of characters in the comment’s body
 - number of words in the comment’s body
 - score of the comment
 - a flag indicating if the comment has been edited

### Stats

Size (compressed): 125GB
Size (uncompressed): 652GB
Number of nodes: 62,402,844
Number of edges: 6,728,759,080

### Notes

Reddit banned the subreddit `/r/Incels` in November of 2017, and its data is no longer available via the Reddit API. This has resulted in the loss of score data for 119,111 comments made in October and November of 2017 in this subreddit. The affected entries have a null value as their score. 

## Citation

If you want to reuse this dataset, you can reference it as follows:

A. Nadiri and F.W. Takes, A large-scale temporal analysis of user lifespan durability on the Reddit social media platform, in Proceedings of the 28th ACM International Web Conference (TheWebConf) Workshops, 2022.

## Online repository

The dataset is available for download at [**LINK**](https://surfdrive.surf.nl/files/index.php/s/M09RDerAMZrQy8q)

## Acknowledgments

The dataset is constructed using data provided by [The Pushshift Reddit Dataset](https://ojs.aaai.org/index.php/ICWSM/article/view/7347)


"""



"""
ideas for temporal heterogenous graph in reddit data:

node types:
1. user
2. subreddit

edge types
1. user post in subreddit (top level)
2. user replies to another user 
3. user replies in subreddit



# node types:
# 1. user 
# 2. subreddit
# 3. comment


# edge types
# 1. user makes comment in subreddit (top level comment)
# 2. user replies to comment in subreddit (comments that has a parent)
# 2. comment is child of comment (comments that has a parent)
# 3. comment belongs to subreddit
"""
import csv
from tgb.utils.utils import save_pkl, load_pkl


def load_csv_raw(fname):
    """
    load the raw csv file and merge them into one
    ts, src, dst, subreddit, reddit_id, reddit_parent_id, num_words, score
    """
    out_dict = {}
    num_lines = 0
    max_words = 0
    min_words = 10000000

    max_score = 0
    min_score = 1000000

    """
    relation types:
    0: user replies to user
    1: user replies to subreddit

    node types:
    0: user
    1: subreddit
    """

    node_dict = {}
    node_type_dict = {}
    reddit_deg_dict = {}
    node_deg_dict = {}
    header = True
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter =',')
        #* ts, src, dst, subreddit, reddit_id, reddit_parent_id, num_words, score
        #? 1388534400,32183137,51851117,AskReddit,t1_ceefsvy,t3_1u4kbf,32,1
        for row in reader: 
            if header:
                header = False
                continue
            ts = int(row[0])
            src = row[1]
            if (src not in node_dict):
                node_dict[src] = len(node_dict)
                node_type_dict[node_dict[src]] = 0
            dst = row[2]
            if (dst not in node_dict):
                node_dict[dst] = len(node_dict)
                node_type_dict[node_dict[dst]] = 0

            if (src not in node_deg_dict):
                node_deg_dict[src] = 1
            else:
                node_deg_dict[src] += 1
            if (dst not in node_deg_dict):
                node_deg_dict[dst] = 1
            else:
                node_deg_dict[dst] += 1

            subreddit = row[3]
            if (subreddit not in node_dict):
                node_dict[subreddit] = len(node_dict)
                node_type_dict[node_dict[subreddit]] = 1
            
            if (subreddit not in reddit_deg_dict):
                reddit_deg_dict[subreddit] = 1
            else:
                reddit_deg_dict[subreddit] += 1
            
            num_words = int(row[6])
            if (num_words > max_words):
                max_words = num_words
            if (num_words < min_words):
                min_words = num_words
            score = int(row[7])
            if (score > max_score):
                max_score = score
            if (score < min_score):
                min_score = score

            if (ts in out_dict):
                out_dict[ts][(node_dict[src], node_dict[dst], 0)] = (num_words, score)
                out_dict[ts][(node_dict[src], node_dict[subreddit], 1)] = (num_words, score)
            else:
                out_dict[ts] = {}
                out_dict[ts][(node_dict[src], node_dict[dst], 0)] = (num_words, score)
                out_dict[ts][(node_dict[src], node_dict[subreddit], 1)] = (num_words, score)
            num_lines += 1

    print ("max words: ", max_words)
    print ("min words: ", min_words)
    print ("max score: ", max_score)
    print ("min score: ", min_score)
    return out_dict, num_lines, node_dict, node_type_dict, reddit_deg_dict, node_deg_dict



def load_csv_filtered_node(fname, low_deg_dict):
    """
    load the raw csv file, remove edges with low degree nodes
    ts, src, dst, subreddit, reddit_id, reddit_parent_id, num_words, score
    """
    out_dict = {}
    num_lines = 0
    """
    relation types:
    0: user replies to user
    1: user replies to subreddit

    node types:
    0: user
    1: subreddit
    """

    node_dict = {}
    node_type_dict = {}
    header = True
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter =',')
        #* ts, src, dst, subreddit, reddit_id, reddit_parent_id, num_words, score
        #? 1388534400,32183137,51851117,AskReddit,t1_ceefsvy,t3_1u4kbf,32,1
        for row in reader: 
            if header:
                header = False
                continue
            ts = int(row[0])
            src = row[1]
            dst = row[2]

            #* filter low degree nodes
            if (src in low_deg_dict or dst in low_deg_dict):
                continue

            if (src not in node_dict):
                node_dict[src] = len(node_dict)
                node_type_dict[node_dict[src]] = 0
            if (dst not in node_dict):
                node_dict[dst] = len(node_dict)
                node_type_dict[node_dict[dst]] = 0

            subreddit = row[3]
            if (subreddit not in node_dict):
                node_dict[subreddit] = len(node_dict)
                node_type_dict[node_dict[subreddit]] = 1
       
            num_words = int(row[6])
            score = int(row[7])

            if (ts in out_dict):
                out_dict[ts][(node_dict[src], node_dict[dst], 0)] = (num_words, score)
                out_dict[ts][(node_dict[src], node_dict[subreddit], 1)] = (num_words, score)
            else:
                out_dict[ts] = {}
                out_dict[ts][(node_dict[src], node_dict[dst], 0)] = (num_words, score)
                out_dict[ts][(node_dict[src], node_dict[subreddit], 1)] = (num_words, score)
            num_lines += 1
    return out_dict, num_lines, node_dict, node_type_dict





def writeNodeType(node_type_dict, outname):
    r"""
    write the node type mapping to a file
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['node_id', 'type'])
        for key in node_type_dict:
            writer.writerow([key, node_type_dict[key]])


def write2csv(outname, out_dict):
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['ts', 'src', 'dst', 'relation_type', 'num_words', 'score'])
        ts_list = list(out_dict.keys())
        ts_list.sort()

        for ts in ts_list:
            for edge in out_dict[ts]:
                head = edge[0]
                tail = edge[1]
                relation_type = edge[2]
                num_words, score = out_dict[ts][edge]
                row = [ts, head, tail, relation_type, num_words, score]
                writer.writerow(row)


def writeNodeIDMapping(node_dict, outname):
    r"""
    write the node id mapping to a file
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['node_name', 'node_id'])
        for key in node_dict:
            writer.writerow([key, node_dict[key]])

def node_deg_filter(node_deg_dict):
    """
    filter out nodes with degree less than threshold
    """
    deg10_nodes = 0
    deg100_nodes = 0
    deg1000_nodes = 0

    for key in node_deg_dict:
        if (node_deg_dict[key] < 10):
            deg10_nodes += 1
        if (node_deg_dict[key] < 100):
            deg100_nodes += 1
        if (node_deg_dict[key] < 1000):
            deg1000_nodes += 1
    print ("nodes with degree less than 10: ", deg10_nodes)
    print ("nodes with degree less than 100: ", deg100_nodes)
    print ("nodes with degree less than 1000: ", deg1000_nodes)

def find_low_degree_nodes(node_deg_dict, threshold=10):
    """
    find nodes with degree less than threshold
    """
    low_degree_nodes = {}
    for key in node_deg_dict:
        if (node_deg_dict[key] < threshold):
            low_degree_nodes[key] = 1
    return low_degree_nodes


def main():
    fname = "reddit_edgelist.csv"
    _, _, _, _, _, node_deg_dict = load_csv_raw(fname)
    # print ("checking node degree")
    # node_deg_filter(node_deg_dict)
    # print ("checking reddit degree")
    # node_deg_filter(reddit_deg_dict)
    # low_degree_nodes = find_low_degree_nodes(node_deg_dict, threshold=100)
    # save_pkl(low_degree_nodes, 'low_degree_nodes.pkl')


    low_degree_nodes = load_pkl('low_degree_nodes.pkl')
    out_dict, num_lines, node_dict, node_type_dict = load_csv_filtered_node(fname, low_degree_nodes)
    writeNodeType(node_type_dict, 'thgl-forum_nodetype.csv')
    writeNodeIDMapping(node_dict, 'thgl-forum_nodeIDmapping.csv')
    write2csv('thgl-forum_edgelist.csv', out_dict)



if __name__ == "__main__":
    main()