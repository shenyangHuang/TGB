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


def combine_edgelist_edgefeat2subreddits(edgefname, 
                              featfname, 
                              outname):
    """
    combine edgelist and edge features
    'ts', 'src', 'subreddit', 'num_words', 'score'
    """
    line_idx = 0

    with open(outname, 'w') as outf:
        write = csv.writer(outf)
        fields = ['ts', 'src', 'subreddit', 'num_words', 'score']
        write.writerow(fields)
        sub_id = 0
        edgelist = open(edgefname, "r")
        edgefeat = open(featfname, "r")
        edgelist.readline()
        edgefeat.readline()

        while (True):
            #'ts', 'src', 'dst', 'edge_id'
            edge_line = edgelist.readline()
            edge_line = edge_line.split(",")
            if (len(edge_line) < 4):
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
            

            if (edge_id != edge_id_feat):
                print("edge_id != edge_id_feat")
                print(edge_id)
                print(edge_id_feat)
                break
            
            write.writerow([ts, src, subreddit, num_words, score])
            line_idx += 1
    print ("processed", line_idx, "lines")


def filter_subreddits(fname):
    """
    check the frequency of subreddits
    """
    subreddit_count = {}
    node_count = {}
    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        #ts, src, subreddit, num_words, score
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                ts = row[0]
                src = row[1]
                subreddit = row[2]
                if (subreddit not in subreddit_count):
                    subreddit_count[subreddit] = 1
                else:
                    subreddit_count[subreddit] += 1
                if (src not in node_count):
                    node_count[src] = 1
                else:
                    node_count[src] += 1
    return subreddit_count, node_count


def clean_edgelist(fname, 
                   node_counts, 
                   outname, 
                   threshold=1000):
    """
    helper function for filtering out low frequency nodes
    """
    node_dict = {}
    
    for node in node_counts:
        if (node_counts[node] >= threshold):
            node_dict[node] = 1
            
            
    with open(outname, 'w') as outf:
        write = csv.writer(outf)
        fields = ['ts', 'user', 'subreddit', 'num_words', 'score']
        write.writerow(fields)
            
        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            #ts, src, subreddit, num_words, score
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    ts = row[0]
                    src = row[1]
                    subreddit = row[2]
                    num_words = int(row[3])
                    score = int(row[4])
                    if (src in node_dict):
                        write.writerow([ts, src, subreddit, num_words, score])


def clean_edgelist_reddits(
                    fname, 
                   reddit_counts, 
                   outname, 
                   threshold=50
                    ):
    """
    helper function for filtering out low frequency subreddits
    """
    reddit_dict = {}
    
    for reddit in reddit_counts:
        if (reddit_counts[reddit] >= threshold):
            reddit_dict[reddit] = 1
            
    with open(outname, 'w') as outf:
        write = csv.writer(outf)
        fields = ['ts', 'user', 'subreddit', 'num_words', 'score']
        write.writerow(fields)
        with open(fname, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                #ts, src, subreddit, num_words, score
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                    else:
                        ts = row[0]
                        src = row[1]
                        subreddit = row[2]
                        num_words = int(row[3])
                        score = int(row[4])
                        if (subreddit in reddit_dict):
                            write.writerow([ts, src, subreddit, num_words, score])
            
            
        
    








        





def main():
    # #? see redditcomments.py for the extraction from the raw files

    # #! combine edgelist and edge feat file check if the edge_id matches
    # edgefname = "redditcomments_edgelist_2005_2010.csv"
    # featfname = "redditcomments_edgefeat_2005_2010.csv"
    # outname = "subreddits_edgelist.csv"
    # combine_edgelist_edgefeat2subreddits(edgefname, featfname, outname)
    
    
    # #! frequency count of nodes
    # fname = "subreddits_edgelist.csv"
    # outname = "subreddits_edgelist_clean.csv"
    # subreddit_count, node_count = filter_subreddits(fname)
    # threshold = 1000
    # clean_edgelist(fname, node_count, outname, threshold=threshold)
    # print ("finish cleaning")
    
    
    #! frequency count of reddits
    fname = "subreddits_edgelist_clean.csv"
    outname = "subreddits_edgelist_clean_reddit.csv"
    subreddit_count, node_count = filter_subreddits(fname)
    clean_edgelist_reddits(fname, subreddit_count, outname, threshold=50)
    
    #! analyze the extracted csv
    # fname = "subreddits_edgelist_clean_reddit.csv" #"subreddits_edgelist_clean.csv"
    # analyze_csv(fname)
    # sub_10 = 0
    # sub_50 = 0 
    # sub_100 = 0
    # sub_1000 = 0
    
    # for sub in subreddit_count:
    #     if (subreddit_count[sub] >= 10):
    #         sub_10 += 1
    #     if (subreddit_count[sub] >= 50):
    #         sub_50 += 1
    #     if (subreddit_count[sub] >= 100):
    #         sub_100 += 1
    #     if (subreddit_count[sub] >= 1000):
    #         sub_1000 += 1
    # print ("subreddit count:", len(subreddit_count))
    # print ("subreddit >= 10:", sub_10)
    # print ("subreddit >= 50:", sub_50)
    # print ("subreddit >= 100:", sub_100)
    # print ("subreddit >= 1000:", sub_1000)
    



if __name__ == "__main__":
    main()

