
import csv
from tqdm import tqdm
from os import listdir
import glob

def find_filenames(path_to_dir):
    r"""
    find all files in a folder
    Parameters:
        path_to_dir (str): path to the directory
    """
    # filenames = glob.glob(path_to_dir)
    filenames = listdir(path_to_dir)
    return filenames

def read_edgelist(fname, outfname, write_header=False):
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

    with open(outfname, "a") as outf:
        write = csv.writer(outf)
        if write_header:
            fields = ["ts", "src", "dst", "edge_id"]
            write.writerow(fields)
        for line in lines:
            line = line.split()
            if len(line) < 4:
                continue
            src = line[0]
            dst = line[1]
            ts = line[2]
            edge_id = line[3]
            write.writerow([ts, src, dst, edge_id])


def read_nodeattr(fname, outfname, write_header=False):
    """
    read a space separated edgelist
    comment’s edge id, Reddit’s identifier of the comment, Reddit’s identifier of the parent (the post that the comment is replied to)
    Reddit’s identifier of the submission that the comment is in, name of the subreddit that the comment is in, number of characters in the comment’s body
    number of words in the comment’s body, score of the comment, a flag indicating if the comment has been edited

    - comment’s edge id
    - Reddit’s identifier of the comment
    - Reddit’s identifier of the parent (the post that the comment is replied to)
    - Reddit’s identifier of the submission that the comment is in
    - name of the subreddit that the comment is in
    - number of characters in the comment’s body
    - number of words in the comment’s body
    - score of the comment
    - a flag indicating if the comment has been edited



    edge_id, subreddit, num_characters, num_words, score, 'edited_flag'
    Parameters:
        fname (str): path to the edgelist
        outfname (str): path to the output file
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    max_words = 0
    max_score = 0
    min_score = 100000000

    with open(outfname, "a") as outf:
        write = csv.writer(outf)
        if write_header:
            fields = [
                "edge_id",
                "reddit_id",
                "reddit_parent_id",
                "subreddit",
                "num_characters",
                "num_words",
                "score",
                "edited_flag",
            ]
            write.writerow(fields)
        for line in lines:
            line = line.split()
            if len(line) < 4:
                continue
            edge_id = line[0]
            reddit_id = line[1]
            reddit_parent_id = line[2]
            subreddit = line[4]
            num_characters = line[5]
            num_words = line[6]
            if (int(num_words) > max_words):
                max_words = int(num_words)
            score = line[7]
            if (int(score) < min_score):
                min_score = int(score)
            edited_flag = line[8].strip("/n")
            write.writerow(
                [edge_id, reddit_id, reddit_parent_id, subreddit, num_characters, num_words, score, edited_flag]
            )
    print("max # words", max_words)
    print("min score", min_score)
    print("max score", max_score)


def combine_edgelist_edgefeat(edgefname, featfname, outname):
    """
    combine edgelist and edge features
    """
    total_lines = sum(1 for line in open(edgefname))
    subreddit_ids = {}

    missing_ts = 0
    missing_src = 0
    missing_dst = 0
    line_idx = 0

    with open(outname, "w") as outf:
        write = csv.writer(outf)
        #fields = ["ts", "src", "dst", "subreddit", "num_words", "score"]
        fields = ["ts", "src", "dst", "reddit_id", "reddit_parent_id", "subreddit", "num_words", "score"]
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
            dst = int(edge_line[2])

            # "edge_id", "reddit_id", "reddit_parent_id", "subreddit", "num_characters", "num_words", "score", "edited_flag",
            feat_line = edgefeat.readline()
            feat_line = feat_line.split(",")
            edge_id_feat = int(feat_line[0])
            reddit_id = feat_line[1]
            reddit_parent_id = feat_line[2]
            subreddit = feat_line[3]
            num_characters = int(feat_line[4])
            num_words = int(feat_line[5])
            score = int(feat_line[6])
            edited_flag = bool(feat_line[7])

            #! check if ts, src, dst is -1
            if ts == -1:
                missing_ts += 1
                continue
            if src == -1:
                missing_src += 1
                continue
            if dst == -1:
                missing_dst += 1
                continue

            if edge_id != edge_id_feat:
                print("edge_id != edge_id_feat")
                print(edge_id)
                print(edge_id_feat)
                break

            # write.writerow([ts, src, dst, subreddit, num_words, score])
            #write.writerow([ts, src, dst, num_words, score])
            #? ts: int, src: int (user_id), dst: int (user_id), subreddit: str, reddit_id: str (comment_id), reddit_parent_id: str (post_id), num_words: int, score: int
            write.writerow([ts, src, dst, subreddit, reddit_id, reddit_parent_id, num_words, score])
            line_idx += 1
    print("processed", line_idx, "lines")
    print ("there are lines", missing_ts, " missing timestamps")
    print ("there are lines", missing_src, " missing src")
    print ("there are lines", missing_dst, " missing dst")



def main():
    # #! unzip all xz files by $ unxz *.xz

    f_dir = "edge_files/" #"raw/raw_2005_2010/" #"raw/raw_2013_2014/"
    fnames = find_filenames(f_dir)
    edgefname = "reddit_edgefile_2014_01.csv" #"redditcomments_edgelist_2013_2014.csv"
    idx = 0
    for fname in tqdm(fnames):
        print ("processing, ", fname)
        if (idx == 0):
            read_edgelist(f_dir+fname, edgefname, write_header=True)
        else:
            read_edgelist(f_dir+fname, edgefname, write_header=False)
        idx += 1

    
    # # #! extract the node attributes
    f_dir = "attribute_files/" 
    fnames = find_filenames(f_dir)
    featfname = "reddit_attribute_2014_01.csv"
    idx = 0
    for fname in tqdm(fnames):
        print ("processing, ", fname)
        if (idx == 0):
            read_nodeattr(f_dir+fname, featfname, write_header=True)
        else:
            read_nodeattr(f_dir+fname, featfname, write_header=False)
        idx += 1

    #! combine edgelist and edge feat file check if the edge_id matches
    # edgefname = "reddit_edgefile_2019_01_03.csv"
    # featfname = "reddit_attribute_2019_01_03.csv"
    outname = "reddit_edgelist.csv"
    combine_edgelist_edgefeat(edgefname, featfname, outname)


if __name__ == "__main__":
    main()
