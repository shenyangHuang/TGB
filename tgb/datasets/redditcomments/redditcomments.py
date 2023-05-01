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

def read_edgelist(fname, 
                  outfname, write_header=False):
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
        if (write_header):
            fields = ['ts', 'src', 'dst', 'edge_id']
            write.writerow(fields)
        for line in lines:
            line = line.split()
            if (len(line) < 4):
                continue
            src = line[0]
            dst = line[1]
            ts = line[2]
            edge_id = line[3]
            write.writerow([ts, src, dst, edge_id])



def read_nodeattr(fname, 
                  outfname,
                  write_header=False):
    """
    read a space separated edgelist
    comment’s edge id, Reddit’s identifier of the comment, Reddit’s identifier of the parent (the post that the comment is replied to)
    Reddit’s identifier of the submission that the comment is in, name of the subreddit that the comment is in, number of characters in the comment’s body
    number of words in the comment’s body, score of the comment, a flag indicating if the comment has been edited


    edge_id, subreddit, num_characters, num_words, score, 'edited_flag'
    Parameters:
        fname (str): path to the edgelist
        outfname (str): path to the output file
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    with open(outfname, 'a') as outf:
        write = csv.writer(outf)
        if (write_header):
            fields = ['edge_id', 'subreddit', 'num_characters', 'num_words', 'score', 'edited_flag']
            write.writerow(fields)
        for line in lines:
            line = line.split()
            if (len(line) < 4):
                continue
            edge_id = line[0]
            subreddit = line[4]
            num_characters = line[5]
            num_words = line[6]
            score = line[7]        
            edited_flag = line[8].strip("/n")
            write.writerow([edge_id, subreddit, num_characters, num_words, score, edited_flag])


def combine_edgelist_edgefeat(edgefname, 
                              featfname, 
                              outname):
    """
    combine edgelist and edge features
    """
    total_lines = sum(1 for line in open(edgefname))

    with open(outname, 'w') as outf:
        write = csv.writer(outf)
        fields = ['ts', 'src', 'dst', 'subreddit', 'num_characters', 'num_words', 'score', 'edited_flag', 'edge_id']
        write.writerow(fields)
        line_idx = 0
        edgelist = open(edgefname, "r")
        edgefeat = open(featfname, "r")
        edgelist.readline()
        edgefeat.readline()

        while (line_idx < total_lines):
            #'ts', 'src', 'dst', 'edge_id'
            edge_line = edgelist.readline()
            edge_id = int(edge_line.split(",")[3])
            ts = int(edge_line.split(",")[0])
            src = int(edge_line.split(",")[1])
            dst = int(edge_line.split(",")[2])


            #'edge_id', 'subreddit', 'num_characters', 'num_words', 'score', 'edited_flag'
            feat_line = edgefeat.readline()
            edge_id_feat = int(feat_line.split(",")[0])
            subreddit = feat_line.split(",")[1]
            num_characters = int(feat_line.split(",")[2])
            num_words = int(feat_line.split(",")[3])
            score = int(feat_line.split(",")[4])
            edited_flag = bool(feat_line.split(",")[5])

            if (edge_id != edge_id_feat):
                print("edge_id != edge_id_feat")
                print(edge_id)
                print(edge_id_feat)
                break
            write.writerow([ts, src, dst, subreddit, num_characters, num_words, score, edited_flag, edge_id])






        





def main():
    # #! unzip all xz files by $ unxz *.xz

    # #f_dir = "raw/"
    # f_dir = "raw/raw_2005_2010/" #"raw/raw_2013_2014/"
    # fnames = find_filenames(f_dir)
    # outname = "redditcomments_edgelist_2005_2010.csv" #"redditcomments_edgelist_2013_2014.csv"
    # idx = 0
    # for fname in tqdm(fnames):
    #     if (idx == 0):
    #         read_edgelist(f_dir+fname, outname, write_header=True)
    #     else:
    #         read_edgelist(f_dir+fname, outname, write_header=False)
    #     idx += 1

    
    # # #! extract the node attributes
    # f_dir = "raw/node_2005_2010/"
    # fnames = find_filenames(f_dir)
    # outname = "redditcomments_edgefeat_2005_2010.csv"
    # idx = 0
    # for fname in tqdm(fnames):
    #     if (idx == 0):
    #         read_nodeattr(f_dir+fname, outname, write_header=True)
    #     else:
    #         read_nodeattr(f_dir+fname, outname, write_header=False)
    #     idx += 1

    #! combine edgelist and edge feat file check if the edge_id matches
    edgefname = "redditcomments_edgelist_2005_2010.csv"
    featfname = "redditcomments_edgefeat_2005_2010.csv"
    outname = "redditcomments_edgelist.csv"
    combine_edgelist_edgefeat(edgefname, featfname, outname)

    
    # #! analyze the extracted csv
    # fname = "redditcomments_edgelist_2005_2010.csv"
    # analyze_csv(fname)



if __name__ == "__main__":
    main()

