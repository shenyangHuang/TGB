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


def filter_subreddits(fname,
                      outname):
    """
    check the frequency of subreddits
    """
    print ("hi")
    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        #callsign,number,icao24,registration,typecode,origin,destination,firstseen,lastseen,day,latitude_1,longitude_1,altitude_1,latitude_2,longitude_2,altitude_2
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                row[1]








        





def main():
    #? see redditcomments.py for the extraction from the raw files

    #! combine edgelist and edge feat file check if the edge_id matches
    edgefname = "redditcomments_edgelist_2005_2010.csv"
    featfname = "redditcomments_edgefeat_2005_2010.csv"
    outname = "subreddits_edgelist.csv"
    combine_edgelist_edgefeat2subreddits(edgefname, featfname, outname)

    
    #! analyze the extracted csv
    # fname = "subreddits_edgelist.csv"
    # analyze_csv(fname)



if __name__ == "__main__":
    main()

