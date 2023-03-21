import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Optional, Dict, Any, Tuple
from datetime import datetime


def get_genre_list(fname):
    """
    edge_id, user_id, timestamp, tags

    0,user_000001,2006-08-13 14:59:59+00:00,"['electronic', 0.5319148936170213]"
    0,user_000001,2006-08-13 14:59:59+00:00,"['alternative', 0.46808510638297873]"
    1,user_000001,2006-08-13 15:36:22+00:00,"['electronic', 0.6410256410256411]"
    1,user_000001,2006-08-13 15:36:22+00:00,"['chillout', 0.358974358974359]"
    2,user_000001,2006-08-13 15:40:13+00:00,"['math rock', 1.0]"
    3,user_000001,2006-08-15 13:41:18+00:00,"['electronica', 1.0]"
    4,user_000001,2006-08-15 13:59:27+00:00,"['acid jazz', 0.3546099290780142]"
    4,user_000001,2006-08-15 13:59:27+00:00,"['nu jazz', 0.3333333333333333]"
    4,user_000001,2006-08-15 13:59:27+00:00,"['chillout', 0.3120567375886525]"
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()
    genre_dict = {}
    for i in range(1,len(lines)):
        vals = lines[i].split(',')
        user_id = vals[1]
        time = vals[2]
        genre = vals[3].strip("\"").strip("['")
        #genre = vals[3]
        w = float(vals[4][:-3])
        if (genre not in genre_dict):
            genre_dict[genre] = 1
        else:
            genre_dict[genre] += 1

    
    # only keep genres that has shown up in more than 100 lines
    genre_list = []
    for key in genre_dict:
        genre_list.append([key])
        # if genre_dict[key] > 100:
        #     genre_list.append([key])
    fields = ['genre']

    with open('genre_list.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(genre_list)
    
    #check the distribution of genres        
    # print ("number of genres: " + str(len(genre_dict)))
    # freq = list(genre_dict.values())
    # freq = np.asarray(freq)
    # c100 = (freq > 100).sum()
    # print ("number of genres with frequency > 100: " + str(c100))
    # c1000 = (freq > 1000).sum()
    # print ("number of genres with frequency > 1000: " + str(c1000))
    # c10000 = (freq > 10000).sum()
    # print ("number of genres with frequency > 10000: " + str(c10000))

    #frequency diagram of genres
    # plt.title("genre distribution")
    # plt.xlabel("genre frequency")
    # plt.ylabel("number of genres")
    # #plt.yscale('log')
    # plt.xscale('log')
    # plt.hist(freq)
    # plt.savefig('genre_hist.pdf')


def load_genre_dict(
        fname: str,
        ) -> Dict[str, Any]:
    """
    reading the list of genres from genre_list.csv
    parameters:
        fname: file name of the genre list
    Returns:
        genre_dict: a dictionary of genres
    """
    genre_dict = {}
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            genre_dict[row[0]] = 1
    return genre_dict




def generate_daily_node_labels(fname: str):
    r"""
    read a temporal edgelist 
    node label = fav genre in this day
    generate the node label for each day for each user
    Note: only genres from the genre_list are considered

    0,user_000001,2006-08-13 14:59:59+00:00,"['electronic', 0.5319148936170213]"
    0,user_000001,2006-08-13 14:59:59+00:00,"['alternative', 0.46808510638297873]"
    1,user_000001,2006-08-13 15:36:22+00:00,"['electronic', 0.6410256410256411]"
    1,user_000001,2006-08-13 15:36:22+00:00,"['chillout', 0.358974358974359]"
    2,user_000001,2006-08-13 15:40:13+00:00,"['math rock', 1.0]"
    3,user_000001,2006-08-15 13:41:18+00:00,"['electronica', 1.0]"
    4,user_000001,2006-08-15 13:59:27+00:00,"['acid jazz', 0.3546099290780142]"
    4,user_000001,2006-08-15 13:59:27+00:00,"['nu jazz', 0.3333333333333333]"
    4,user_000001,2006-08-15 13:59:27+00:00,"['chillout', 0.3120567375886525]"
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    format = "%Y-%m-%d %H:%M:%S"
    day_dict = {} #store the weights of genres on this day
    cur_day = -1

    with open('daily_labels.csv', 'w') as outf:
        outf.write("year,month,day,user_id,fav_genre\n")
        #generate daily labels for users
        for i in range(1,len(lines)):
            vals = lines[i].split(',')
            user_id = vals[1]
            time = vals[2][:-7]
            date_object = datetime.strptime(time, format)
            if (i == 1):
                cur_day = date_object.day

            genre = vals[3].strip("\"").strip("['")
            w = float(vals[4][:-3])
            if (date_object.day != cur_day):
                #print (date_object.year, date_object.month, date_object.day)
                try:
                    fav_genre = max(day_dict, key=day_dict.get)
                except:
                    print ("error at line " + str(i))
                outf.write(str(date_object.year) + "," + str(date_object.month) +"," + str(date_object.day) + "," + user_id + "," + fav_genre + "\n")
                cur_day = date_object.day
                day_dict = {}

            if (genre not in day_dict):
                day_dict[genre] = w
            else:
                day_dict[genre] += w

            # if (genre not in genre_dict):
            #     print ("includes all genres now")
            #     print (genre)
            #     continue
            # else:
            #     if (date_object.day != cur_day):
            #         #print (date_object.year, date_object.month, date_object.day)
            #         try:
            #             fav_genre = max(day_dict, key=day_dict.get)
            #         except:
            #             print (i)
            #         outf.write(str(date_object.year) + "," + str(date_object.month) +"," + str(date_object.day) + "," + user_id + "," + fav_genre + "\n")
            #         cur_day = date_object.day
            #         day_dict = {}
            #     else:
            #         if (genre not in day_dict):
            #             day_dict[genre] = w
            #         else:
            #             day_dict[genre] += w

def load_node_labels(fname: str):
    """
    loading the node labels from the file
    year,month,day,user_id,fav_genre
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    fav_genre_dict = {}

    for i in range(1,len(lines)):
        vals = lines[i].split(',')
        year = int(vals[0])
        month = int(vals[1])
        day = int(vals[2])
        user_id = vals[3]
        fav_genre = vals[4]
        if fav_genre not in fav_genre_dict:
            fav_genre_dict[fav_genre] = 1
    
    print ("there are ", len(fav_genre_dict), "genres in total")




















if __name__ == "__main__":
    #get_genre_list("/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/lastfmGenre/dataset.csv")
    #genre_dict = load_genre_dict("/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/lastfmGenre/genre_list.csv")
    #generate_daily_node_labels("/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/lastfmGenre/dataset.csv")
    load_node_labels("/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/lastfmGenre/daily_labels.csv")