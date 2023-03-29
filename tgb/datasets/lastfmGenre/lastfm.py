import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from datetime import date
from difflib import SequenceMatcher



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

    
    # TODO check the frequency of genres and threshold
    genre_list_10 = []
    genre_list_100 = []
    genre_list_1000 = []
    genre_list_2000 = []
    for key,freq in genre_dict.items():
        if (freq > 10):
            genre_list_10.append([key])
        if (freq > 100):
            genre_list_100.append([key])
        if (freq > 1000):
            genre_list_1000.append([key])
        if (freq > 2000):
            genre_list_2000.append([key])

    
    print ("number of genres with frequency > 10: " + str(len(genre_list_10)))
    print ("number of genres with frequency > 100: " + str(len(genre_list_100)))
    print ("number of genres with frequency > 1000: " + str(len(genre_list_1000)))
    print ("number of genres with frequency > 2000: " + str(len(genre_list_2000)))


    fields = ['genre']

    with open('genre_list_10.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(genre_list_10)
    
    with open('genre_list_100.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(genre_list_100)

    with open('genre_list_1000.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(genre_list_1000)
    
    with open('genre_list_2000.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(genre_list_2000)


def find_unique_genres(fname: str,
                       threshold: float = 0.8):
    """
    identify fuzzy strings which are actually the same genre, differences can be spacing, typo etc. 
    """
    #load all genre names into a list
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    genres = []
    sim_genres = {}
    for i in range(1,len(lines)):
        line = lines[i]
        genre = line.strip("\n")
        genres.append(genre)
    
    for i in range(len(genres)):
        for j in range(i+1,len(genres)):
            text = genres[i]
            search_key = genres[j]
            sim = SequenceMatcher(None, text, search_key)
            sim = sim.ratio()
            if (sim >= threshold):
                sim_genres[(text, search_key)] = sim

    print ("there are " + str(len(sim_genres)) + " similar genres")
    print (sim_genres)


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


def generate_weekly_labels(
                    fname: str,
                    days : int = 7,
                    ):
    """
    load daily node labels, generate weekly node labels
    if there is a tie, choose early genre
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()


    with open('weekly_avg_labels.csv', 'w') as outf:
        outf.write("year,month,day,user_id,fav_genre\n")
        for i in range(1,len(lines)):
            vals = lines[i].split(',')
            year = int(vals[0])
            month = int(vals[1])
            day = int(vals[2])
            user_id = vals[3]
            fav_genre = vals[4]
            fav_list = []

            #check next 7 lines to generate the label
            cur_date = date(year, month, day)
            if ((i + days) < len(lines)):
                for j in range(days):
                    vals = lines[i+j].split(',')
                    n_year = int(vals[0])
                    n_month = int(vals[1])
                    n_day = int(vals[2])
                    n_user_id = vals[3]
                    n_fav_genre = vals[4]
                    n_date = date(n_year, n_month, n_day)
                    diff = (n_date - cur_date).days
                    if (n_user_id != user_id):
                        break
                    if (diff <= days):
                        fav_list.append(n_fav_genre)
                    else:
                        break
            if (len(fav_list) < 1):
                print ("finished processing at line " + str(i))
                return None
            label = most_frequent(fav_list)
            outf.write(str(year) + "," + str(month) +"," + str(day) + "," + user_id + "," + label.strip("\n") + "\n")

            

            

        # date = date(year, month, day)
        # diff_days = (date - prev_date).days

        # #check if the duration has passed
        # if (diff_days > days):
        #     label = most_frequent(fav_list)
        #     prev_date = prev_date + datetime.timedelta(days=days)
        #     fav_list = []
        # fav_list.append(fav_genre)



def most_frequent(List):
    '''
    helper function to find the most frequent element in a list
    the ties are broken by choosing the earlier element
    '''
    counter = 0
    out = List[0]
     
    for item in List:
        curr_frequency = List.count(item)
        if(curr_frequency> counter):  #update on most frequent item is found
            counter = curr_frequency
            out = item
    return out







if __name__ == "__main__":

    #! generate the list of genres by frequency
    # get_genre_list("/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/lastfmGenre/dataset.csv")
    #genre_dict = load_genre_dict("/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/lastfmGenre/genre_list.csv")

    #! find similar genres 
    find_unique_genres("genre_list_1000.csv",
                       threshold= 0.8)

    #! generate the daily node labels
    #generate_daily_node_labels("/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/lastfmGenre/dataset.csv")
    #load_node_labels("/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/lastfmGenre/daily_labels.csv")

    #! generate the rolling weekly labels
    # fname = "/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/lastfmGenre/daily_labels.csv"
    # generate_weekly_labels(fname, days=7)

