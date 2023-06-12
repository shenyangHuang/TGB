import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Optional, Dict, Any, Tuple
import datetime
from datetime import date, timedelta
from difflib import SequenceMatcher


# similarity_dict = {('electronic', 'electronica'): 0.9523809523809523, ('electronic', 'electro'): 0.8235294117647058, ('alternative', 'alternative rock'): 0.8148148148148148, ('nu jazz', 'nu-jazz'): 0.8571428571428571,
#                    ('funky', 'funk'): 0.8888888888888888, ('funky', 'funny'): 0.8, ('post rock', 'pop rock'): 0.8235294117647058, ('post rock', 'post-rock'): 0.8888888888888888,
#                    ('instrumental', 'instrumental rock'): 0.8275862068965517, ('chill', 'chile'): 0.8, ('Drum and bass', 'Drum n Bass'): 0.8333333333333334, ('female vocalists', 'female vocalist'): 0.967741935483871,
#                    ('female vocalists', 'male vocalists'): 0.9333333333333333, ('female vocalists', 'male vocalist'): 0.896551724137931, ('electro', 'electropop'): 0.8235294117647058, ('funk', 'fun'): 0.8571428571428571,
#                    ('hip hop', 'trip hop'): 0.8, ('hip hop', 'hiphop'): 0.9230769230769231, ('trip-hop', 'trip hop'): 0.875, ('indie rock', 'indie folk'): 0.8, ('new age', 'new wave'): 0.8, ('new age', 'new rave'): 0.8,
#                    ('synthpop', 'synth pop'): 0.9411764705882353, ('industrial', 'industrial rock'): 0.8, ('cover', 'covers'): 0.9090909090909091, ('post hardcore', 'post-hardcore'): 0.9230769230769231, ('mathcore', 'deathcore'): 0.8235294117647058,
#                    ('deutsch', 'dutch'): 0.8333333333333334, ('swing', 'sting'): 0.8, ('female vocalist', 'male vocalists'): 0.896551724137931, ('female vocalist', 'male vocalist'): 0.9285714285714286, ('new wave', 'new rave'): 0.875,
#                    ('male vocalists', 'male vocalist'): 0.9629629629629629, ('Progressive rock', 'Progressive'): 0.8148148148148148, ('Alt-country', 'alt country'): 0.8181818181818182, ('favorites', 'Favourites'): 0.8421052631578947,
#                    ('favorites', 'favourite'): 0.8888888888888888, ('favorites', 'Favorite'): 0.8235294117647058, ('1970s', '1980s'): 0.8, ('1970s', '1990s'): 0.8, ('proto-punk', 'post-punk'): 0.8421052631578947,
#                    ('folk rock', 'folk-rock'): 0.8888888888888888, ('1980s', '1990s'): 0.8, ('favorite songs', 'Favourite Songs'): 0.8275862068965517, ('melancholic', 'melancholy'): 0.8571428571428571,
#                    ('Favourites', 'favourite'): 0.8421052631578947, ('Favourites', 'Favorite'): 0.8888888888888888, ('Favourites', 'Favourite Songs'): 0.8, ('favourite', 'Favorite'): 0.8235294117647058,
#                    ('american', 'americana'): 0.9411764705882353, ('american', 'african'): 0.8, ('american', 'mexican'): 0.8, ('rock en español', 'Rock en Espanol'): 0.8, ('trance', 'psytrance'): 0.8,
#                    ('power pop', 'powerpop'): 0.9411764705882353, ('psychill', 'psychobilly'): 0.8421052631578947, ('Progressive metal', 'progressive death metal'): 0.8, ('Progressive metal', 'progressive black metal'): 0.8,
#                    ('progressive death metal', 'progressive black metal'): 0.8260869565217391, ('romantic', 'new romantic'): 0.8, ('hair metal', 'Dark metal'): 0.8, ('melodic metal', 'melodic black metal'): 0.8125,
#                    ('funk metal', 'folk metal'): 0.8, ('death metal', 'math metal'): 0.8571428571428571, ('Technical Metal', 'Technical Death Metal'): 0.8333333333333334, ('speed metal', 'sid metal'): 0.8}

#! map diferent spelling and similar ones to the same one, use space if possible
# ? key = to replace, value = to keep

similarity_dict = {
    "nu-jazz": "nu jazz",
    "funky": "funk",
    "post-rock": "post rock",
    "Drum n Bass": "Drum and bass",
    "female vocalists": "female vocalist",
    "male vocalists": "male vocalist",
    "hiphop": "hip hop",
    "trip-hop": "trip hop",
    "synthpop": "synth pop",
    "covers": "cover",
    "post-hardcore": "post hardcore",
    "Favourites": "favorites",
    "favourite": "favorites",
    "Favorite": "favorites",
    "folk-rock": "folk rock",
    "favorite songs": "favorites",
    "Favourite Songs": "favorites",
    "americana": "american",
    "Rock en Espanol": "rock en español",
    "melancholy": "melancholic",
    "powerpop": "power pop",
}


def filter_genre_edgelist(fname, genres_dict):
    """
    rewrite the edgelist but only keeping the genres with high frequency, also uses similarity_dict
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    with open("lastfm_edgelist_clean.csv", "w") as f:
        write = csv.writer(f)
        fields = ["user_id", "timestamp", "tags", "weight"]
        write.writerow(fields)

        for i in range(1, len(lines)):
            vals = lines[i].split(",")
            user_id = vals[1]
            time = vals[2]
            genre = vals[3].strip('"').strip("['")
            w = vals[4][:-3]
            if genre in genres_dict:
                if genre in similarity_dict:
                    genre = similarity_dict[genre]
                write.writerow([user_id, time, genre, w])


def get_genre_list(fname):
    """
    edge_id, user_id, timestamp, tags

    0,user_000001,2006-08-13 14:59:59+00:00,"['electronic', 0.5319148936170213]"
    0,user_000001,2006-08-13 14:59:59+00:00,"['alternative', 0.46808510638297873]"
    1,user_000001,2006-08-13 15:36:22+00:00,"['electronic', 0.6410256410256411]"
    1,user_000001,2006-08-13 15:36:22+00:00,"['chillout', 0.358974358974359]"
    2,user_000001,2006-08-13 15:40:13+00:00,"['math rock', 1.0]"
    3,user_000001,2006-08-15 13:41:18+00:00,"['electronica', 1.0]"
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()
    genre_dict = {}
    for i in range(1, len(lines)):
        vals = lines[i].split(",")
        user_id = vals[1]
        time = vals[2]
        genre = vals[3].strip('"').strip("['")
        # genre = vals[3]
        w = float(vals[4][:-3])
        if genre not in genre_dict:
            genre_dict[genre] = 1
        else:
            genre_dict[genre] += 1

    # TODO check the frequency of genres and threshold
    genre_list_10 = []
    genre_list_100 = []
    genre_list_1000 = []
    genre_list_2000 = []
    for key, freq in genre_dict.items():
        if freq > 10:
            genre_list_10.append([key])
        if freq > 100:
            genre_list_100.append([key])
        if freq > 1000:
            genre_list_1000.append([key])
        if freq > 2000:
            genre_list_2000.append([key])
    print("number of genres with frequency > 10: " + str(len(genre_list_10)))
    print("number of genres with frequency > 100: " + str(len(genre_list_100)))
    print("number of genres with frequency > 1000: " + str(len(genre_list_1000)))
    print("number of genres with frequency > 2000: " + str(len(genre_list_2000)))
    fields = ["genre"]

    with open("genre_list_1000.csv", "w") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(genre_list_1000)


def find_unique_genres(fname: str, threshold: float = 0.8):
    """
    identify fuzzy strings which are actually the same genre, differences can be spacing, typo etc.
    """
    # load all genre names into a list
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    genres = []
    sim_genres = {}
    for i in range(1, len(lines)):
        line = lines[i]
        genre = line.strip("\n")
        genres.append(genre)

    for i in range(len(genres)):
        for j in range(i + 1, len(genres)):
            text = genres[i]
            search_key = genres[j]
            sim = SequenceMatcher(None, text, search_key)
            sim = sim.ratio()
            if sim >= threshold:
                sim_genres[(text, search_key)] = sim

    print("there are " + str(len(sim_genres)) + " similar genres")
    print(sim_genres)


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
    with open(fname, "r") as f:
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

    user_000001,2006-08-13 14:59:59+00:00,"['electronic', 0.5319148936170213]"
    user_000001,2006-08-13 14:59:59+00:00,"['alternative', 0.46808510638297873]"
    user_000001,2006-08-13 15:36:22+00:00,"['electronic', 0.6410256410256411]"
    user_000001,2006-08-13 15:36:22+00:00,"['chillout', 0.358974358974359]"
    user_000001,2006-08-13 15:40:13+00:00,"['math rock', 1.0]"
    user_000001,2006-08-15 13:41:18+00:00,"['electronica', 1.0]"
    user_000001,2006-08-15 13:59:27+00:00,"['acid jazz', 0.3546099290780142]"
    user_000001,2006-08-15 13:59:27+00:00,"['nu jazz', 0.3333333333333333]"
    user_000001,2006-08-15 13:59:27+00:00,"['chillout', 0.3120567375886525]"
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    format = "%Y-%m-%d %H:%M:%S"
    day_dict = {}  # store the weights of genres on this day
    cur_day = -1

    with open("daily_labels.csv", "w") as outf:
        write = csv.writer(outf)
        fields = ["user_id", "year", "month", "day", "genre", "weight"]
        write.writerow(fields)

        # generate daily labels for users
        for i in range(1, len(lines)):
            vals = lines[i].split(",")
            user_id = vals[0]
            time = vals[1][:-7]
            date_object = datetime.datetime.strptime(time, format)
            if i == 1:
                cur_day = date_object.day

            genre = vals[2]
            w = float(vals[3].strip())
            if date_object.day != cur_day:
                #! normalize the weights in the day_dict to sum 1
                # * remove normalization for future aggregation
                # total = sum(day_dict.values())
                # day_dict = {k: v / total for k, v in day_dict.items()}

                #! user,time,genre,weight  # genres = # of weights
                out = [
                    user_id,
                    str(date_object.year),
                    str(date_object.month),
                    str(date_object.day),
                ]
                for genre, w in day_dict.items():
                    write.writerow(out + [genre] + [w])

                cur_day = date_object.day
                day_dict = {}
            else:
                if genre not in day_dict:
                    day_dict[genre] = w
                else:
                    day_dict[genre] += w


def generate_aggregate_labels(fname: str, days: int = 7):
    """
    aggregate the genres over a number of days,  as specified by days
    #! current generation includes edges from the day of the label, thus the label should be set to be beginning of the day
    prediction should always be at the first second of the day
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()
    date_prev = 0

    genre_dict = {}
    user_prev = 0

    # "user_id", "year", "month", "day", "genre", "weight"
    with open(str(days) + "days_labels.csv", "w") as outf:
        write = csv.writer(outf)
        fields = ["user_id", "year", "month", "day", "genre", "weight"]
        write.writerow(fields)

        for i in range(1, len(lines)):
            vals = lines[i].split(",")
            user_id = vals[0]
            year = int(vals[1])
            month = int(vals[2])
            day = int(vals[3])
            genre = vals[4]
            w = float(vals[5])
            if i == 1:
                date_prev = date(year, month, day)
                user_prev = user_id

            date_cur = date(year, month, day)

            if user_id != user_prev:
                date_prev = date(year, month, day)
                user_prev = user_id

            if (
                date_cur - date_prev
            ).days <= days:  #! this means that the date = [0,7] which includes the current day
                if genre not in genre_dict:
                    genre_dict[genre] = w
                else:
                    genre_dict[genre] += w
            else:
                # start a new week
                # normalize the weight to sum 1
                total = sum(genre_dict.values())
                genre_dict = {k: v / total for k, v in genre_dict.items()}

                out = [
                    user_id,
                    str(date_prev.year),
                    str(date_prev.month),
                    str(date_prev.day),
                ]
                for genre, w in genre_dict.items():
                    write.writerow(out + [genre] + [w])
                date_prev = date_prev + datetime.timedelta(days=1)
                genre_dict = {}


def most_frequent(List):
    """
    helper function to find the most frequent element in a list
    the ties are broken by choosing the earlier element
    """
    counter = 0
    out = List[0]

    for item in List:
        curr_frequency = List.count(item)
        if curr_frequency > counter:  # update on most frequent item is found
            counter = curr_frequency
            out = item
    return out


def convert_ts_unix(fname: str, outname: str):
    """
    convert all time from datetime to unix time
    """
    TIME_FORMAT = "%Y-%m-%d"
    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["ts", "user_id", "genre", "weight"]
        write.writerow(fields)

        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            # time,user_id,genre,weight
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    ts = datetime.datetime.strptime(row[0], TIME_FORMAT)
                    ts += timedelta(days=1)
                    ts = int(ts.timestamp())
                    user_id = row[1]
                    genre = row[2]
                    weight = float(row[3])
                    write.writerow([ts, user_id, genre, weight])


def convert_ts_edgelist(fname: str, outname: str):
    """
    convert all time from datetime to unix time
    """
    TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["ts", "user_id", "genre", "weight"]
        write.writerow(fields)

        with open(fname, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            # time,user_id,genre,weight
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    ts = datetime.datetime.strptime(row[0], TIME_FORMAT)
                    ts = int(ts.timestamp())
                    user_id = row[1]
                    genre = row[2]
                    weight = float(row[3])
                    write.writerow([ts, user_id, genre, weight])


def sort_node_labels(fname, outname):
    r"""
    sort the node labels by time
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["time", "user_id", "genre", "weight"]
        write.writerow(fields)
        rows_dict = {}

        for i in range(1, len(lines)):
            vals = lines[i].split(",")
            user_id = vals[0]
            year = int(vals[1])
            month = int(vals[2])
            day = int(vals[3])
            genre = vals[4]
            w = float(vals[5])
            date_cur = datetime(year, month, day)
            time_ts = date_cur.strftime("%Y-%m-%d")
            if time_ts not in rows_dict:
                rows_dict[time_ts] = [(user_id, genre, w)]
            else:
                rows_dict[time_ts].append((user_id, genre, w))

        time_keys = list(rows_dict.keys())
        time_keys.sort()

        for ts in time_keys:
            rows = rows_dict[ts]
            for user_id, genre, w in rows:
                write.writerow([ts, user_id, genre, w])


def sort_edgelist(fname, outname="sorted_lastfm_edgelist.csv"):
    r"""
    sort the edgelist by time
    """
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    with open(outname, "w") as outf:
        write = csv.writer(outf)
        fields = ["time", "user_id", "genre", "weight"]
        write.writerow(fields)

        rows_dict = {}
        for idx in range(1, len(lines)):
            vals = lines[idx].split(",")
            user_id = vals[0]
            time_ts = vals[1][:-7]
            genre = vals[2]
            w = float(vals[3].strip())
            if time_ts not in rows_dict:
                rows_dict[time_ts] = [(user_id, genre, w)]
            else:
                rows_dict[time_ts].append((user_id, genre, w))

        time_keys = list(rows_dict.keys())
        time_keys.sort()

        for ts in time_keys:
            rows = rows_dict[ts]
            for user_id, genre, w in rows:
                write.writerow([ts, user_id, genre, w])


if __name__ == "__main__":
    #! generate the list of genres by frequency
    # get_genre_list("/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/lastfmGenre/dataset.csv")
    # genre_dict = load_genre_dict("/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/lastfmGenre/genre_list.csv")

    #! find similar genres
    # find_unique_genres("genre_list_1000.csv",threshold= 0.8)

    #! filter edgelist with genres to keep
    # genres_dict = load_genre_dict("genre_list_1000.csv")
    # filter_genre_edgelist("dataset.csv", genres_dict)

    #! generate the daily node labels
    # generate_daily_node_labels("lastfm_edgelist_clean.csv")

    # generate_daily_node_labels("/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/lastfmGenre/dataset.csv")
    # load_node_labels("/mnt/c/Users/sheny/Desktop/TGB/tgb/datasets/lastfmGenre/daily_labels.csv")

    # #! generate normalized weekly node labels
    # generate_aggregate_labels("daily_labels.csv", days=7)\

    # """
    # sort edgelist by time for lastfm dataset
    # """
    # fname = "../datasets/lastfmGenre/lastfm_edgelist_clean.csv"
    # outname = '../datasets/lastfmGenre/sorted_lastfm_edgelist.csv'
    # sort_edgelist(fname,
    #               outname = outname)

    # """
    # sort node labels by time for lastfm dataset
    # """
    # fname = "../datasets/lastfmGenre/7days_labels.csv"
    # outname = '../datasets/lastfmGenre/sorted_7days_node_labels.csv'
    # sort_node_labels(fname,
    #                  outname)

    # #! convert from date to ts
    # convert_ts_unix("lastfmgenre_node_labels_datetime.csv",
    #                 "lastfmgenre_node_labels.csv")
    convert_ts_edgelist("lastfmgenre_edgelist.csv", "lastfmgenre_edgelist_ts.csv")
