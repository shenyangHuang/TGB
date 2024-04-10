import dateutil.parser as dparser
import csv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from os import listdir
from datetime import datetime



def date2ts(date_str: str) -> float:
    r"""
    convert date string to timestamp
    """
    TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
    date_cur = datetime.strptime(date_str, TIME_FORMAT)
    return int(date_cur.timestamp())


"""
app_name	user_id	datetime	is_update
com.cocoplay.erpetvet	392863962	2020-06-17 23:55:17.460	0
com.titan.royal	790103760	2020-06-17 23:55:19.583	0
com.tencent.ig	-1651723014	2020-06-17 23:55:20.647	0
com.cyberlink.youperfect	-2116095669	2020-06-17 23:55:20.723	1
com.whatsapp	1591275459	2020-06-17 23:55:20.820	0
com.nexttechgamesstudio.house.paint.craft.coloring.book.pages	-984956295	2020-06-17 23:55:21.840	0
com.lenovo.anyshare.gps	1643649087	2020-06-17 23:55:21.853	1
com.kurankarim.mp3	1316745267	2020-06-17 23:55:22.537	0
com.google.android.dialer	239675079	2020-06-17 23:55:22.950	1
com.ma.textgraphy	-951808761	2020-06-17 23:55:22.977	0
ir.shahbaz.SHZToolBox	1643649087	2020-06-17 23:55:22.987	1
picture.instagram.makers	-1898448882	2020-06-17 23:55:23.010	0
ir.shahbaz.SHZToolBox	780669111	2020-06-17 23:55:23.600	1
fantasy.survival.game.rpg	1849120437	2020-06-17 23:55:23.980	0
com.ags.flying.muscle.car.transform.robot.war.robot.games	1751574033	2020-06-17 23:55:24.680	0
"""
def read_csv2dict(fname):
    r"""
    load from the raw data and retrieve, timestamp, head, tail, relation 
    also return a mapping from node text to node id
    convert all dates into unix timestamps
    """
    out_dict = {}
    first_row = True
    num_lines = 0
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter ='\t')
        for row in reader: 
            if first_row:
                first_row = False
                continue
            app = row[0]
            user = row[1]
            date = row[2]
            is_update = int(row[3])
            if (len(date) == 0 or date is None):
                continue
            else:
                ts = date2ts(date)
                head = user
                tail = app
                if (ts not in out_dict):
                    out_dict[ts] = {(head,tail,is_update): 1}
                else:
                    out_dict[ts][(head,tail,is_update)] = 1
                num_lines += 1
    print ("there are {} lines in the file".format(num_lines))
    return out_dict


def writeIDmapping(id_dict, outname):
    r"""
    write the id mapping to a file
    """
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['ID', 'name'])
        for key in id_dict:
            writer.writerow([key, id_dict[key]])


def write2edgelist(out_dict, outname):
    r"""
    Write the dictionary to a csv file
    """
    num_lines = 0
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['timestamp', 'head', 'tail', 'relation_type'])
        dates = list(out_dict.keys())
        dates.sort()
        for date in dates:
            for edge in out_dict[date]:
                head = edge[0]
                tail = edge[1]
                relation_type = int(edge[2])
                row = [date, head, tail, relation_type]
                writer.writerow(row)
                num_lines += 1
    print ("there are {} lines in the file".format(num_lines))
        



def main():
    fname = "raw_myket_input-001.csv"
    out_dict = read_csv2dict(fname)
    write2edgelist (out_dict, "thgl-myket_edgelist.csv")




if __name__ == "__main__":
    main()