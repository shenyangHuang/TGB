import json
import networkx as nx 
import numpy as np
import csv
from datetime import date
import gzip



def load_full_json(fname):
    json_str = ""
    ctr = 0
    with open(fname, "r", encoding='utf-8') as f:

        #TODO need to determine how many lines form a json object 
        for line in f:
            # each line is json 
            '''
            {"_id":{"$oid":"63f3cd78f6d8723177e0edc1"},"json":{"created_at":{"$date":"2007-01-21T20:05:56Z"},"id":3603643,"id_str":"3603643","full_text":"Watching a scary climate change program on BBC 1. We're gonna get hot.","truncated":false,"display_text_range":[0,70],"entities":{"hashtags":[],"symbols":[],"user_mentions":[],"urls":[]},"source":"<a href=\"http://www.cloudhopper.com/\" rel=\"nofollow\">Twitter SMS</a>","in_reply_to_status_id":null,"in_reply_to_status_id_str":null,"in_reply_to_user_id":null,"in_reply_to_user_id_str":null,"in_reply_to_screen_name":null,"user":{"id":11674,"id_str":"11674","name":"Ben Mason","screen_name":"benmason","location":"London","description":"Entrepreneur. Marketing person. Founder @LoveTriPizza & @MasonsBeans | Elephant & Castle based.","url":"https://t.co/LOJz6o7rd4","entities":{"url":{"urls":[{"url":"https://t.co/LOJz6o7rd4","expanded_url":"http://lovetriangle.pizza","display_url":"lovetriangle.pizza","indices":[0,23]}]},"description":{"urls":[]}},"protected":false,"followers_count":1266,"friends_count":1480,"listed_count":89,"created_at":"Tue Nov 07 16:16:03 +0000 2006","favourites_count":3027,"utc_offset":null,"time_zone":null,"geo_enabled":true,"verified":false,"statuses_count":10734,"lang":null,"contributors_enabled":false,"is_translator":false,"is_translation_enabled":false,"profile_background_color":"050505","profile_background_image_url":"http://abs.twimg.com/images/themes/theme1/bg.png","profile_background_image_url_https":"https://abs.twimg.com/images/themes/theme1/bg.png","profile_background_tile":false,"profile_image_url":"http://pbs.twimg.com/profile_images/752844242540699648/pjtW8v7c_normal.jpg","profile_image_url_https":"https://pbs.twimg.com/profile_images/752844242540699648/pjtW8v7c_normal.jpg","profile_banner_url":"https://pbs.twimg.com/profile_banners/11674/1358506224","profile_link_color":"8CA1A6","profile_sidebar_border_color":"DAEAF1","profile_sidebar_fill_color":"DAEAF1","profile_text_color":"000000","profile_use_background_image":true,"has_extended_profile":false,"default_profile":false,"default_profile_image":false,"following":false,"follow_request_sent":false,"notifications":false,"translator_type":"none","withheld_in_countries":[]},"geo":null,"coordinates":null,"place":null,"contributors":null,"is_quote_status":false,"retweet_count":0,"favorite_count":0,"favorited":false,"retweeted":false,"lang":"en"},"tid":"3603643","uid":"11674"}
            '''
            data = json.loads(line)
            print (data)
            quit() #remove this when you write the code



# def load_gz_json(fname):
#     with gzip.open(fname, 'r') as fin:
#         data = json.loads(fin.read().decode('utf-8'))
#         print (data)
#         quit()


#https://stackoverflow.com/questions/39450065/python-3-read-write-compressed-json-objects-from-to-gzip-file
            

if __name__ == "__main__":
    """
    load from full json file
    """
    fname = "latest-all.json.gz"