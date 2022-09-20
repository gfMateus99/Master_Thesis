# -*- coding: utf-8 -*-
"""
Program to collect tweets 

Based on https://towardsdatascience.com/an-extensive-guide-to-collecting-tweets-from-twitter-api-v2-for-academic-research-using-python-3-518fcb71df2a

@author: Gonçalo Mateus
"""

#%% Import and define tokens 

import pandas as pd
import requests
import os
import csv
import json
import time
import numpy as np
import datetime
from datetime import date
from datetime import datetime
import re

#%%
#Token with elevated access
#os.environ['BEARER_TOKEN'] = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

#Token with Academic access
os.environ['BEARER_TOKEN'] = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

#%% Create a connection to twitter API

def connect_to_twitter():
    bearer_token = os.environ.get("BEARER_TOKEN")
    return {"Authorization": "Bearer {}".format(bearer_token)}
headers = connect_to_twitter()


#%% Endpoints to make a request to Twitter API

# Twitter endpoints:
# https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-recent

# Search by specific media:
# query_params = {'query': 'from:' + from_name,

# Search by topic:
# query_params = {'query': '("Novo banco" -is:retweet) OR (novobanco -is:retweet)',

def make_request(headers, from_name, next_page, number_of_results, start_date, end_time):
    url = "https://api.twitter.com/2/tweets/search/all"
    query_params = {'query': 'from:' + from_name,
    'max_results': number_of_results,
    "start_time": start_date,
    "end_time": end_time,
    'expansions': 'attachments.poll_ids,attachments.media_keys,entities.mentions.username,referenced_tweets.id,referenced_tweets.id.author_id',
    'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,public_metrics,possibly_sensitive,referenced_tweets,reply_settings,source,text,withheld',
    'user.fields': 'created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified,withheld',
    'media.fields': 'duration_ms,height,media_key,preview_image_url,type,url,width,public_metrics,non_public_metrics,organic_metrics,promoted_metrics,alt_text',
    'poll.fields': 'duration_minutes,end_datetime,id,options,voting_status',
    'place.fields': 'contained_within,country,country_code,full_name,geo,id,name,place_type',
    'next_token': next_page}
    return requests.request("GET", url, params=query_params, headers=headers)


#%% Function to help in loop to get tweets

def getTweets(headers, from_name, next_page, number_of_results, start_date, count, allTweets, end_time):
    while count:
        response = make_request(headers, from_name, next_page, number_of_results, start_date, end_time)
        
        time.sleep(2) #Just to delay requests because of the limits
   
        #Make sure the request was well done
        print("(Remmaining: " + str(int(response.headers['x-rate-limit-remaining'])) + ") Endpoint Response Code: " + str(response.status_code))
        if response.status_code != 200:
            print(response.text)
            print("Next page code:" + next_page)
            if response.status_code == 429:
                print(datetime.datetime.fromtimestamp(int(response.headers['x-rate-limit-reset'])))
                return next_page
                break
            
        
        response = response.json() #Put in Json format
        allTweets += response['data'] #Concatenate the tweets from the requests

        if 'next_token' not in response['meta']:
            count = False #When there are no more tweets to return
        else:
            next_page = response['meta']['next_token'] #To retrieve the next tweets in the next request


#%% Loop to get all tweets with the info provided

def makeTweetCollect(headers, from_name, number_of_results, start_date, name_files_saved, allDataTogether, end_time):
    optimize = 0
    allTweets = []
    count = True
    next_page = {}
        
    while count:
        next_page = getTweets(headers, from_name, next_page, number_of_results, start_date, count, allTweets, end_time);
    
        allDataTogether += allTweets
    
        print(next_page == None)
    
        #Save tweets in Json file
        #with open(from_name + str(optimize)+'.json', 'w') as outfile:
        #    json.dump(allTweets, outfile)
            
        #name_files_saved.append(from_name + str(optimize)+'.json')
        
        print("----------------------------------------------------------------")
        print("Saved optimize: " + str(optimize))
        optimize = optimize + 1
    
        allTweets = []
        if next_page == None:
            print("No more tweets to retrieve")    
            count = False
        else:
            print("Sleeping...")
            time.sleep(70) # To deal with twitter restrictions
            print("Go to optimize: " + str(optimize) + " with token: " + next_page)
        print("----------------------------------------------------------------")


#%% Info about tweets we want to get

all_media = ['cmjornal', 
             'cnnportugal', 
             'dntwit', 
             'expresso', 
             'JNegocios', 
             'JornalNoticias', 
             'Lusa_noticias', 
             'observadorpt', 
             'ojeconomico', 
             'Publico', 
             'sapo', 
             'SICNoticias', 
             'SolOnline', 
             'TSFRadio', 
             'RTPNoticias',
             'RTP1']

name_files_saved = []
allDataTogether = []
#start_date = "2021-12-27T00:00:00.000Z" #Retrive tweets from start_date to actual date
start_date = "2022-06-07T00:00:00.000Z" #Retrive tweets from start_date to actual date
end_time = "2022-06-08T00:00:00.000Z"
for x in all_media:
        number_of_results = 100
        from_name = x
        print("----------------------------")
        print(x)
        print("----------------------------")
       	makeTweetCollect(headers, from_name, number_of_results, start_date, name_files_saved, allDataTogether, end_time)

filename = str(end_time.split("T")[0])

with open(str(filename)+".json", 'w') as outfile:
    json.dump(allDataTogether, outfile)

#%%

today = date.today()

# dd/mm/YY
d1 = today.strftime("%d/%m/%Y")
print("d1 =", d1)
#today.year + "-"
#2014-01-01T00:00:00.000Z


#%% Put in csv format

f = open(filename+'.json')
data = json.load(f)
    

def make_df(data):
    return pd.DataFrame(data)

df = make_df(data)
df

df.to_csv(filename+'.csv')


#%% Filter By Keywords

keywords = ['novo banco', 'novobanco']

count = 1

with open(filename +'.csv', 'r', encoding="utf8") as csvfile:
    with open (filename + '(Filter by keywords).csv','w', newline='', encoding='utf-8') as fout:        
        datareader = csv.reader(csvfile)
        
        writer = csv.writer(fout)   
        writer.writerow(next(datareader))
        
        for row in datareader:
            sentence = row[9] #text is the position 9 of the filename file
            
            word = "novo banco"
            word2 = "novobanco"
            word3 = "bes"
                                       
            if (word in sentence.lower()) or (word2 in sentence.lower()) or (word3 in sentence.lower()):
                writer.writerow(row)
                print(count)
                count = count + 1
            
