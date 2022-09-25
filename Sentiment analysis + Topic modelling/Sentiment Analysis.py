# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:37:01 2022

@author: Gonçalo Mateus

Sentiment Analysis with rule based with Sentilex.pt and Emotaix.pt dictionaries

This Script need a csv file with the {fileName} to work and will return 3 files: 
    1: (f'{fileName} - Sentiment (NotIdiom).csv') - with the sentiment 
        analysis with SentiLex-PT dictionary without taking into consideration 
        idiomatic expressions
    2: (f'{fileName} - Sentiment (Idiomatic).csv') - with the sentiment 
        analysis with SentiLex-PT dictionary taking into consideration 
        idiomatic expressions
    3: (f'{fileName} - Sentiment_ALL.csv') - with the sentiment 
        analysis with SentiLex-PT dictionary taking into consideration 
        idiomatic expressions + the sentiment analysis with Emotaix.pt 
        dictionary

"""
# %% Imports

import pandas as pd
from datetime import datetime
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import re

#%% Load data

#custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z")

fileName = "AllMediaTweets_DF(Filter by keywords)"

df = pd.read_csv(f'{fileName}.csv', 
                  #parse_dates=['created_at'],
                  #date_parser=custom_date_parser,
                  low_memory=False)

df.drop("Unnamed: 0", inplace=True, axis=1)

#%% Import SentiLex-PT Dictionary and organize info

sentilex_flex = pd.read_csv(r'SentiLex-PT02/SentiLex-flex-PT02.txt', delimiter=".",
                   header=None)

split_words = sentilex_flex[0].str.split(',', expand=True)
classify = sentilex_flex[1].str.split(';', expand=True)
sentilex_dataframe = pd.DataFrame(split_words)      

sentilex_dataframe["polNo"] = classify[3].str.split('=', expand=True)[1] #polarity
sentilex_dataframe["pOs"] = classify[0].str.split('=', expand=True)[1] #idiomatic
sentilex_IDIOM = sentilex_dataframe[sentilex_dataframe["pOs"] == "IDIOM"]

#%% Import emotaix.PT Dictionary

emotaix = pd.read_csv(r'EMOTAIX.PT/emotaix.csv', low_memory=False)

#%% Initial data preprocessing for sentiment analysis
    #Cleaning the text - Use REGEX rules
        #Normalize text
            #Remove URL’s, special caracters($%&#) Hashtags, Mentions
            #Remove Punctuation
            #convert to lowercase
    
        #Didn't take the hifens because of words like aborrecer-se ou aproveitar-se that are in the database
       
#Removing RT, Punctuation etc
remove_rt = lambda x: re.sub('RT @\w+: '," ", x)
rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^-,%0-9A-ZÁÀÂÃÉÈÊÍÌÓÒÔÕÚÇa-záàâãéèêíìóòôõúç \t])|(\w+:\/\/\S+)"," ",x)
remove_double_space = lambda x: re.sub("  "," ", x)
removed = df.text.map(remove_rt).map(rt).map(remove_double_space)
text_cleaned = removed.str.lower()
text_cleaned_df = pd.DataFrame(text_cleaned)    
df["text_cleaned"] = text_cleaned

#%% Sentiment analysis with SentiLex-PT

#Notes:
    # 1 - We didn´t tokenize for this analysis because of the existence of ideomatic expressions and if we tokenize we coun´t classify all the expressions and have a more accurate rate of tweets classification

    # 2 - Not much good datasets in portuguese
        #Best on is the SentiLex present in http://b2find.eudat.eu/dataset/b6bd16c2-a8ab-598f-be41-1e7aeecd60d3
        #Is the more completed one available in portuguese of Portugal
            #In detail, the lexicon describes: 4,779 (16,863) adjectives, 1,081 (1,280) nouns, 489 (29,504) verbs, and 666 (34,700) idiomatic expressions.
            #Main features:
                #Part-of-speech (ADJ(ective), N(oun), V(erb) and IDIOM)
                #Sentiment attributes:  Polarity (POL), which can be positive (1), negative (-1) or neutral (0); 
                                        #Target of polarity (TG), which corresponds to a human noun (HUM), functioning as the subject (N0) and/or the complement (N1) of the predicate; 
                                        #Polarity annotation (ANOT), which was performed manually (MAN) or automatically, by the Judgment Analysis Lexicon Classifier (JALC) tool, developed by the project team.   
            #Polarity annotation (ANOT), which was performed manually (MAN) or automatically, by the Judgment Analysis Lexicon Classifier (JALC) tool, developed by the project team
                       
positive = []
negative = []
neutral = []
total = []
    
for x in tqdm(range(len(text_cleaned_df))):
    sum_negative = 0
    sum_positive = 0
    sum_neutral = 0
    sum_total = 0
    
    text=text_cleaned_df.iloc[x].text
    tokenize_text = word_tokenize(text)
    
    for z in range(len(tokenize_text)):
        isInDatabase = sentilex_dataframe[sentilex_dataframe[0] == tokenize_text[z]]
        isInDatabase2 = sentilex_dataframe[sentilex_dataframe[1] == tokenize_text[z]]

        if(len(isInDatabase) + len(isInDatabase2) != 0):
            polarity = 0
            
            if(len(isInDatabase) != 0):
                if(len(isInDatabase) > 1):
                    polarity = int(isInDatabase[:1]["polNo"])
                else:
                    polarity = int(isInDatabase["polNo"])                   
                
            if(len(isInDatabase2) != 0 and len(isInDatabase) == 0):    
                print("Special case: " + tokenize_text[z])
                
                if(len(isInDatabase2) > 1):
                    polarity = int(isInDatabase2[:1]["polNo"])
                else:
                    polarity = int(isInDatabase2["polNo"])                   
            
            sum_total += polarity
            if(polarity != 0):
                if(polarity>0):
                    sum_positive += 1
                else:
                    sum_negative += 1
            else:
                sum_neutral += 1                    
            
    total.append(sum_total)
    positive.append(sum_positive)
    negative.append(sum_negative)
    neutral.append(sum_neutral)      

df["sentiment_total"] = total
df["sentiment_positive"] = positive
df["sentiment_negative"] = negative
df["sentiment_neutral"] = neutral

df.to_csv(f'{fileName} - Sentiment (NotIdiom).csv')

#-----------------------------------------------------------------------------
# Check For Idiomatic expressions with SentiLex-PT
#-----------------------------------------------------------------------------

positive_idiomatic = np.zeros(len(text_cleaned_df))
negative_idiomatic = np.zeros(len(text_cleaned_df))
neutral_idiomatic = np.zeros(len(text_cleaned_df))
total_idiomatic = np.zeros(len(text_cleaned_df))

for z in tqdm(range(len(sentilex_IDIOM[0]))): 
    
    for x in (range(len(text_cleaned_df))):
        
        if(sentilex_IDIOM[0].iloc[z] in text_cleaned_df.iloc[x].text):
            
            sum_negative = 0
            sum_positive = 0
            sum_neutral = 0
            sum_total = 0
            
            polarity = int(sentilex_IDIOM.iloc[z]["polNo"])
            
            sum_total += polarity
            if(polarity != 0):
                if(polarity>0):
                    sum_positive += 1
                else:
                    sum_negative += 1
            else:
                sum_neutral += 1         
                
                
            positive_idiomatic[x] = positive_idiomatic[x] + sum_positive
            negative_idiomatic[x] = negative_idiomatic[x] + sum_negative
            neutral_idiomatic[x] = neutral_idiomatic[x] + sum_neutral
            total_idiomatic[x] = total_idiomatic[x] + sum_total
            
            print(text_cleaned_df.iloc[x].text)
            print(sentilex_IDIOM[0].iloc[z] )

df["sentiment_total_idiomatic"] = total_idiomatic
df["sentiment_positive_idiomatic"] = positive_idiomatic
df["sentiment_negative_idiomatic"] = negative_idiomatic
df["sentiment_neutral_idiomatic"] = neutral_idiomatic

df.to_csv(f'{fileName} - Sentiment (Idiomatic).csv')
     
#%% Sentiment analysis with Emotaix.pt

#custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")
df = pd.read_csv(r'AllMediaTweets_DF(Filter by keywords) - Sentiment (Idiomatic).csv', 
                  #parse_dates=['created_at'],
                  #date_parser=custom_date_parser,
                  low_memory=False)
df.drop("Unnamed: 0", inplace=True, axis=1)

emotaix = emotaix.iloc[: , 1:]
emotaix = emotaix.reset_index()  # make sure indexes pair with number of rows
emotaix = emotaix.iloc[: , 1:]
array =[]
colunas = emotaix.columns
colunasfinal = []

for x in colunas:
    colunasfinal.append(x.split(".")[0])
    
colunasfinal = list(dict.fromkeys(colunasfinal))

BENEVOLÊNCIA = []
MALEVOLÊNCIA= []
BEMESTAR= [] 
MALESTAR= [] 
SEGURANÇA= []
ANSIEDADE= []
SURPRESA= []
EMOÇÕESNÃOESPECÍFICAS= [] 
INDIFERENÇA= []

text_cleaned_df = df["text_cleaned"]
count = 0
for x in tqdm(range(len(text_cleaned_df))):
    
    sum_BENEVOLÊNCIA = 0
    sum_MALEVOLÊNCIA = 0
    sum_BEMESTAR = 0
    sum_MALESTAR = 0
    sum_SEGURANÇA = 0
    sum_ANSIEDADE = 0
    sum_SURPRESA = 0
    sum_EMOÇÕESNÃOESPECÍFICAS = 0
    sum_NDIFERENÇA = 0


    text=text_cleaned_df.iloc[x]    
    tokenize_text = word_tokenize(text)
    
    for z in range(len(tokenize_text)):
    
        for column in emotaix: #vai a cada coluna
        
            array = emotaix[column].values         
            isInDatabase = len(array[array == tokenize_text[z]]) != 0            
                
            if(isInDatabase):
                count = count + 1
                if("BENEVOLÊNCIA" in column):
                    sum_BENEVOLÊNCIA = sum_BENEVOLÊNCIA + 1
                if("MALEVOLÊNCIA" in column):
                    sum_MALEVOLÊNCIA = sum_MALEVOLÊNCIA + 1                    
                if("BEM ESTAR" in column):
                    sum_BEMESTAR = sum_BEMESTAR + 1
                
                if("MAL ESTAR" in column):
                    sum_MALESTAR = sum_MALESTAR + 1                    

                if("SEGURANÇA" in column):
                    sum_SEGURANÇA = sum_SEGURANÇA + 1                  
                if("ANSIEDADE" in column):
                    sum_ANSIEDADE = sum_ANSIEDADE + 1                    
                if("SURPRESA" in column):
                    sum_SURPRESA = sum_SURPRESA + 1    
                if("EMOÇÕES NÃO ESPECÍFICAS" in column):
                    sum_EMOÇÕESNÃOESPECÍFICAS = sum_EMOÇÕESNÃOESPECÍFICAS + 1                        
                if("INDIFERENÇA" in column):
                    sum_NDIFERENÇA = sum_NDIFERENÇA + 1                               
                    
        
    BENEVOLÊNCIA.append(sum_BENEVOLÊNCIA)
    MALEVOLÊNCIA.append(sum_MALEVOLÊNCIA)
    BEMESTAR.append(sum_BEMESTAR)
    MALESTAR.append(sum_MALESTAR)
    SEGURANÇA.append(sum_SEGURANÇA)
    ANSIEDADE.append(sum_ANSIEDADE)
    SURPRESA.append(sum_SURPRESA)
    EMOÇÕESNÃOESPECÍFICAS.append(sum_EMOÇÕESNÃOESPECÍFICAS)
    INDIFERENÇA.append(sum_NDIFERENÇA)
    
print(count)
confirm = sum(BENEVOLÊNCIA)+sum(BEMESTAR)+sum(SEGURANÇA)
print(confirm)
confirm2 = sum(MALEVOLÊNCIA)+sum(MALESTAR)+sum(ANSIEDADE) 
print(confirm2)
confirm3 = sum(SURPRESA)+sum(EMOÇÕESNÃOESPECÍFICAS)+sum(INDIFERENÇA)
print(confirm3)

df["emotaix_benevolencia"] = BENEVOLÊNCIA
df["emotaix_malevolencia"] = MALEVOLÊNCIA
df["emotaix_bemestar"] = BEMESTAR
df["emotaix_malestar"] = MALESTAR
df["emotaix_seguranca"] = SEGURANÇA
df["emotaix_ansiadade"] = ANSIEDADE
df["emotaix_surpresa"] = SURPRESA
df["emotaix_emocoesnaoespecificas"] = EMOÇÕESNÃOESPECÍFICAS
df["emotaix_indiferenca"] = INDIFERENÇA

df.to_csv(f'{fileName} - Sentiment_ALL.csv')