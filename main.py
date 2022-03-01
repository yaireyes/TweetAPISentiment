################################################################

                        #Yaisiel Reyes

################################################################
#Time Needed 3 hours

# -*- coding: utf-8 -*-
# Import Libraries
#~/.matplotlib/matplotlibrc
from turtle import rt

import pc as pc
from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
#dataframe.plot.hist()
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string

from tweepy import API
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer


# Authentication
consumerKey = "qCakzeAs29UJmF14nGv70gJHD"
consumerSecret = "rcI1dQnk3O2KL7JtBXwQs3ixo22V1QlwqPTYK3x4ggNXoO9DuR"
accessToken = "1498069965873639427-1g08ha7ghcHbtBzAOYBtaDCAyjZTIw"
accessTokenSecret = "0nZ4EQgPWY22FHcGVWucj5MD7ICEiMM34V0ngFn27PPgV"
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)


# Sentiment Analysis
def percentage(part, whole):
    return 100 * float(part) / float(whole)


keyword = raw_input("Please enter keyword or hashtag to search: ")
noOfTweet = eval(raw_input("Please enter how many tweets to analyze: "))
tweets = tweepy.Cursor(api.search, q=keyword).items(noOfTweet)
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []
for tweet in tweets:

    # print(tweet.text)
    tweet_list.append(tweet.text)
    analysis = TextBlob(tweet.text)
    score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    polarity += analysis.sentiment.polarity

if neg > pos:
    negative_list.append(tweet.text)
    negative += 1
elif pos > neg:
    positive_list.append(tweet.text)
    positive += 1

elif pos == neg:
    neutral_list.append(tweet.text)
neutral += 1
positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
polarity = percentage(polarity, noOfTweet)
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')

#Number of Tweets (Total, Positive, Negative, Neutral)
tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)
print("total number: ",len(tweet_list))
print("positive number: ",len(positive_list))
print("negative number: ", len(negative_list))
print("neutral number: ",len(neutral_list))

#Creating PieCart
labels = ['Positive ['+str(positive)+'%]' , 'Neutral['+str(neutral)+'%]','Negative ['+str(negative)+'%]']
sizes = [positive, neutral, negative]
colors = ['yellowgreen', 'blue','red']
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title("Sentiment Analysis Result for keyword= "+keyword+"" )
plt.axis('equal')
plt.show()

tweet_list.drop_duplicates(inplace = True)

#Cleaning Text (RT, Punctuation etc)
#Creating new dataframe and new features
tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list[0]
#Removing RT, Punctuation etc
remove_rt = lambda x: re.sub('RT @\w+: '," ",x)

tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
tw_list["text"] = tw_list.text.str.lower()
tw_list.head(10)

#Calculating Negative, Positive, Neutral and Compound values
tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tw_list['text'].iteritems():
 score = SentimentIntensityAnalyzer().polarity_scores(row)
 neg = score['neg']
 neu = score['neu']
 pos = score['pos']
 comp = score['compound']
 if neg > pos:
     tw_list.loc[index, 'sentiment'] = "negative"
 elif pos > neg:
     tw_list.loc[index, 'sentiment'] = "positive"
 else:
     tw_list.loc[index, 'sentiment'] = "neutral"
     tw_list.loc[index, 'neg'] = neg
     tw_list.loc[index, 'neu'] = neu
     tw_list.loc[index, 'pos'] = pos
     tw_list.loc[index, 'compound'] = comp
     tw_list.head(10)

#Creating new data frames for all sentiments (positive, negative and neutral)
tw_list_negative = tw_list[tw_list["sentiment"]=="negative"]
tw_list_positive = tw_list[tw_list["sentiment"]=="positive"]
tw_list_neutral = tw_list[tw_list["sentiment"]=="neutral"]

def count_values_in_column(data,feature):
 total=data.loc[:,feature].value_counts(dropna=False)
 percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
 return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
#Count_values for sentiment
count_values_in_column(tw_list,"sentiment")

 # create data for Pie Chart
pichart = count_values_in_column(tw_list,"sentiment")
names = pc.index
size = pc["Percentage"]

 # Create a circle for the center of the plot
my_circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(size, labels=names, colors=['green', 'blue', 'red'])
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

#Function to Create Wordcloud
def create_wordcloud(text):
 mask = np.array(Image.open("cloud.png"))
 stopwords = set(STOPWORDS)
 wc = WordCloud(background_color="white",
 mask = mask,
 max_words=3000,
 stopwords=stopwords,
 repeat=True)
 wc.generate(str(text))
 wc.to_file("wc.png")
 print("Word Cloud Saved Successfully")
 path="wc.png"
 display(Image.open(path))

 # Creating wordcloud for all tweets
 create_wordcloud(tw_list["text"].values)

 # Creating wordcloud for positive sentiment
 create_wordcloud(tw_list_positive["text"].values)

 # Creating wordcloud for negative sentiment
 create_wordcloud(tw_list_negative["text"].values)

# Calculating tweet’s lenght and word count
 tw_list['text_word_count'] = tw_list['text'].apply(lambda x: len(str(x).split()))
 round(pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()), 2)
 round(pd.DataFrame(tw_list.groupby("sentiment").text_word_count.mean()), 2)

 # Removing Punctuation
 def remove_punct(text):
     text = "".join([char for char in text if char not in string.punctuation])
     text = re.sub('[0–9]+', '', text)
     return text

 tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punct(x))

 # Appliyng tokenization
 def tokenization(text):
     text = re.split('\W+', text)
     return text

 tw_list['tokenized'] = tw_list['punct'].apply(lambda x: tokenization(x.lower()))

 # Removing stopwords
 stopword = nltk.corpus.stopwords.words('english')

 def remove_stopwords(text):
     text = [word for word in text if word not in stopword]
     return text

 tw_list['nonstop'] = tw_list['tokenized'].apply(lambda x: remove_stopwords(x))
 # Appliyng Stemmer
 ps = nltk.PorterStemmer()

 def stemming(text):
     text = [ps.stem(word) for word in text]
     return text

 tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: stemming(x))

 # Cleaning Text
 def clean_text(text):
     text_lc = "".join([word.lower() for word in text if word not in string.punctuation])  # remove puntuation
     text_rc = re.sub('[0-9]+', '', text_lc)
     tokens = re.split('\W+', text_rc)  # tokenization
     text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
     return text

 tw_list.head()

 # Appliyng Countvectorizer
 countVectorizer = CountVectorizer(analyzer=clean_text)
 countVector = countVectorizer.fit_transform(tw_list['text'])
 print('{} Number of reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))
 # print(countVectorizer.get_feature_names())
 #1281 Number of reviews has 2966 words
 count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
 count_vect_df.head()

 # Most Used Words
 count = pd.DataFrame(count_vect_df.sum())
 countdf = count.sort_values(0, ascending=False).head(20)
 countdf[1:11]

 # Function to ngram
 def get_top_n_gram(corpus, ngram_range, n=None):
     vec = CountVectorizer(ngram_range=ngram_range, stop_words= 'english').fit(corpus)
     bag_of_words = vec.transform(corpus)
     sum_words = bag_of_words.sum(axis=0)
     words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
     words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
     return words_freq[:n]

 # n2_bigram
 n2_bigrams = get_top_n_gram(tw_list['text'], (2, 2), 20)
 n2_bigrams

 # n3_trigram
 n3_trigrams = get_top_n_gram(tw_list['text'], (3, 3), 20)
 n3_trigrams