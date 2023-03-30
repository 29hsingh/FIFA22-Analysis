import sys
sys.stdout.reconfigure(encoding='utf-8')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import snscrape.modules.twitter as sntwitter
import pandas as pd
import textblob
from textblob import TextBlob
import re
def analyze_sentiment(tweet):
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    tweet_proc = " ".join(tweet_words)
    print(tweet_proc)
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    

    # sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores

# query = "(from:EASPORTSFIFA) until:2019-01-01 since:2018-01-01"
query = "(from:richardbranson)"
limit = 1
tweets_list1 = []
# data = sntwitter.TwitterSearchScraper(query).get_items()
# for i in data:
#     print(i)

for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()): 
    if i>limit-1: 
        break
    # if tweet.location != ' ':
    #     print(tweet.location)
    tweets_list1.append([tweet.date, tweet.id, tweet.rawContent, tweet.user.username, tweet.likeCount]) 

# tweets_df1 = pd.DataFrame(tweets_list1, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
# print(tweets_df1)
ans = [0,0,0]
for item in tweets_list1:
    temp = []
    labels = ['Negative', 'Neutral', 'Positive']
    temp = analyze_sentiment(item[2])
    print(item[4])                                 ######   ADDITIONAL
    ans[0] = ans[0]+temp[0]
    ans[1] = ans[1]+temp[1]
    ans[2] = ans[2]+temp[2]

for i in range(len(ans)):
    l = labels[i]
    s = round ( (ans[i]/(limit) * 100) , 3 )
    print(l,s)
print("\n\n")