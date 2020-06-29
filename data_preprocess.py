# -*- encoding: utf-8 -*-
'''
@File    :   data_process.py
@Time    :   2020/06/16 21:51:13
@Author  :   Luo Jianhui 
@Version :   1.0
@Contact :   kid1412ljh@outlook.com
'''

# here put the import lib

import numpy as np 
import pandas as pd
import re
import string

import re
from textblob import Word
from nltk.corpus import stopwords
from tqdm import tqdm
from BERTweet.TweetNormalizer import normalizeTweet

# def processrow(row):
#     tweet = row
#     # Lower case         
#     tweet = tweet.lower()
#     # Removes unicode strings like "\ u002c" and "x96"         
#     tweet = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', tweet)
#     tweet = re.sub(r'[^\x00-\x7f]', r'', tweet)
#     # convert any url to HTTPURL         
#     # tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','HTTPURL', tweet)
#     # Convert any @Username to "@USER"         
#     # tweet = re.sub('@[^\s]+','@USER', tweet)
#     # Remove additional white spaces        
#     tweet = re.sub('[\s]+', ' ', tweet)
#     tweet = re.sub('[\n]+', ' ', tweet)
#     # Replace '`' with '\''
#     tweet = re.sub('[`]+', '\'', tweet)
#     # Remove not alphanumeric symbols white spaces         
#     # tweet = re.sub(r'[^\w]', ' ', tweet)
#     # Removes hastag in front of a word """         
#     # tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
#     # Replace #word with word        
#     # tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
#     # Remove :( or :)         
#     # tweet = tweet.replace(':)','')
#     # tweet = tweet.replace(':(','')
#     # remove numbers         
#     # tweet = ''.join([ i for i in tweet if not i.isdigit()])
#     # remove multiple exclamation         
#     # tweet = re.sub(r"(\!)\1+", ' ', tweet)
#     # remove multiple question marks         
#     # tweet = re.sub(r"(\?)\1+", ' ', tweet)
#     # remove multistop         
#     # tweet = re.sub(r"(\.)\1+", ' ', tweet)
#     # removing stop_words
#     # tweet =" ".join(word for word in tweet.split() if word not in stopwords.words('english'))
#     # lemma                 
#     # tweet =" ".join([Word(word).lemmatize() for word in tweet.split()])
#     # Removes emoticons from text         
#     # tweet = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', tweet)
#     # trim         
#     tweet = tweet.strip()
#     row = tweet
#     return row

def processrow(row):
    # Replace '`' with '\''
    row = re.sub('[`]+', '’', row)
    row = normalizeTweet(row)
    return row


def load_preprocessed_data(path):
    
    data = pd.read_csv(path)
    data.dropna(inplace = True)
    data = data.loc[data.sentiment != 'neutral', :]
    data['text'] = data['text'].apply(lambda x: processrow(x))
    #data['selected_text'] = data['selected_text'].apply(lambda x: processrow(x))
    
    return data

def label_noise_sample(x, y):
    """labeling the noise sample 
    """
    ori = np.array(x.split())
    sel = np.array(y.split())
    hold = False
    for i in range(len(sel)): 
        if sel[i] in ori:
            hold = True
            break
    return hold

def removing_noise_sample(path):
    """Removing the noise sample 
    """
    preprocessed_data = load_preprocessed_data(path)
    hold = []
    for i in tqdm(range(len(preprocessed_data))):
        ori = preprocessed_data.text.values
        sel = preprocessed_data.selected_text.values
        hold.append(label_noise_sample(ori[i], sel[i]))

    clean_data = preprocessed_data.loc[hold,:]
    clean_data.reset_index(drop = True, inplace = True)
    return clean_data


if __name__ == '__main__':
    # data = removing_noise_sample("datasets/train.csv")
    # data.to_csv('clean_data.csv', index = False)

    path = "datasets/test.csv"
    data = load_preprocessed_data(path)
    data.to_csv('clean_test_data.csv', index=False)