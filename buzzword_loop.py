import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from nltk.corpus import reuters, stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Code to download corpora
import nltk
nltk.download('reuters')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def process_text(doc):
    sw = set(stopwords.words('english'))
    regex = re.compile("[^a-zA-Z ]")
    re_clean = regex.sub('', doc)
    words = word_tokenize(re_clean)
    lem = [lemmatizer.lemmatize(word) for word in words]
    output = [word.lower() for word in lem if word.lower() not in sw]
    return output


bitcoin_df = pd.read_csv('Data/bitcoin_df.csv', parse_dates=True)
bitcoin_df = bitcoin_df.rename(columns={"Unnamed: 0":"date"})
bitcoin_df['date'] = pd.to_datetime(bitcoin_df['date']).dt.date

google_trend_df = pd.read_csv('Data/btc_google_trend.csv', parse_dates=True)
google_trend_df['date'] = pd.to_datetime(google_trend_df['date']).dt.date
google_trend_df = google_trend_df.rename(columns={"bitcoin":"google trend"})

reddit_sentiments_df = pd.read_csv('Data/reddit_sentiments_df.csv')
reddit_sentiments_df.drop('Unnamed: 0', axis=1, inplace=True)
reddit_sentiments_df = reddit_sentiments_df.rename(columns={"text":"Reddit text"})

google_sentiments_df = pd.read_csv('Data/google_sentiment_df.csv')
google_sentiments_df['date'] = pd.to_datetime(google_sentiments_df['date']).dt.date

bitcoin_df = bitcoin_df.merge(google_trend_df, on="date")
bitcoin_df = bitcoin_df.merge(reddit_sentiments_df, on="datetime")
bitcoin_df = bitcoin_df.merge(google_sentiments_df, on="date")

bitcoin_df.drop('datetime', axis=1, inplace=True)

print(bitcoin_df.head())

date_list = bitcoin_df.loc[bitcoin_df['class'] == 1].index.to_list()
bitcoin_df['wordbank'] = 0
days_back = 3
for i in range(days_back):
    bitcoin_df.loc[[date - i for date in date_list], 'wordbank'] = 1
# bitcoin_df.loc[date_list, 'wordbank'] = 1
# bitcoin_df.loc[[date - 1 for date in date_list], 'wordbank'] = 1
# bitcoin_df.loc[[date - 2 for date in date_list], 'wordbank'] = 1
bitcoin_df.loc[bitcoin_df['wordbank'] == 1]


temp1 = []
temp2 = []
for i in range(len(bitcoin_df.loc[bitcoin_df['wordbank'] == 1])):
    temp1.append(str(bitcoin_df.loc[i]['Reddit text']))
    temp2.append(str(bitcoin_df.loc[i]['Google text']))

reddit_text = ' '.join(temp1)
google_text = ' '.join(temp2)

# Getting the TF-IDF
reddit_vectorizer = TfidfVectorizer(stop_words="english")
reddit_corpus = reddit_vectorizer.fit_transform([reddit_text])
reddit_words_corpus = reddit_vectorizer.get_feature_names()
# Getting the TF-IDF weight of each word in corpus as DataFrame
reddit_words_corpus_df = pd.DataFrame(
    list(zip(reddit_words_corpus, np.ravel(reddit_corpus.mean(axis=0)))), columns=["Word", "TF-IDF"]
)

reddit_words_corpus_df = reddit_words_corpus_df.sort_values(by=["TF-IDF"], ascending=False)
reddit_words_corpus_df = reddit_words_corpus_df.loc[reddit_words_corpus_df['TF-IDF'] > 0.04]

google_vectorizer = TfidfVectorizer(stop_words="english")
google_corpus = google_vectorizer.fit_transform([google_text])
google_words_corpus = google_vectorizer.get_feature_names()
# Getting the TF-IDF weight of each word in corpus as DataFrame
google_words_corpus_df = pd.DataFrame(
    list(zip(google_words_corpus, np.ravel(google_corpus.mean(axis=0)))), columns=["Word", "TF-IDF"]
)

google_words_corpus_df = google_words_corpus_df.sort_values(by=["TF-IDF"], ascending=False)
google_words_corpus_df = google_words_corpus_df.loc[google_words_corpus_df['TF-IDF'] > 0.04]

print(reddit_words_corpus_df.head())
print(google_words_corpus_df.head())

reddit_buzzword_score = []
google_buzzword_score = []
index_list = bitcoin_df.index.to_list()
for i in index_list:
    print(i)
    try:
        reddit_score = 0
        for word in process_text(str(bitcoin_df['Reddit text'][i])):
            for index, row in reddit_words_corpus_df.iterrows():
                if row['Word'] in [word]:
                    reddit_score += row['TF-IDF']
            
        google_score = 0
        for word in process_text(str(bitcoin_df['Google text'][i])):
            for index, row in google_words_corpus_df.iterrows():
                if row['Word'] in [word]:
                    google_score += row['TF-IDF']        
        
        reddit_buzzword_score.append({
            "index": i,
            "reddit buzzword score": reddit_score
        })
        google_buzzword_score.append({
            "index": i,
            "google buzzword score": google_score
        })
    except AttributeError:
        pass

reddit_buzzword_df = pd.DataFrame(reddit_buzzword_score)
reddit_buzzword_df.to_csv('reddit_buzzwords.csv')

google_buzzword_df = pd.DataFrame(google_buzzword_score)
google_buzzword_df.to_csv('google_buzzwords.csv')



