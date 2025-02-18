import re, nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

model = joblib.load('svc.sav')
vocabulary = pd.read_csv('vocabulary.csv', header=None)
vocabulary_dict = {}
for i, word in enumerate(vocabulary[0]):
      vocabulary_dict[word] = i
print(vocabulary_dict)
tfidf = TfidfVectorizer(vocabulary = vocabulary_dict,lowercase=False)

# Reading new data as dataframe
df = pd.read_csv("trump_tweets.csv")
pd.set_option('display.max_colwidth', None) # Setting this so we can see the full content of cells
pd.set_option('display.max_columns', None) # to make sure we can see all the columns in output window

# Cleaning Tweets
def cleaner(tweet):
    soup = BeautifulSoup(tweet, 'lxml') # removing HTML entities such as ‘&amp’,’&quot’,'&gt'; lxml is the html parser and shoulp be installed using 'pip install lxml'
    souped = soup.get_text()
    re1 = re.sub(r"(@|http://|https://|www|\\x)\S*", " ", souped) # substituting @mentions, urls, etc with whitespace
    re2 = re.sub("[^A-Za-z]+"," ", re1) # substituting any non-alphabetic character that repeats one or more times with whitespace

    """
    For more info on regular expressions visit -
    https://docs.python.org/3/howto/regex.html
    """

    tokens = nltk.word_tokenize(re2)
    lower_case = [t.lower() for t in tokens]

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

df['cleaned_tweets'] = df.tweets.apply(cleaner)
df = df[df['cleaned_tweets'].map(len) > 0] # removing rows with cleaned tweets of length 0
print("Printing top 5 rows of dataframe showing original and cleaned tweets....")
print(df[['tweets','cleaned_tweets']].head())
df['cleaned_tweets'] = [" ".join(row) for row in df['cleaned_tweets'].values] # joining tokens to create strings. TfidfVectorizer does not accept tokens as input
data = df['cleaned_tweets']
print(df['cleaned_tweets'].head())
tfidf.fit(data)
data_tfidf = tfidf.transform(data)
y_pred = model.predict(data_tfidf)
#### Saving predicted sentiment of tweets to csv
df['predicted_sentiment'] = y_pred.reshape(-1,1)
df.drop(['id', 'created_at'], axis=1, inplace=True)
df.to_csv('predicted_sentiment.csv', index=False)
