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
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# Reading dataset as dataframe
df = pd.read_csv("News.csv")
pd.set_option('display.max_colwidth', None) # Setting this so we can see the full content of cells
pd.set_option('display.max_columns', None) # to make sure we can see all the columns in output window
df['Category'] = df['Category'].map({'Sport':1, 'Sci/Tech':0})

# Cleaning summaries
def cleaner(summary):
    soup = BeautifulSoup(summary, 'lxml') # removing HTML entities such as ‘&amp’,’&quot’,'&gt'; lxml is the html parser and shoulp be installed using 'pip install lxml'
    souped = soup.get_text()
    re1 = re.sub(r"(#|@|http://|https://|www)\S*", " ", souped) # substituting hashtags, @mentions, urls, etc with whitespace
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

df['cleaned_summary'] = df.Summary.apply(cleaner)
df = df[df['cleaned_summary'].map(len) > 0] # removing rows with cleaned summaries of length 0
print("Printing top 5 rows of dataframe showing original and cleaned summaries....")
print(df[['Summary','cleaned_summary']].head())
df['cleaned_summary'] = [" ".join(row) for row in df['cleaned_summary'].values] # joining tokens to create strings. TfidfVectorizer does not accept tokens as input
data = df['cleaned_summary']
Y = df['Category'] # target column
tfidf = TfidfVectorizer(min_df=.0005, ngram_range=(1,3)) # min_df=.0005 means that each ngram (unigram, bigram, & trigram) must be present in at least 30 documents for it to be considered as a token (60000*.0005=30). This is a clever way of feature engineering
tfidf.fit(data) # learn vocabulary of entire data
data_tfidf = tfidf.transform(data) # creating tfidf values
pd.DataFrame(pd.Series(tfidf.get_feature_names_out())).to_csv('news_vocabulary.csv', header=False, index=False)
print("Shape of tfidf matrix: ", data_tfidf.shape)

print("Implementing SVC.....")
# Implementing Support Vector Classifier
svc_clf = LinearSVC() # kernel = 'linear' and C = 1

# Running cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) # 10-fold cross-validation
scores=[]
iteration = 0
for train_index, test_index in kf.split(data_tfidf, Y):
    iteration += 1
    print("Iteration ", iteration)
    X_train, Y_train = data_tfidf[train_index], Y.iloc[train_index]
    X_test, Y_test = data_tfidf[test_index], Y.iloc[test_index]
    svc_clf.fit(X_train, Y_train) # Fitting SVC
    Y_pred = svc_clf.predict(X_test)
    score = metrics.accuracy_score(Y_test, Y_pred) # Calculating accuracy
    print("Cross-validation accuracy: ", score)
    scores.append(score) # appending cross-validation accuracy for each iteration
svc_mean_accuracy = np.mean(scores)
print("Mean cross-validation accuracy: ", svc_mean_accuracy)

print("Implementing NBC.....")
# Implementing Naive Bayes Classifier
nbc_clf = MultinomialNB()

# Running cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) # 10-fold cross-validation
scores=[]
iteration = 0
for train_index, test_index in kf.split(data_tfidf, Y):
    iteration += 1
    print("Iteration ", iteration)
    X_train, Y_train = data_tfidf[train_index], Y.iloc[train_index]
    X_test, Y_test = data_tfidf[test_index], Y.iloc[test_index]
    nbc_clf.fit(X_train, Y_train) # Fitting NBC
    Y_pred = nbc_clf.predict(X_test)
    score = metrics.accuracy_score(Y_test, Y_pred) # Calculating accuracy
    print("Cross-validation accuracy: ", score)
    scores.append(score) # appending cross-validation accuracy for each iteration
nbc_mean_accuracy = np.mean(scores)
print("Mean cross-validation accuracy: ", nbc_mean_accuracy)