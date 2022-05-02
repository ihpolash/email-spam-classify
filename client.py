import os
import pandas as pd

import argparse
import json
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

from sklearn.preprocessing import LabelEncoder
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle

def train_model(max_features):
    path = 'dataset/hamnspam/'
    mails = []
    labels = []

    for label in ['ham/', 'spam/']:
    #     filenames = os.listdir(os.path.join(path, label))
        filenames = os.listdir(path + label)
        for file in filenames:
            f = open((path + label + file), 'r', encoding = 'latin-1')
            bolk = f.read()
            mails.append(bolk)
            labels.append(label)
            
    df = pd.DataFrame({'emails': mails, 'labels': labels})
    encoder = LabelEncoder()
    df['labels'] = encoder.fit_transform(df['labels'])
    df['emails'] = df['emails'].apply(lambda x:x.lower())
    df['emails'] = df['emails'].apply(lambda x: x.replace('\n', ' '))
    df['emails'] = df['emails'].apply(lambda x: x.replace('\t', ' '))

    nltk.download('stopwords')
    ps = PorterStemmer()

    corpus = []
    for i in range(len(df)):
        ## Becuase of removed duplicated entries...
        review = re.sub('[^a-zA-Z]', ' ', df['emails'][i])
        review = review.split()
        review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    
    cv = CountVectorizer(max_features = max_features)
    X = cv.fit_transform(corpus).toarray()
    y = df['labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    CM = confusion_matrix(y_test, y_pred)

    #Save Model 
    f = open('spam_classifier.pickle', 'wb')
    pickle.dump(model, f)
    f.close()

    response = {"result": accuracy, "confusion_matrix": CM}

    return response

def email_classify(email_string):

    # image_path = default_storage.save("tmp/test.txt", ContentFile(email_string.read()))
    # tmp_file = os.path.join(settings.MEDIA_ROOT, image_path)
    # f = open(tmp_file, 'r')
    # content = f.read() 
    df = pd.DataFrame({'emails': email_string}, index=[0])
    print(df)

    df['emails'] = df['emails'].apply(lambda x:x.lower())
    df['emails'] = df['emails'].apply(lambda x: x.replace('\n', ' '))
    df['emails'] = df['emails'].apply(lambda x: x.replace('\t', ' '))

    nltk.download('stopwords')

    ps = PorterStemmer()

    # Load Corpus
    f = open('corpus.pickle', 'rb')
    corpus = pickle.load(f)
    f.close()
    ## Becuase of removed duplicated entries...
    for i in range(len(df)):
        ## Becuase of removed duplicated entries...
        
        review = re.sub('[^a-zA-Z]', ' ', df['emails'][i])
        review = review.split()
        review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

    #Load Model
    f = open('spam_classifier.pickle', 'rb')
    model = pickle.load(f)
    f.close()

    cv = CountVectorizer(max_features = 2500)
    X = cv.fit_transform(corpus).toarray()

    result = model.predict([X[-1]])
    labels = ["ham", "spam"]
    # default_storage.delete(tmp_file)
    response = {"result": labels[int(result)]}

    return response

