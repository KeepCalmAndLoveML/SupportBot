# coding: utf-8
# In[1]:
import os
import re
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

from collections import Counter
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")

# engine=python для русскоязычных путей
db_path = input()
questions = pd.read_csv(db_path, engine='python')
questions = np.array_split(questions, 600)[0]

count_per_qs = int(input())


def get_question():
    q = input()
    return q


def tokenize(sentence):
    text = re.sub("\'s", " ", sentence)
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("[^a-zA-Z]", " ", text).lower().split()
    return text


def get_weight(count, total_count, min_count=3):
    if count < min_count:
        return 0
    else:
        return count / total_count


stops = set(stopwords.words('english'))


def word_match_share(q1, q2):
    q1words = {}
    q2words = {}
    for word in q1:
        if word not in stops:
            q1words[word] = 1
    for word in q2:
        if word not in stops:
            q2words[word] = 1

    shared_words = [w for w in q1words.keys() if w in q2words]
    res = 2 * len(shared_words) / (len(q1words) + len(q2words))
    return res


words = questions['Question'].apply(str).apply(tokenize).tolist()
sentences = words
words = [w for sentence in words for w in sentence]
counts = Counter(words)
weights = {word: get_weight(count, len(words)) for word, count in counts.items()}


def weight_word_match_share(usr_question, question2):
    usr_words = {}
    q2words = {}
    for word in usr_question:
        usr_words[word] = 1
    for word in question2:
        q2words[word] = 1

    shared_weights = [weights.get(w, 0) for w in usr_words.keys() if w in weights.keys() and w in q2words]
    shared_weights += [weights.get(w, 0) for w in q2words.keys() if w in usr_words]

    total_weights = [weights.get(w, 0) for w in usr_words if w in weights.keys()] + [weights.get(w, 0) for w in q2words]

    res = np.sum(shared_weights) / np.sum(total_weights)
    return res


def predict(q_text, bst):
    df = pd.DataFrame()
    weight_wms = []
    wms = []
    param1 = tokenize(str(q_text))
    for stc in sentences:
        weight_wms.append(weight_word_match_share(param1, stc))
        wms.append(word_match_share(param1, stc))

    df['wms'] = wms
    df['weight_wms'] = weight_wms

    matrix = xgb.DMatrix(df)
    return list(zip(list(questions['Answer']), list(bst.predict(matrix))))


model = xgb.Booster({'n_thread': 4})
model.load_model(os.path.dirname(os.path.realpath(__file__)) + '\wms_weightwms.model')

while True:
    question = get_question()
    if question == ';!END!;':
        sys.exit('Exit requested by user...')
    predictions = predict(question, model)
    predictions.sort(key=lambda x: x[1], reverse=True)

    for i in range(count_per_qs + 1):
        if predictions[i][1] >= 0.45:
            print(predictions[i][0])
        else:
            print(';!END!;')
            break
