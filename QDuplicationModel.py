
# coding: utf-8

# In[48]:




# In[49]:

'''
Интересные kernels
https://www.kaggle.com/hubert0527/spacy-name-entity-recognition
https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb
'''


# In[67]:

import pandas as pd
folder = r'C:\TMP'

raw_train = pd.read_csv(folder + r'\train.csv', engine = 'python')

tmp = pd.DataFrame()
qs = [list(zip([row['question1']], [row['question2']], [row['is_duplicate']]))
       for index, row in raw_train.iterrows() if len(str(row['question1'])) <= 50 and len(str(row['question2'])) <= 50]
print(len(qs))

tmp['question1'] = [item[0][0] for item in qs]
tmp['question2'] = [item[0][1] for item in qs]
tmp['is_duplicate'] = [item[0][2] for item in qs]
raw_train = tmp
raw_train.head()


# In[68]:

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stops = set(stopwords.words('english'))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    
    shared_words = [w for w in q1words.keys() if w in q2words]
    res = 2 * len(shared_words) / (len(q1words) + len(q2words))
    return res


# In[69]:

import re
import numpy as np
from collections import Counter

def get_weight(count, total_count, min_count=3):
    if count < min_count:
        return 0
    else:
        return count / total_count
    
def tokenize(sentence):
    text = re.sub("\'s", " ", sentence) # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("[^a-zA-Z]"," ", text).lower().split()
    return text
    
words = raw_train['question1'].apply(str).tolist() + raw_train['question2'].apply(str).tolist()
words = [tokenize(s) for s in words]
words = [word for sentence in words for word in sentence]
counts = Counter(words)
weights = {word: get_weight(count, len(words)) for word, count in counts.items()}


# In[70]:

from gensim.models import word2vec as w2v

my_w2v = w2v.Word2Vec.load(folder + r'\w2v.model')


# In[71]:

def word_similarity(row):
    q1words = {}
    q2words = {}
    for word in tokenize(str(row['question1'])):
        if word not in stops:
            q1words[word] = 1
    for word in tokenize(str(row['question2'])):
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    
    similarities = [max([my_w2v.similarity(w1, w2) for w2 in q2words.keys()]) for w1 in q1words.keys()]
    return np.sum(similarities) / len(q1words)


# In[72]:

def weight_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in tokenize(str(row['question1'])):
        if word not in stops:
            q1words[word] = 1
    for word in tokenize(str(row['question2'])):
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    res = np.sum(shared_weights) / np.sum(total_weights)
    return res


# In[73]:

from nltk.corpus import wordnet as wn

def is_synonym(word, synonym):
    synsets = wn.synsets(word)
    res = []
    for ss in synsets:
        if ss.name().startswith(word):
            res.append(ss.lemma_names())
            
    res = list(set([item for lst in res for item in lst]))
    return synonym in res

def synonym_wms(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    
    shared_words = [w for w in q1words.keys() for w2 in q2words.keys() if is_synonym(w2, w) or w == w2]
    return 2 * len(shared_words) / (len(q1words) + len(q2words))


# In[74]:

from nltk.tag import pos_tag

def get_prop_nouns(sentence):
    tagged_sent = pos_tag(sentence.split())
    proper_nouns = [word for word,pos in tagged_sent if pos == 'NNP']
    
    return proper_nouns

def propn_wms(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    
    prop1 = get_prop_nouns(' '.join(list(q1words.keys())))
    prop2 = get_prop_nouns(' '.join(list(q2words.keys())))
    shared_props = [w for w in prop1 if w in prop2]
    
    return (2 * len(shared_props)) / (len(q1words.keys()) + len(q2words.keys()))


# In[75]:

from nltk import pos_tag
from nltk.corpus import wordnet as wn
 
def penn_to_wn(tag):
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None
    

def my_max(lst):
    if len(lst) > 0:
        return max(lst)
    else:
        return 0
    
def sentence_similarity(row, rec = False):
    # Tokenize and tag
    if not rec:
        s1 = row['question1']
        s2 = row['question2']
    else:
        s1 = row['question2']
        s2 = row['question1']
        
    s1 = tokenize(str(s1))
    s2 = tokenize(str(s2))
    s1 = [w for w in s1 if w not in stops]
    s2 = [w for w in s2 if w not in stops]
        
    sentence1 = pos_tag(s1)
    sentence2 = pos_tag(s2)
    
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss is not None]
    synsets2 = [ss for ss in synsets2 if ss is not None]
    #print(synsets1, synsets2)

    score, count = 0.0, 0
 
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = my_max([synset.path_similarity(ss) for ss in synsets2 if synset.path_similarity(ss) is not None])
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1
            
    # Average the values
    if count > 0:
        score /= count
    else:
        score = 0
    # sentence_sim(s1, s2) != sentence_sim(s2, s1), по этому берем среднее арифметическое
    if not rec:
        score += sentence_similarity(row, rec=True)
        score/=2
    return score


# In[78]:

#prop_wms = raw_train.apply(propn_wms, axis=1, raw=True)
#weight_wms = raw_train.apply(weight_word_match_share, axis=1, raw=True)
wms = raw_train.apply(word_match_share, axis=1, raw=True)
weight_wms = raw_train.apply(weight_word_match_share, axis=1, raw=True)

x = pd.DataFrame()
x['wms'] = wms
x['weight_wms'] = weight_wms

y = raw_train['is_duplicate'].values


# In[ ]:




# In[79]:

import xgboost as xgb
from sklearn.cross_validation import train_test_split

x_train, x_check, y_train, y_check = train_test_split(x, y, test_size=0.2, random_state=271828)

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = ['logloss', 'auc']
params['eta'] = 0.5
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_check = xgb.DMatrix(x_check, label=y_check)

watchlist = [(d_train, 'train'), (d_check, 'valid')]

bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50, verbose_eval=25)


# In[61]:

#bst.save_model(r'C:\Python\MODELS\wms_weightwms.model')
#bst.save_model(r'C:\Python\MODELS\synwms_weightwms.model')
#bst.save_model(r'C:\Python\MODELS\prop_weigth_wms.model')
#bst.save_model(r'C:\Python\Models\wms_sim.model')
#bst.save_model(r'C:\Python\Models\wms_newsim.model')


# In[ ]:




# In[ ]:




# In[ ]:



