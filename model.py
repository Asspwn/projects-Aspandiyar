####
# Importing relevant libraries
####
import nltk
from collections import Counter
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import pymorphy2
import string
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from string import digits 
import math
from nltk import word_tokenize
from collections import defaultdict
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import gc
from scipy import sparse
import os
from sklearn.datasets import make_classification


directory = r'C:\Users\anurimanov\OneDrive - KPMG\Desktop\Tender_Analysis\Train Data'

os.chdir(directory)

df = pd.read_excel(r'C:\Users\anurimanov\KPMG\Aleksandrova, Nataliia - DnA - Hackaton\Train Data\Rus\Train_Rus.xlsx')

####
# Data cleansing
####
df['Text'] = df['Tender Name'] + ' ' + df['Description']
df = df[df['Text'].notna()]
df = df.replace(np.nan, '', regex=True)
df.reset_index(drop=True, inplace=True)

df = df.head(100)
df.iloc[0,2] = 1

full_data = df

kpmg_2 = full_data[['Tender Name','Description','Results']]


kpmg_2['Text'] = kpmg_2['Tender Name'] + ' ' + kpmg_2['Description']
kpmg_2 = kpmg_2[kpmg_2['Tender Name'].notna()]




def remove(text):
    return text[10:]

kpmg_2['Text'] = kpmg_2['Text'].apply(remove)

df1 = kpmg_2[['Text','Results']]
df1.columns = ['Description','Result']

def r_dup(sent):
    words = sent.split() 
    res = [] 
    for word in words: 
        # If condition is used to store unique string  
        # in another list 'k'  
        if (sent.count(word)>1 and (word not in res)or sent.count(word)==1): 
            res.append(word) 
    return ' '.join(res)


df1['Description'] = df1['Description'].apply(r_dup)


####
#Remove words | bag of words
####


def text_process(mess):
    
    mess = mess.replace('/', ' ')
    mess = mess.replace('(',' ')
    mess = mess.replace(')',' ')
    mess = mess.replace('-',' ')
    
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    nopunc = [word for word in nopunc.split() if word.lower() not in stopwords.words('russian')]
    return ' '.join(nopunc)
####
# Delete stopwords
# Delete numbers
####
morph = pymorphy2.MorphAnalyzer()


def lemmatize(text):
    res = list()
    words = text.split() # Decomposing text to words
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.normal_form)
        
    #removing numbers    
    res = [''.join(x for x in i if x.isalpha()) for i in res] 
    while '' in res:
        
        res.remove('')     
    
    return ' '.join(res)

####
# Post taging/morphology of words
####

def pos_tag(text):
    words = text.split()
    res = list()
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.tag.POS)
    return res       


####
# Feature enginerring, extracting information from the text
####

df1['Description'] = df1['Description'].apply(r_dup)
df1['Lemma'] = df1['Description'].apply(text_process)
df1['Lemma'] = df1['Lemma'].apply(lemmatize)

print('LR l2 norm, balanced')                                
    
                              
df1['Description'] = df1['Description'].apply(lemmatize)
df1['pos_teg'] = df1['Result']
df1['pos_teg']= df1['Description'].apply(pos_tag)                                        
  

####
# Devide to results to positive and negative sentiment
####

df_1 = df1.loc[df1.Result == 1].reset_index(drop=True)
df_0 = df1.loc[df1.Result == 0].reset_index(drop=True)

####
# Splitting the dataframe to test and train
####

x_train, x_test, y_train, y_test = train_test_split(df1['Description'],df1['Result'],test_size=0.1)

####
# Creating the pipeline: TFIDF Vectorization
# Logistic Regression with Ridge normalization
# Initially our dataset was imbalanced, around 10% of relevant data. So we decided to upsample our dataset
####

pipeline = Pipeline([
    ('tfidf',TfidfVectorizer(max_features=20000,analyzer='char',ngram_range=(3,6),dtype=np.float32)),
    ('classifier', LogisticRegression('l2',class_weight='balanced'))
])

pipeline.fit(df1['Description'],df1['Result'])