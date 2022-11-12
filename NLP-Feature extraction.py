import string
import nltk
import pandas as pd
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
import os
wd='/Users/kamran/Desktop/3-year/unigrams'
os.chdir(wd)
#Lemma function
def get_lemma(text):
    lem=WordNetLemmatizer().lemmatize
    return lem(text)
#Stem function
def get_stem(text):
    stem = PorterStemmer().stem
    return stem(text)
# current directory csv files
csvs = [x for x in os.listdir('.') if x.endswith('.csv')]
# stats.csv -> stats
fns = [os.path.splitext(os.path.basename(x))[0] for x in csvs]
#reading dataset - Features extraction: Lemma,Stem,POS_tag - Export CSV file
df={}
d = {}
for i in range(len(fns)):
    d[fns[i]] = pd.read_csv(csvs[i])
    d[fns[i]] = d[fns[i]].applymap(str)
    d[fns[i]]["Lem"]=d[fns[i]].Keyword.apply(lambda x : get_lemma(x))
    d[fns[i]]['Stem']=d[fns[i]].Keyword.apply(lambda x : get_stem(x))
    df[fns[i]]=pd.DataFrame(nltk.tag.pos_tag(d[fns[i]].Keyword))
    df[fns[i]]=d[fns[i]].join(df[fns[i]])
    df[fns[i]]=df[fns[i]].drop(0, axis=1)
    df[fns[i]]=df[fns[i]].rename({1: 'POS_tag'}, axis=1)
    x = df[fns[i]]
    for i in fns:
        x.to_csv(str(i)+ "_"+ ".csv", index = False)
# Doing the same for bigrams
#changing the wd
new_wd='/Users/kamran/Desktop/3-year/bigrams'
os.chdir(new_wd)
# current directory csv files
csvs = [x for x in os.listdir('.') if x.endswith('.csv')]
# stats.csv -> stats
fns = [os.path.splitext(os.path.basename(x))[0] for x in csvs]
df={}
d = {}
#reading dataset - Features extraction: Lemma,Stem,POS_tag - Export CSV file
for i in range(len(fns)):
    d[fns[i]] = pd.read_csv(csvs[i])
    d[fns[i]] = d[fns[i]].applymap(str)
    d[fns[i]]["Lem"]=d[fns[i]].Keyword.apply(lambda x : get_lemma(x))
    d[fns[i]]['Stem']=d[fns[i]].Keyword.apply(lambda x : get_stem(x))
    df[fns[i]]=pd.DataFrame(nltk.tag.pos_tag(d[fns[i]].Keyword))
    df[fns[i]]=d[fns[i]].join(df[fns[i]])
    df[fns[i]]=df[fns[i]].drop(0, axis=1)
    df[fns[i]]=df[fns[i]].rename({1: 'POS_tag'}, axis=1)
    x = df[fns[i]]
    for i in fns:
        x.to_csv(str(i)+ "_"+ ".csv", index = False)

