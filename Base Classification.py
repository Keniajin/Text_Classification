#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mayank Singh
"""
import re
import datetime
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

#Enter Path
##path='//filepath'

#Enter SubCategory
SubCategory='Subcategory1'

#Enter File Names
Train_Data='Train'

#Clean text
exclude = set(string.punctuation)

freq_term = ['corporation','inc', 'corporate', 'ltd', 'laboratories',
             'all', 'brand','international','gm','ml', SubCategory.lower()] # filter out freq terms

def brand_clean(txt):
    txt = str(txt)    
    txt = ''.join([' ' + x if x.isupper() and i > 0 and txt[i-1] != '-' and i+2 < len(txt) and
                   txt[i+1].islower() and txt[i+2].islower() else x for i, x in enumerate(txt)]).strip() # separate out "BrandProduct" to "Brand Product"
    txt=txt.lower()
    txt = ''.join([ch if ch not in exclude else ' ' for ch in txt])
    txt = ' '.join(x for x in txt.split(' ') if x not in freq_term)
    txt=re.sub(' +',' ',txt)
    return txt.strip()

def clean(txt):
    txt = str(txt)    
    txt = ''.join([' ' + x if x.isupper() and i > 0 and txt[i-1] != '-' and i+2 < len(txt) and
                   txt[i+1].islower() and txt[i+2].islower() else x for i, x in enumerate(txt)]).strip() # separate out "BrandSubbrand" to "Brand Subbrand"
    txt=txt.lower()
    txt=''.join([ch for ch in txt if not ch.isdigit()])
    txt = ''.join([ch if ch not in exclude else ' ' for ch in txt])
    txt = ' '.join(x for x in txt.split(' ') if x not in freq_term)
    txt=re.sub(' +',' ',txt)
    return txt.strip()


def creatingtext(local_brand, local_manufacturer,local_product, local_subbrand,local_variant):
    
    local_variant = clean(local_variant)
    local_brand=brand_clean(local_brand)
    local_manufacturer=brand_clean(local_manufacturer)
    local_product=clean(local_product)
    local_subbrand=clean(local_subbrand)
    local_concat_txt=local_brand.split(' ')
    local_concat_txt.extend(local_manufacturer.split(' '))
    local_concat_txt.extend(local_product.split(' '))
    local_concat_txt.extend(local_subbrand.split(' '))
    local_concat_txt.extend(local_variant.split(' '))
    
    local_concat_txt = list(set(local_concat_txt))  # remove duplicates
    local_concat_txt = ' '.join(local_concat_txt)
    local_concat_txt=re.sub(' +',' ',local_concat_txt)
    return local_concat_txt.strip()


#Read Data
df_train=pd.read_csv(path+'/'+Train_Data+'.csv',encoding='utf8',dtype=str)

#Rename Columns
df_train=df_train.rename(columns={'Col1_name':'NewColname','SUBCATEGORY_NAME':'Subcategory','PRODUCT':'product',
                                  'LOCAL_MANUFACTURER':'manufacturer','LOCAL_BRAND':'brand',
                                  'LOCAL_SUBBRAND':'subbrand','LOCAL_VARIANT':'variant',
                                  'GLOBAL_MANUFACTURER':'GLOBAL_MANUFACTURER',
                                  'GLOBAL_BRAND':'GLOBAL_BRAND',
                                  'GLOBAL_SUBBRAND':'GLOBAL_SUBBRAND',
                                  'GLOBAL_VARIANT':'GLOBAL_VARIANT'})

#Replace missing text with blanks
df_train.fillna('',inplace=True)

#Create train text
train_text=[]
for row in df_train.itertuples():
    concat_text=creatingtext(row.brand,row.manufacturer,row.product,row.subbrand,row.variant)
    train_text.append(concat_text)

df_train['text']=train_text
#

#df_train['text']=df_train.apply(lambda (row['brand'],row['manufacturer'],row['product'],row['subbrand'],row['variant']):creatingtext(row['brand'],row['manufacturer'],row['product'],row['subbrand'],row['variant']))
#

#Training

classifiers = {
    'Ridge': RidgeClassifier(solver="sag"),
    'Passive-Aggressive': PassiveAggressiveClassifier(n_iter=50,n_jobs=-1),
    'LinSVM': LinearSVC(C=0.9),
    'MultinomialNB' : MultinomialNB(alpha=0.001)
    }

doc= list(df_train['text'])

vector = TfidfVectorizer(ngram_range=(1,3),min_df=1)
x = vector.fit_transform(doc)

for level in ['GLOBAL_VARIANT']:
    
    print("\nFor : %s"%(level))
    y = list(df_train[level])
    
    for l in classifiers.items():
        print("\n\tAlgorithm: %s"%(l[0]))
        print(datetime.datetime.now().time())        
        l[1].fit(x,y)
        print("\tTraining score : %f"%(l[1].score(x,y)))
    
        joblib.dump(l[1],path+'/'+l[0]+"_"+level+".pkl")  

#Modified training Data with Text column

df_train.to_csv(path+'/'+"Train_v2.csv",index=False,encoding='utf-8')
