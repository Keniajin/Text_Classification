#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mayank Singh
"""
#import librabries
import string
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn import metrics

#Enter Path
##path='//filepath'

#Enter SubCategory
SubCategory='Subcategory1'

#Enter File Names
Train_Data='Train_v2'
Test_Data='Test'
Global_Data='Global'

#Clean and create Text
exclude = set(string.punctuation)

freq_term = ['corporation','inc', 'corporate', 'ltd', 'laboratories',
             'brand','international','company','gm','ml', SubCategory.lower()] # filter out freq terms

def brand_clean(txt):
    txt = str(txt)    
    txt = ''.join([' ' + x if x.isupper() and i > 0 and txt[i-1] != '-' and i+2 < len(txt) and
                   txt[i+1].islower() and txt[i+2].islower() else x for i, x in enumerate(txt)]).strip() # separate out "BrandProduct" to "Brand Product"
    txt = txt.lower()
    txt = ''.join([ch if ch not in exclude else ' ' for ch in txt])
    txt = ' '.join(x for x in txt.split(' ') if x not in freq_term)
    txt=re.sub(' +',' ',txt)
    return txt.strip()


def clean(txt):
    txt = str(txt)    
    txt = ''.join([' ' + x if x.isupper() and i > 0 and txt[i-1] != '-' and i+2 < len(txt) and
                   txt[i+1].islower() and txt[i+2].islower() else x for i, x in enumerate(txt)]).strip() # separate out "BrandSubbrand" to "Brand Subbrand"
    txt = txt.lower()
    txt=''.join([ch for ch in txt if not ch.isdigit()])
    txt = ''.join([ch if ch not in exclude else ' ' for ch in txt])
    txt = ' '.join(x for x in txt.split(' ') if x not in freq_term)
    txt=re.sub(' +',' ',txt)
    return txt.strip()


def creatingtext(local_brand, local_manufacturer,local_product, local_subbrand,local_variant):
    
    local_variant = brand_clean(local_variant)
    local_brand=brand_clean(local_brand)
    local_manufacturer=brand_clean(local_manufacturer)
    local_product=clean(local_product)
    local_subbrand=brand_clean(local_subbrand)
    local_concat_txt=local_brand.split(' ')
    local_concat_txt.extend(local_manufacturer.split(' '))
    local_concat_txt.extend(local_product.split(' '))
    local_concat_txt.extend(local_subbrand.split(' '))
    local_concat_txt.extend(local_variant.split(' '))
    
    local_concat_txt = list(set(local_concat_txt))  # remove duplicates
    local_concat_txt = ' '.join(local_concat_txt)
    local_concat_txt=re.sub(' +',' ',local_concat_txt)
    return local_concat_txt.strip()
    
    
#Contain a word
def contain_words(s,w):
    return(' '+w+' ') in (' '+s+' ')


#Best global manufacturer match

def manu_match(manufacturer,train_data,global_data):
    manu_match_score=list(train_data['manufacturer'].apply(lambda x:fuzz.ratio(brand_clean(manufacturer),brand_clean(x))))
    max_manu_score=np.max(manu_match_score)
    max_manu_score_index=np.argmax(manu_match_score)
    
    best_global_manu=train_data.loc[max_manu_score_index,'GLOBAL_MANUFACTURER']

    if max_manu_score<70:
        manu_match_score=list(global_data['GLOBAL_MANUFACTURER'].apply(lambda x:fuzz.ratio(brand_clean(manufacturer),brand_clean(x))))
        max_manu_score=np.max(manu_match_score)
        max_manu_score_index=np.argmax(manu_match_score)
        best_global_manu=global_data.loc[max_manu_score_index,'GLOBAL_MANUFACTURER']
        
        if max_manu_score<60:
            best_global_manu='No Match'

    return max_manu_score,best_global_manu
    
#Best global brand match

def brand_match(brand,global_data,train_data,best_global_manu,text,manufacturer,unsp_list):
    bm_score=list(train_data['brand'].apply(lambda x:fuzz.ratio(brand_clean(brand),brand_clean(x))))
    max_brand_score=np.max(bm_score)
    max_brand_score_index_list = [f for f,x in enumerate(bm_score) if x == max_brand_score]
    best_brand_list=list(set([train_data.loc[y,'GLOBAL_BRAND'] for y in max_brand_score_index_list]))
    check='All Other Brands Unsp'
    
    if not any([un for un in unsp_list if contain_words(clean(un),clean(manufacturer))]) and check in best_brand_list and len(best_brand_list)>1:
        best_brand_list.remove(check)
    
    if len(best_brand_list)>1 and max_brand_score>=85:
#        print('Multiple')
        best_brand=list(set([train_data.loc[y,'GLOBAL_BRAND'] for y in max_brand_score_index_list if train_data.loc[y,'GLOBAL_MANUFACTURER']==best_global_manu]))
        if len(best_brand)>1:
#            print('Still Multiple')
            train_data_1=train_data[(train_data['GLOBAL_BRAND'].isin(best_brand)) & (train_data['GLOBAL_MANUFACTURER']==best_global_manufacturer)].reset_index()
            text_score=list(train_data_1['text'].apply(lambda x:fuzz.token_sort_ratio(clean(text),clean(x))))
            max_text_score_index=np.argmax(text_score)
            best_brand=train_data_1.loc[max_text_score_index,'GLOBAL_BRAND']
        elif len(best_brand)==0:
            train_data_1=train_data[(train_data['GLOBAL_BRAND'].isin(best_brand_list))].reset_index()
            text_score=list(train_data_1['text'].apply(lambda x:fuzz.token_sort_ratio(clean(text),clean(x))))
            max_text_score_index=np.argmax(text_score)
            best_brand=train_data_1.loc[max_text_score_index,'GLOBAL_BRAND']
            
        else:
            best_brand=best_brand[0]
    else:
        best_brand=best_brand_list[0]
        
    if max_brand_score<85:
        bm_score=list(global_data['GLOBAL_BRAND'].apply(lambda x:fuzz.ratio(brand_clean(brand),brand_clean(x))))
        max_brand_score=np.max(bm_score)
        max_brand_score_index = np.argmax(bm_score)
        best_brand=global_data.loc[max_brand_score_index,'GLOBAL_BRAND']
        if max_brand_score<70:
            best_brand='No Match'
                
    return best_brand

    
#Approach 1(Use 3 Fuzzy techniques and get best match as per each technique)

def fuzzy_match(train_data,test_text,levels):
    
    if train_data.shape[0]!=0:              
        ratio_score=list(train_data['text'].apply(lambda x:fuzz.ratio(test_text,x)))
        token_set_score=list(train_data['text'].apply(lambda x:fuzz.token_set_ratio(test_text,x)))
        token_sort_score=list(train_data['text'].apply(lambda x:fuzz.token_sort_ratio(test_text,x)))
        
        max_ratio_score=np.max(ratio_score)   
        max_ratio_score_index_list = [f for f,x in enumerate(ratio_score) if x == max_ratio_score]
        
        max_token_set_score=np.max(token_set_score)   
        max_token_set_score_index_list = [f for f,x in enumerate(token_set_score) if x == max_token_set_score]
        
        max_token_sort_score=np.max(token_sort_score)   
        max_token_sort_score_index_list = [f for f,x in enumerate(token_sort_score) if x == max_token_sort_score]
        
        match1=list(set([train_data.loc[y,levels] for y in max_ratio_score_index_list]))
        match2=list(set([train_data.loc[y,levels] for y in max_token_set_score_index_list]))
        match3=list(set([train_data.loc[y,levels] for y in max_token_sort_score_index_list]))
        
    else:
        match1=['could not be mapped']
        match2=['could not be mapped']
        match3=['could not be mapped']
    match_list=match1+match2+match3
    
    return match_list

#Approach 2(Use text classification algorithms)

def algos(train_data,test_data,levels):
        
    classifiers = ['LinSVM','Passive-Aggressive','Ridge','MultinomialNB']
    
    vector = TfidfVectorizer(ngram_range=(1,3),min_df=1)
    x_train=vector.fit_transform(list(train_data['text']))

    x_test = vector.transform(list(test_data['text']))
#    y_test=list(test_data[levels])
    
    prediction=[]
    for j in classifiers:
        print("/n/tClassifier : %s"%(j))
        mod = joblib.load(path+'/'+j+'_'+ levels +".pkl")
        pred = mod.predict(x_test)
#        print("/tTest Accuracy : %f"%(metrics.accuracy_score(y_test,pred)))
        
        prediction.append(pred)
    
    predict=list(zip(*prediction))
    
    for j in range(len(predict)):
        predict[j]=list(predict[j])
 
    return predict  

#Approach 3(Use Global Master Data)

def global_match(global_data,test_text,levels,best_brand):
    
    ex_1=best_brand+' '+'Base'+' '+SubCategory
    ex_2=best_brand+' '+'Original'+' '+SubCategory
    
    if global_data.shape[0]!=0:
        ratio_score=list(global_data['text'].apply(lambda x:fuzz.ratio(test_text,x)))
        token_set_score=list(global_data['text'].apply(lambda x:fuzz.token_set_ratio(test_text,x)))
        token_sort_score=list(global_data['text'].apply(lambda x:fuzz.token_sort_ratio(test_text,x)))
        
        max_ratio_score=np.max(ratio_score)   
        max_ratio_score_index_list = [f for f,x in enumerate(ratio_score) if x == max_ratio_score]
        
        max_token_set_score=np.max(token_set_score)   
        max_token_set_score_index_list = [f for f,x in enumerate(token_set_score) if x == max_token_set_score]
        
        max_token_sort_score=np.max(token_sort_score)   
        max_token_sort_score_index_list = [f for f,x in enumerate(token_sort_score) if x == max_token_sort_score]
        
        
        match1=list(set([global_data.loc[y,levels] for y in max_ratio_score_index_list]))
        match2=list(set([global_data.loc[y,levels] for y in max_token_set_score_index_list]))
        match3=list(set([global_data.loc[y,levels] for y in max_token_sort_score_index_list]))
        
        x=list(global_data[(global_data['GLOBAL_VARIANT'].map(lambda x:x.lower())==ex_1.lower()) | (global_data['GLOBAL_VARIANT'].map(lambda x:x.lower())==ex_2.lower())]['GLOBAL_VARIANT'])
        if levels=='GLOBAL_VARIANT':
            if np.mean([max_ratio_score,max_token_set_score,max_token_sort_score])<40 and len(x)!=0 :
                match1=list(global_data[(global_data['GLOBAL_VARIANT'].map(lambda x:x.lower())==ex_1.lower()) | (global_data['GLOBAL_VARIANT'].map(lambda x:x.lower())==ex_2.lower())]['GLOBAL_VARIANT'])
                match2=match1
                match3=match1
    else:
        match1=['could not be mapped']
        match2=['could not be mapped']
        match3=['could not be mapped']
    match_list=match1+match2+match3
        
    return match_list
        
#Exceptions
def exceptions(global_data,manufacturer,best_global_manu,levels,unsp,max_manu_score,best_brand):
        
    if best_brand=='No Match':
        if max_manu_score>=70:  
            df_subset=global_data[global_data['GLOBAL_MANUFACTURER'].map(lambda x:clean(x))==clean(best_global_manu)].reset_index()
            if df_subset.shape[0]==1:
                match1=list(df_subset[levels])
                match2=match1
                match3=match2
            elif any([variant for variant in list(df_subset['GLOBAL_VARIANT']) if contain_words(clean(variant),clean(unsp[0])) or contain_words(clean(variant),clean(unsp[2]))]):
                match1=list(df_subset[df_subset['GLOBAL_VARIANT']==[l for l in list(df_subset['GLOBAL_VARIANT']) if clean(unsp[0]) in clean(l) or clean(unsp[2]) in clean(l)][0]][levels])
                match2=match1
                match3=match2
            else:
                match1=['Could not be mapped']
                match2=match1
                match3=match2
                
        elif any([un for un in unsp if contain_words(clean(un),clean(manufacturer))]):
        
            match1=['All Other Brands Unsp'+' '+SubCategory]
            match2=match1
            match3=match2
        else:
            match1=['All Other Brands Sp'+' '+SubCategory]
            match2=match1
            match3=match2
    match_list=match1+match2+match3
    
    return match_list


#Read Data Files

df_train=pd.read_csv(path+'/'+Train_Data+'.csv',encoding='utf-8',dtype=str)
df_test=pd.read_csv(path+'/'+Test_Data+'.csv',encoding='utf-8',dtype=str)
df_global=pd.read_csv(path+'/'+Global_Data+'.csv',encoding='utf-8',dtype=str)

#Rename columns

df_global=df_global.rename(columns={'CATEGORY_TEXT"DIV_TXT':'CATEGORY_TEXT"DIV_TXT',
                                            'PRODUCT_CATEGORY"ZPRCATEG':'PRODUCT_CATEGORY"ZPRCATEG',
                                            'PRODUCT_CATEGORY_TEXT"DIV_TXT':'PRODUCT_CATEGORY_TEXT"DIV_TXT',
                                            'SUBCATEGORY"ZSUBCATS':'SUBCATEGORY"ZSUBCATS',
                                            'SUBCATEGORY_TEXT"DIV_TXT':'SUBCATEGORY_TEXT"DIV_TXT',
                                            'BRAND_EQUITY"ZBREQTY':'BRAND_EQUITY"ZBREQTY',
                                            'BRAND_EQUITY_TEXT"DIV_TXT':'GLOBAL_BRAND',
                                            'SUBBRAND"ZSUBBR':'SUBBRAND"ZSUBBR',
                                            'SUBBRAND_TEXT"DIV_TXT':'GLOBAL_SUBBRAND',
                                            'VARIANT"ZGPVAR':'VARIANT"ZGPVAR',
                                            'VARIANT_TEXT"DIV_TXT':'GLOBAL_VARIANT',
                                            'Manufacturer':'GLOBAL_MANUFACTURER'})



#Replace missing text with blanks
df_test.fillna('',inplace=True)
df_global.fillna('',inplace=True)

#Create test and global text

test_text=[]
for row1 in df_test.itertuples():
    concat_test_text=creatingtext(row1.brand,row1.manufacturer,row1.product,row1.sub_brand,row1.variant)
    test_text.append(concat_test_text)

df_test['text']=test_text
                
       
global_text=[]
for row2 in df_global.itertuples():
    concat_global_text=creatingtext(row2.GLOBAL_BRAND,row2.GLOBAL_MANUFACTURER,'',row2.GLOBAL_SUBBRAND,row2.GLOBAL_VARIANT)
    global_text.append(concat_global_text)

df_global['text']=global_text

#Create Output

unsp_list=['other','fractal created','a/o','na','not available','others','autre','autres','otro','otros','altro','ao']
predict=['GLOBAL_VARIANT']

for level in predict:
    final_list=[]
    col10='matchlist_'+level
    col20='global_matchlist_'+level
    col21='Algo_reco_'+level
    col23='Final_list_'+level
    col24='Recommendation_1_'+level
    col25='Accuracy_1_'+level
    col26='Recommendation_2_'+level
    col27='Accuracy_2_'+level
    col28='Recommendation_3_'+level
    col29='Accuracy_3_'+level    
    
    l_alg=algos(df_train,df_test,level)
    
    for row_test in df_test.itertuples():
        print(row_test.Index)
        
        best_manu_score,best_global_manufacturer=manu_match(row_test.manufacturer,df_train,df_global)
        best_global_brand=brand_match(row_test.brand,df_global,df_train,best_global_manufacturer,row_test.text,row_test.manufacturer,unsp_list)
        
        if best_global_brand!='No Match':
        
            df_subset_global_bm=df_global[(df_global['GLOBAL_BRAND'].map(lambda x:clean(x))==clean(best_global_brand))&(df_global['GLOBAL_MANUFACTURER'].map(lambda x:clean(x))==clean(best_global_manufacturer))].reset_index()
            df_subset_train_bm=df_train[(df_train['GLOBAL_BRAND'].map(lambda x:brand_clean(x))==clean(best_global_brand))&(df_train['GLOBAL_MANUFACTURER'].map(lambda x:brand_clean(x))==brand_clean(best_global_manufacturer))].reset_index()
           
            df_subset_global_b=df_global[(df_global['GLOBAL_BRAND'].map(lambda x:brand_clean(x))==brand_clean(best_global_brand))].reset_index()
            df_subset_train_b=df_train[(df_train['GLOBAL_BRAND'].map(lambda x:brand_clean(x))==brand_clean(best_global_brand))].reset_index()
            
            if df_subset_train_bm.shape[0]!=0:    
                df_subset_train=df_subset_train_bm
                df_subset_global=df_subset_global_bm
            else:
                df_subset_train=df_subset_train_b
                df_subset_global=df_subset_global_b
                
            l_fuzzy=fuzzy_match(df_subset_train,row_test.text,level)
            l_global=global_match(df_subset_global,row_test.text,level,best_global_brand)
            l=l_fuzzy+l_global+l_alg[row_test.Index]
            final_list.append(l)
        else:
            l_fuzzy=exceptions(df_global,row_test.manufacturer,best_global_manufacturer,level,unsp_list,best_manu_score,best_global_brand)
            l_global=l_fuzzy
            l=l_fuzzy+l_global
            final_list.append(l)
            
    df_test[col23]=final_list
    df_test[col24]=df_test[col23].apply(lambda x:Counter([i.lower() for i in x]).most_common()[0][0])
    df_test[col25]=df_test[col23].apply(lambda x:(Counter([i.lower() for i in x]).most_common()[0][1]/len(x))*100)
   
    df_test[col26]=df_test[col23].apply(lambda x:Counter([i.lower() for i in x]).most_common()[1][0] if len(set([i.lower() for i in x]))>1 else '')
    df_test[col27]=df_test[col23].apply(lambda x:(Counter([i.lower() for i in x]).most_common()[1][1]/len(x))*100 if len(set([i.lower() for i in x]))>1 else '')
   
    df_test[col28]=df_test[col23].apply(lambda x:Counter([i.lower() for i in x]).most_common()[2][0] if len(set([i.lower() for i in x]))>2 else '')
    df_test[col29]=df_test[col23].apply(lambda x:(Counter([i.lower() for i in x]).most_common()[2][1]/len(x))*100 if len(set([i.lower() for i in x]))>2 else '')
