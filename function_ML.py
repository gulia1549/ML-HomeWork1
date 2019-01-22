#!/usr/bin/env python
# coding: utf-8

# In[102]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
data = pd.read_csv('weatherAUS.csv')


def isNaN(num):
    if num != num :
        return True
    else:
        return False
def mymean(column):
    s=0
    j=0
    for i in column:
        if isNaN(i)==False:
            s+=i
            j+=1
    rez=s/j
    return rez

def std_new (column):
    mymean1=mymean(column)
    sigma=0
    j=0
    for i in column:
        if isNaN(i)==False:
            sigma+=(i-mymean1)**2
            j+=1
    mystd=(sigma/j)**0.5
    return mystd
        
def replace_mean(column):
    mymean1=mymean(column)
    for i in range(len(column)):
        if isNaN(column[i])==True:
            column[i]=mymean1   
    return column        

def median_new(column):
    l=[]
    for i in column:
        if isNaN(i)==False:
            l.append(i)
    l=sorted(l)
    n=len(l)
    if n % 2 == 1:
            return l[n//2]
    else:
            return sum(l[n//2-1:n//2+1])/2   

def replace_median(column):
    med=median_new(column)
    for i in range(len(column)):
        if isNaN(column[i])==True:
            column[i]=med   
    return column

def moda_new(column):
    d={}
    for i in column:
        if isNaN(i)==False:
            if i not in d.keys():
                d[i]=1
            else:
                d[i]+=1
    v=list(d.values())
    k=list(d.keys())
    return k[v.index(max(v))]            

def replace_moda(column):
    mod=moda_new(column)
    for i in range(len(column)):
        if isNaN(column[i])==True:
            column[i]=mod   
    return column   

def drop_column(df,list_name):
    for column_name in list_name:
        for i in df[column_name]:
            if isNaN(i)==True:
                df=df.drop(column_name,1)
                break
    return df

def drop_row(df,row_number):
    for i in df.iloc[row_number]:
        if isNaN(i)==True:
            df=df.drop(row_number,axis=0)
            break
    return df

def ln_regres(df,target,list_column):
    df_copy=df.copy()
    l=list_column.copy()
    l.append(target)
    new_df=df_copy[l]
    train_df=new_df.dropna(axis=0, how='any')
    X_train=train_df.iloc[:,:-1]
    y_train=train_df.iloc[:,-1]
    
    LR = LinearRegression()
    LR.fit(X_train, y_train)
    for i in range(df_copy.shape[0]):
        if isNaN(df_copy[target].iloc[i])==True:
            df_copy[target].iloc[i] = LR.predict(df_copy[list_column].iloc[i].values.reshape(1,-1) )
    return df_copy        
            
def scaling(column):
    max_value = max(column)
    min_value = min(column)
    for i in range(len(column)):
            column[i] = (column[i]-min_value)/(max_value-min_value)  
    return column 

def standart(column):
    mean = mymean(column)
    sigma = std_new (column)
    for i in range(len(column)):
            column[i] =(column[i]-mean)/sigma 
    return column     
              

