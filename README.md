# Project-1
import warnings 
<br>
warnings.filterwarnings("ignore")

<br>
import numpy as np
<br>
import pandas as pd
<br>
import matplotlib.pyplot as plt
<br>
import seaborn as sns

<br>
df = pd.read_csv('movie.csv')
<br>
df.head()

<br>
df = df.drop(['Unnamed: 0','Name of movie','Year of relase','Description','Director','Star'],axis=1)

<br>
df.head()

<br>
df[df['Metascore'] == '^^^^^^']
<br>
df['Metascore'] = df['Metascore'].replace('^^^^^^','0')
<br>
df['Gross collection'] = df['Gross collection'].replace('*****','0')

<br>
df.head()

<br>
cols = df.columns
<br>
cols
<br>
for col in cols:
<br>
    df[col] = df[col].astype('string') 
    
<br>
data = df.copy()
<br>
data.head()

<br>
j = 0
<br>
for i in data['Gross collection']:
<br>
    if i == '0':
<br>
        data['Gross collection'][j] = i
 <br>
        j = j+1
<br>
    else:
<br>    
        n = len(i)
<br>
        z = i
<br>
        i = i[1:n-1]
<br>
        data['Gross collection'][j] = i
<br>
        j = j+1
<br>
    #df[i] = df[1:n-1]
<br>
data.head()

<br>
j = 0
<br>
for i in data['Votes']:
<br>
    n = len(i)
    <br>
    z = i
    <br>
    i = i.replace(',','')
    <br>
    data['Votes'][j] = i
    <br>
    j = j+1
    <br>
    #df[i] = df[1:n-1]
    <br>
data.head()
<br>

data['Gross collection'] = data['Gross collection'].astype('float')







