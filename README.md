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



