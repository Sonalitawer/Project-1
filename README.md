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
<br>

cols = data.columns
<br>
cols
<br>
for col in cols:
    <br>
    data[col] = data[col].astype('float')
<br>
data.corr()

<br>
cols = data.columns
<br>
cols
<br>
for col in cols:
    <br>
    data[col] = data[col].replace(0,np.median(data[col]))
    <br>
data.head()
<br>

new_data = data[data['Gross collection']>5]
<br>
new_data = new_data.drop('Metascore',axis=1)
<br>
#new_data = new_data[new_data['Metascore']>50]
<br>
new_data.shape

<br>
x = data.drop('Gross collection',axis=1)
<br>
y = data['Gross collection']
<br>

info = data.describe()
<br>
sns.heatmap(info,annot=True,fmt='.2f')
<br>

cols = x.columns
<br>
n = len(cols)
<br>
cols = cols[:n]
<br>
for i in cols:
<br>
    x[i] =  (x[i]- min(x[i]))/(max(x[i]-min(x[i])))
<br>
y
<br>

plt.figure(figsize=(10,10))
<br>
#df.columns
<br>
cols = x.columns
<br>
i = 0
<br>
for col in cols:
<br>
    plt.subplot(2,2,i+1)
    <br>
    sns.scatterplot(x=x[col],y=y,color='red')
    <br>
    i = i+1

<br>
plt.figure(figsize=(10,10))
<br>
#df.columns
<br>
cols = x.columns
<br>
i = 0
<br>
for col in cols:
    <br>
    plt.subplot(2,2,i+1)
    <br>
    sns.distplot(x[col])
    <br>
    i = i+1
<br>
info = x.describe()
<br>
sns.heatmap(info,annot=True,fmt='.2f')
<br>
info = x.corr()
<br>
sns.heatmap(info,annot=True,fmt='.2f')
<br>

x.shape
<br>

x_train = x[:720]
<br>
y_train = y[:720]
<br>
x_test = x[721:800]
<br>
y_test = y[721:800]
<br>
from sklearn.linear_model import LinearRegression

<br>
model = LinearRegression()
<br>
model = model.fit(x_train,y_train)
<br>
model.score(x_test,y_test)
<br>

ypred = model.predict(x_test)
<br>
from sklearn.metrics import mean_absolute_error
<br>
print("MAE",mean_absolute_error(y_test,ypred))
<br>
from sklearn.metrics import mean_squared_error
<br>
print("MSE",mean_squared_error(y_test,ypred))
<br>
print("RMSE",(np.sqrt(mean_squared_error(y_test,ypred))))
<br>
ypred1 = ypred
<br>
from sklearn.metrics import r2_score
<br>
print("R2 score:",r2_score(y_test, ypred))
<br>

sns.scatterplot(x = y_test,y = ypred,color='orange',label='Predicted')
<br>
x = np.arange(0,np.max(y_test),0.1)
<br>
y = np.arange(0,np.max(y_test),0.1)
<br>
sns.lineplot(x=x,y=y,label='Actual')
<br>
plt.xlabel("Actual Collection")
<br>
plt.ylabel("Predicted Collection")
<br>
plt.title('LinearRegression')
<br>
plt.show()
<br>

from sklearn.model_selection import GridSearchCV
<br>
n_estimators=[100,200,300]
<br>
max_depth=[int(x) for x in np.linspace(10,200,50)]
<br>
#max_depth = int(max_depth)
<br>
random_grid = {
<br>   
'n_estimators':n_estimators,
<br>   
'max_depth':max_depth,
<br>
}

<br>
from sklearn.ensemble import RandomForestRegressor
<br>
rfc = RandomForestRegressor()
<br>
forest_params = random_grid
<br>
clf = GridSearchCV(rfc, forest_params)
<br>
clf.fit(x_train, y_train)
<br>
clf.best_params_
<br>

clf.best_params_
<br>

from sklearn.ensemble import RandomForestRegressor
<br>
model = RandomForestRegressor(max_depth=37, n_estimators=100)
<br>
model = model.fit(x_train,y_train)
<br>
ypred = model.predict(x_test)
<br>
from sklearn.metrics import mean_absolute_error
<br>
print("MAE",mean_absolute_error(y_test,ypred))
<br>
from sklearn.metrics import mean_squared_error
<br>
print("MSE",mean_squared_error(y_test,ypred))
<br>
print("RMSE",(np.sqrt(mean_squared_error(y_test,ypred))))
<br>
ypred2 = ypred
<br>
from sklearn.metrics import r2_score
<br>
print("R2 score:",r2_score(y_test, ypred))
<br>

sns.scatterplot(x = y_test,y = ypred,color='orange',label='Predicted')
<br>
x = np.arange(0,np.max(y_test),0.1)
<br>
y = np.arange(0,np.max(y_test),0.1)
<br>
sns.lineplot(x=x,y=y,label='Actual')
<br>
plt.xlabel("Actual Collection")
<br>
plt.ylabel("Predicted Collection")
<br>
plt.title('Random Forest')
<br>
plt.show()
<br>

n_estimators=[100,200,300]
<br>
learning_rate=np.arange(0.1,1,0.1)
<br>
#max_depth = int(max_depth)
<br>
loss = ['square','linear']
<br>

random_grid = {
<br>
    'n_estimators':n_estimators,
    <br>
    'loss':loss,
    <br>
    'learning_rate':learning_rate
    <br>
    
}
<br>

from sklearn.ensemble import AdaBoostRegressor
<br>
rfc = AdaBoostRegressor()
<br>
forest_params = random_grid
<br>
clf = GridSearchCV(rfc, forest_params)
<br>
clf.fit(x_train, y_train)
<br>
clf.best_estimator_
<br>

from sklearn.ensemble import AdaBoostRegressor
<br>
model = AdaBoostRegressor(learning_rate=0.1, loss='square', n_estimators=100)
<br>
model = model.fit(x_train,y_train)
<br>
ypred = model.predict(x_test)
<br>
from sklearn.metrics import mean_absolute_error
<br>
print("MAE",mean_absolute_error(y_test,ypred))
<br>
from sklearn.metrics import mean_squared_error
<br>
print("MSE",mean_squared_error(y_test,ypred))
<br>
print("RMSE",(np.sqrt(mean_squared_error(y_test,ypred))))
<br>
ypred3 = ypred
<br>
from sklearn.metrics import r2_score
<br>
print("R2 score:",r2_score(y_test, ypred))
<br>

sns.scatterplot(x = y_test,y = ypred,color='orange',label='Predicted')
<br>
x = np.arange(0,np.max(y_test),0.1)
<br>
y = np.arange(0,np.max(y_test),0.1)
<br>
sns.lineplot(x=x,y=y,label='Actual')
<br>
plt.xlabel("Actual Collection")
<br>
plt.ylabel("Predicted Collection")
<br>
plt.title('Adaboost Regressor')
<br>
plt.show()
<br>

n_estimators=[100,200,300]
<br>
max_depth=[int(x) for x in np.linspace(10,200,50)]
<br>
#max_depth = int(max_depth)
<br>
learning_rate = np.arange(0.1,1,0.1)
<br>
max_leaves = np.arange(10,60,10)
<br>
random_grid = {
<br>
    'n_estimators':n_estimators,
    <br>
    'max_depth':max_depth,
    <br>
    'max_leaves':max_leaves
<br>
}
<br>

from xgboost import XGBRegressor
<br>
rfc = XGBRegressor()
<br>
forest_params = random_grid
<br>
clf = GridSearchCV(rfc, forest_params)
<br>
clf.fit(x_train, y_train)
<br>
clf.best_estimator_
<br>

from xgboost import XGBRegressor
<br>
model = XGBRegressor(max_delta_step=0, max_depth=10, max_leaves=10, min_child_weight=1)
<br>
model = model.fit(x_train,y_train)
<br>
ypred = model.predict(x_test)
<br>
from sklearn.metrics import mean_absolute_error
<br>
print("MAE",mean_absolute_error(y_test,ypred))
<br>
from sklearn.metrics import mean_squared_error
<br>
print("MSE",mean_squared_error(y_test,ypred))
<br>
print("RMSE",(np.sqrt(mean_squared_error(y_test,ypred))))
<br>
ypred4 = ypred
<br>
from sklearn.metrics import r2_score
<br>
print("R2 score:",r2_score(y_test, ypred))
<br>

sns.scatterplot(x = y_test,y = ypred,color='orange',label='Predicted')
<br>
x = np.arange(0,np.max(y_test),0.1)
<br>
y = np.arange(0,np.max(y_test),0.1)
<br>
sns.lineplot(x=x,y=y,label='Actual')
<br>
plt.xlabel("Actual Collection")
<br>
plt.ylabel("Predicted Collection")
<br>
plt.title('XgBoost Regressor')
<br>
plt.show()
<br>

report1 = pd.DataFrame()
<br>
report1['Actual'] = y_test
<br>
report1['Predicted'] = ypred1
<br>
report1.head()
<br>

report2 = pd.DataFrame()
<br>
report2['Actual'] = y_test
<br>
report2['Predicted'] = ypred2
<br>
report2.head()
<br>

report3 = pd.DataFrame()
<br>
report3['Actual'] = y_test
<br>
report3['Predicted'] = ypred3
<br>
report3.head()
<br>

report4 = pd.DataFrame()
<br>
report4['Actual'] = y_test
<br>
report4['Predicted'] = ypred4
<br>
report4.head()
<br>


