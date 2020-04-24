import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from math import sqrt
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from IPython import get_ipython

bike_df = pd.read_csv('day.csv')

print(bike_df['season'].value_counts())
print(bike_df['yr'].value_counts())
print(bike_df['mnth'].value_counts())
print(bike_df['holiday'].value_counts())
print(bike_df['weekday'].value_counts())
print(bike_df['workingday'].value_counts())
print(bike_df['weathersit'].value_counts())

del bike_df['instant']
del bike_df['casual']
del bike_df['registered']
del bike_df['holiday']
bike_df.shape

bike_df.dtypes

del bike_df['dteday']

bike_df['season']=bike_df['season'].astype('category')
bike_df['yr'] = bike_df['yr'].astype('category')
bike_df['mnth']=bike_df['mnth'].astype('category')
bike_df['weekday']=bike_df['weekday'].astype('category')
bike_df['workingday']=bike_df['workingday'].astype('category')
bike_df['weathersit']=bike_df['weathersit'].astype('category')

bike_df['cnt']=bike_df['cnt'].astype('float64')

bike_df.dtypes

bike_df.describe()

bike_df.isnull().sum()

#Check for outliers in data using boxplot
sns.boxplot(data=bike_df[['temp','atemp','windspeed','hum']])
fig=plt.gcf()
fig.set_size_inches(8,8)

sns.boxplot(bike_df['cnt'])

#since only 2 variables have outliers, so now removing these.
num_variables = ['windspeed','hum']
for i in num_variables:
    q75,q25=np.percentile(bike_df.loc[:,i],[75,25])
    iqr=q75-q25
    min=q25 -(1.5*iqr)
    max=q75 +(1.5*iqr)
    
    bike_df=bike_df.drop(bike_df[bike_df.loc[:,i]<min].index)
    bike_df=bike_df.drop(bike_df[bike_df.loc[:,i]>max].index)

bike_df.describe()


#Feature Selection
f, ax=plt.subplots(figsize=(7,5))
n_names = ['temp','atemp','hum','windspeed']
df = bike_df.loc[:,n_names]
sns.heatmap(df.corr(),mask=np.zeros_like(df.corr(),dtype=np.bool),
           cmap=sns.diverging_palette(220,10,as_cmap=True),ax=ax,annot = True)

cnames = ['season','workingday','weathersit','yr','mnth']
from scipy.stats import chi2_contingency
for i in cnames:
    print(i)
    chi2,p,dof,ex = chi2_contingency(pd.crosstab(bike_df['cnt'],bike_df[i]))
    print(p)
    
#dropping correlated variable
bike_df = bike_df.drop(['atemp'], axis=1)
bike_df.shape

bike_df['temp'] = bike_df['temp']*39
bike_df['hum'] = bike_df['hum']*100
bike_df['windspeed'] = bike_df['windspeed']*67

plt.hist(bike_df['cnt'],bins = 100)
plt.xlabel('Bike Rentals')
plt.ylabel('Frequency')

sns.boxplot(bike_df['yr'],bike_df['cnt'])
sns.boxplot(bike_df['mnth'],bike_df['cnt'])
sns.boxplot(bike_df['season'],bike_df['cnt'])
sns.boxplot(bike_df['weathersit'],bike_df['cnt'])
sns.boxplot(bike_df['weekday'],bike_df['cnt'])
sns.scatterplot(bike_df['temp'],bike_df['cnt'])
sns.scatterplot(bike_df['hum'],bike_df['cnt'])
sns.scatterplot(bike_df['windspeed'],bike_df['cnt'])



#Train_test Splitting : Simple Random Sampling as we are dealing with continuous variables
X = bike_df.iloc[:,:-1].values
Y = bike_df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)

#Multiple Linear Regression
lr_model = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()
lr_model.summary()

#Predict the results of test data
lr_predictions = lr_model.predict(X_test)

def MAPE(y_actual,y_pred):
    mape = np.mean(np.abs((y_actual - y_pred)/y_actual))
    return mape

MAPE(y_test,lr_predictions)*100

import pickle
pickle.dump(lr_model,open("model.pkl",'wb'))

#Loading model
model = pickle.load(open("model.pkl",'rb'))