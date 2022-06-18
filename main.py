import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load data
features=pd.read_csv('features.csv', parse_dates=['Date'])
print(features.info())
#print(features.dtypes)

stores=pd.read_csv('stores.csv')
#print(stores.dtypes)
print(stores.info())

test=pd.read_csv('test.csv', parse_dates=['Date'])
#print(test.dtypes)

train=pd.read_csv('train.csv', parse_dates=['Date'])
#print(train)
#print(train.dtypes)

# merge data
df_ft=features.merge(train, on=['Store', 'Date', 'IsHoliday'], how='inner')
#print(df_ft)

df_tfs=df_ft.merge(stores, on=['Store'], how='inner')
print(df_tfs)

print(df_tfs.isnull().sum())
df_tfs.loc[df_tfs['MarkDown1'].isnull(), 'MarkDown1']=0
df_tfs.loc[df_tfs['MarkDown2'].isnull(), 'MarkDown2']=0
df_tfs.loc[df_tfs['MarkDown3'].isnull(), 'MarkDown3']=0
df_tfs.loc[df_tfs['MarkDown4'].isnull(), 'MarkDown4']=0
df_tfs.loc[df_tfs['MarkDown5'].isnull(), 'MarkDown5']=0
print(df_tfs.isnull().sum())

#EDA
print(df_tfs.groupby('Date')['Weekly_Sales'].sum())
#df_fts.groupby('Date')['Weekly_Sales'].sum().plot()
#plt.show()

import seaborn as sns
sns.distplot(df_tfs.groupby('Date')['Weekly_Sales'].sum())
#plt.show()
'''
sns.distplot(df_tfs.groupby('Date')['MarkDown1'].sum())
sns.distplot(df_tfs.groupby('Date')['MarkDown2'].sum())
sns.distplot(df_tfs.groupby('Date')['MarkDown3'].sum())
sns.distplot(df_tfs.groupby('Date')['MarkDown4'].sum())
sns.distplot(df_tfs.groupby('Date')['MarkDown5'].sum())
'''
#sns.countplot(df_tfs.Type)
'''
sns.distplot(df_tfs[df_tfs.Type=='A'].groupby('Date')['Weekly_Sales'].sum())
sns.distplot(df_tfs[df_tfs.Type=='B'].groupby('Date')['Weekly_Sales'].sum())
sns.distplot(df_tfs[df_tfs.Type=='C'].groupby('Date')['Weekly_Sales'].sum())
'''
plt.figure()
df_fts[df_tfs.Type=='A'].groupby('Date')['Weekly_Sales'].sum().plot(x_compat=True)
df_fts[df_tfs.Type=='B'].groupby('Date')['Weekly_Sales'].sum().plot(x_compat=True)
df_fts[df_tfs.Type=='C'].groupby('Date')['Weekly_Sales'].sum().plot(x_compat=True)

plt.show()