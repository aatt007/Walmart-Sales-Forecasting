import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import seaborn as sns

df_tfs = pd.read_csv('walmart_fts.csv', parse_dates=['Date'])
#print(df_tfs)
print(df_tfs.dtypes)

#Convert categorical data to numerical data
df_tfs['IsHoliday'] = df_tfs['IsHoliday'].apply( lambda x: 1 if x==True else 0)
#print(df_tfs['IsHoliday'].unique())
print(df_tfs['Type'].unique())

le = preprocessing.LabelEncoder()
le.fit(df_tfs['Type'].unique())
df_tfs['Type'] = le.transform(df_tfs['Type'].to_list())
print(df_tfs['Type'].unique())

print(df_tfs['Temperature'].unique())
sns.jointplot(data=df_tfs, x="Temperature", y="Weekly_Sales")
#plt.show()

t_30_40 = df_tfs[( (df_tfs['Temperature'] > 30) & (df_tfs['Temperature'] <= 40 ))].Weekly_Sales.sum()
t_40_50 = df_tfs[( (df_tfs['Temperature'] > 40) & (df_tfs['Temperature'] <= 50 ))].Weekly_Sales.sum()
t_50_60 = df_tfs[( (df_tfs['Temperature'] > 50) & (df_tfs['Temperature'] <= 60 ))].Weekly_Sales.sum()
t_60_70 = df_tfs[( (df_tfs['Temperature'] > 60) & (df_tfs['Temperature'] <= 70 ))].Weekly_Sales.sum()
t_70_80 = df_tfs[( (df_tfs['Temperature'] > 70) & (df_tfs['Temperature'] <= 80 ))].Weekly_Sales.sum()
t_80_90 = df_tfs[( (df_tfs['Temperature'] > 80) & (df_tfs['Temperature'] <= 90 ))].Weekly_Sales.sum()
t_90_100 = df_tfs[( (df_tfs['Temperature'] > 90) & (df_tfs['Temperature'] <= 100 ))].Weekly_Sales.sum()
t_list=[t_30_40, t_40_50,t_50_60, t_60_70,t_70_80, t_80_90, t_90_100]
t_df = pd.Series(t_list, index=['t_30_40', 't_40_50','t_50_60', 't_60_70','t_70_80', 't_80_90', 't_90_100'])
#print(t_df)
#t_df.plot()
#plt.show()

t_df=t_df.sort_values(ascending=True)
#print(t_df)

t_df=t_df.to_frame()
t_df.reset_index(inplace=True)
#print(t_df)

df_tfs['t_rank']=np.nan
df_tfs.loc[((df_tfs['Temperature'] >90) & (df_tfs['Temperature'] <= 100)), 't_rank']=0
df_tfs.loc[( (df_tfs['Temperature'] > 30) & (df_tfs['Temperature'] <= 40 )),'t_rank'] = 1
df_tfs.loc[( (df_tfs['Temperature'] > 80) & (df_tfs['Temperature'] <= 90 )), 't_rank'] = 2
df_tfs.loc[( (df_tfs['Temperature'] > 40) & (df_tfs['Temperature'] <= 50 )), 't_rank'] = 3
df_tfs.loc[( (df_tfs['Temperature'] > 50) & (df_tfs['Temperature'] <= 60 )), 't_rank'] = 4
df_tfs.loc[( (df_tfs['Temperature'] > 70) & (df_tfs['Temperature'] <= 80 )), 't_rank'] = 5
df_tfs.loc[( (df_tfs['Temperature'] > 60) & (df_tfs['Temperature'] <= 70 )), 't_rank'] = 6
#print(df_tfs[['Temperature','t_rank']])

df_tfs['Year'] = df_tfs['Date'].dt.year
df_tfs['Month'] = df_tfs['Date'].dt.month

df_tfs['Day'] = df_tfs['Date'].dt.day
print(df_tfs[['Date', 'Year', 'Month', 'Day']])