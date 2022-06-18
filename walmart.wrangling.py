import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import seaborn as sns
import datetime as dt

df_tfs = pd.read_csv('walmart_fts.csv', parse_dates=['Date'])
#print(df_tfs)
#print(df_tfs.dtypes)

#Convert categorical data to numerical data
df_tfs['IsHoliday'] = df_tfs['IsHoliday'].apply( lambda x: 1 if x==True else 0)
#print(df_tfs['IsHoliday'].unique())
#print(df_tfs['Type'].unique())

le = preprocessing.LabelEncoder()
le.fit(df_tfs['Type'].unique())
df_tfs['Type'] = le.transform(df_tfs['Type'].to_list())
#print(df_tfs['Type'].unique())

df_tfs['Year'] = df_tfs['Date'].dt.year
df_tfs['Month'] = df_tfs['Date'].dt.month
df_tfs['Week']= df_tfs['Date'].dt.week
df_tfs['Day'] = df_tfs['Date'].dt.day
#print(df_tfs[['Date', 'Year', 'Month', 'Day']])

print(df_tfs['Year'].unique())
print(df_tfs[df_tfs['IsHoliday']==1] ['Week'].unique())

'''
le=preprocessing.LabelEncoder()
x=df_tfs[df_tfs['IsHoliday']==1] ['Week'].unique()
x=x.tolist()
x.append(0)
print(x)
le.fit(x)
df_tfs['Holiday_Type']=df_tfs['Week'].apply(lambda x:0 if x not in [6,36,47,52] else x)
df_tfs['Holiday_Type']=le.transform(df_tfs['Holiday_Type'].tolist())
print(df_tfs['Holiday_Type'].unique())

cor=df_tfs[['IsHoliday', 'Holiday_Type','Weekly_Sales']].corr()
sns.heatmap(cor, annot=True)
plt.show()
'''

def weeks_pre_holiday(x):

    diff_list = []
    if x['Year'] == 2010:
        for d in [dt.datetime(2010, 12, 31), dt.datetime(2010, 11, 26),
                  dt.datetime(2010, 9, 10), dt.datetime(2010, 2, 12)]:
            d_diff = d- x['Date']
            if d_diff.days < 0:
                diff_list.append(0)
            else:
                diff_list.append(d_diff.days / 7)
            return int(min(diff_list))

    if x['Year'] == 2011:
        for d in [dt.datetime(2010, 12, 30), dt.datetime(2010, 11, 25),
                  dt.datetime(2010, 9, 9), dt.datetime(2010, 12, 11)]:
            d_diff = d - x['Date']
            if d_diff.days< 0:
                diff_list.append(0)
            else:
                diff_list.append(d_diff.days / 7)
            return int(min(diff_list))

    if x['Year'] == 2012:
        for d in [dt.datetime(2010, 12, 28), dt.datetime(2010, 11, 23),
                  dt.datetime(2010, 9, 7), dt.datetime(2010, 2, 10)]:
            d_diff = d - x['Date']
            if d_diff.days < 0:
                diff_list.append(0)
            else:
                diff_list.append(d_diff.days / 7)
            return int(min(diff_list))



df_tfs['weeks_pre_holiday'] = df_tfs.apply(weeks_pre_holiday, axis=1)
print(df_tfs)

cor =df_tfs[['Weekly_Sales', 'weeks_pre_holiday']].corr()
sns.heatmap(cor, annot= True, cmap=plt.cm.Reds)
plt.show()
