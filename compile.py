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
#print(df_fts)
df_tfs.to_csv('walmart_tfs.csv')
df_tfs=pd.read_csv('walmart_tfs.csv')

print(df_tfs.info)

