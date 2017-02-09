# -*- coding: utf-8 -*-
"""
Hard Drive Test Data 
"""

import os 
os.getcwd()
os.chdir(r'D:\Dataset\kaggle_Hard_Drive_Test_Data')

import pandas as pd 
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt 
%matplotlib inline 
import seaborn as sns

###  overview on the dataset 
hd = pd.read_csv('harddrive.csv', encoding='utf-8')

print('the shape:')
print(hd.shape)

print('the information:')
print(hd.info())

print('if each column has NAs:')
print(hd.isnull().any())
print('how many?')
print(np.sum(hd.isnull().any()))
print('proportion?')
print(np.sum(hd.isnull().any())/hd.shape[1])
print('how many NAs per column?')
print((np.sum(pd.isnull(hd))/hd.shape[0]*100))


### split the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(hd.ix[:, ~(hd.columns == 'failure')], hd['failure'],
                                                    test_size=0.25, random_state=100, stratify=hd['failure'])

hd_train = pd.concat([X_train, y_train], axis=1)
del (X_train, y_train)


### EDA (on training dataset)

hd_train.index = list(range(len(hd_train)))

## column: date
print(hd_train['date'].head())
print(hd_train['date'].tail())
print(hd_train['date'].value_counts().shape) # it only contains Jan and April data

# are there duplicated drives i different months
from datetime import datetime 
hd_train['date'] = hd_train['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
print(hd_train['date'].head())   

#hd['month'] = hd['date'].apply(lambda x: x.month)
#print(hd['month'].value_counts())

#month_1_serial = np.array(hd.ix[hd['month'] == 1, ]['serial_number'].value_counts().index)
#month_4_serial = np.array(hd.ix[hd['month'] == 4, ]['serial_number'].value_counts().index)

#print(len(np.intersect1d(month_1_serial, month_4_serial))) # most of the drives survive through Jan to April
#double_month_drive = np.intersect1d(month_1_serial, month_4_serial)

#hd['double_month'] = hd['serial_number'].apply(lambda x: 1 if x in double_month_drive else 0)


## column: serial_number
print(hd_train['serial_number'].value_counts())
print(hd_train['serial_number'].value_counts().shape)


## column: model
print(hd_train['model'].value_counts())
print(hd_train['model'].value_counts().shape)
# extract the brand names
import re
brand_name = hd_train['model']
# split by the whitespace
brand_name = brand_name.apply(lambda x: re.split(r'\s', x)[0] if ' ' in x else x)
print(brand_name.value_counts()) # the brand names starting with ST have the same brand Seagate
brand_name = brand_name.apply(lambda x: x[:2] if x.startswith('ST') else x)
print(brand_name.value_counts())
# add the brand names as new column to the original dataset
hd_train['brand_name'] = brand_name
print(hd_train['brand_name'].value_counts())


## column: capacity_bytes
print(hd_train['capacity_bytes'].value_counts())
# transform the capacity to the sylte we often encounter
import struct
def unmangle_float(x):
    return struct.unpack('>Q', struct.pack('>d', x))[0]
hd_train['capacity_bytes'] = hd_train['capacity_bytes'].map(unmangle_float)
print(hd_train['capacity_bytes'].value_counts())


## column: failure (label)
print(hd_train['failure'].value_counts()) # the number of failures is fairly small relatie to the normal type
print('failure proportion: %.6f %%' % (hd_train['failure'].value_counts()[1]/np.sum(hd['failure'].value_counts())*100))
### does this can be treated as a anomaly detection problem?


## Visualization: failure by date
date_group = hd_train.groupby(['date'])
print(date_group['failure'].sum())

date_li = np.array(list(date_group['failure'].sum().index))
failure_li = date_group['failure'].sum().values

fig, axes = plt.subplots(figsize=[30, 10])
sns.set_style('whitegrid')

axes.tick_params(labelsize=15)
axes.plot(date_li, failure_li, '--o', color='#2E8B57', linewidth=0.5)
axes.set_ylim(np.min(failure_li)-5, np.max(failure_li)+5)
axes.set_xlabel('Time', fontsize=20)
axes.set_ylabel('Count', fontsize=20)
axes.set_title('Count of Failure by Time', fontsize=20)
fig.tight_layout()
# there is a trend that the tests in April produce more failures

## Visualization: Failure ratio by brand
brand_group = hd_train.groupby(['brand_name'])
print(brand_group['failure'].sum())
print(brand_group['failure'].count())

brand_failure_tb = (brand_group['failure'].sum()/brand_group['failure'].count()).sort_values(ascending=False)
brand_ratio = brand_failure_tb.values
brand_name = list(brand_failure_tb.index)

fig, axes = plt.subplots(figsize=[15, 15])
axes.bar(range(len(brand_name)), brand_ratio*1e2, color='#2E8B57', width=0.8)
axes.tick_params(labelsize=15)
axes.set_xticks(np.arange(len(brand_name))+0.4)
axes.set_xticklabels(brand_name, fontsize=15)
axes.set_xlabel('Brand', fontsize=20)
axes.set_ylabel('Ratio', fontsize=20)
axes.set_title('Failure Ratio for Each Brand', fontsize=30, loc='center')

## NAs preprocessing 

print('if each column has NAs:')
print(hd_train.isnull().any())
print('how many?')
print(np.sum(hd_train.isnull().any()))
print('proportion?')
print(np.sum(hd_train.isnull().any())/hd_train.shape[1])
print('how many NAs per column?')
print((np.sum(pd.isnull(hd_train))/hd_train.shape[0]*100))

# filter out the NAs remove columns with NAs)
hd_train_nonna = hd_train.ix[:, ~(hd_train.isnull().any())]
print(hd_train_nonna.shape)
print(hd_train_nonna.columns)

hd_train_nonna_feature = hd_train_nonna.drop(['date', 'serial_number',
                                              'model', 'failure', 'brand_name'], axis=1)
print(hd_train_nonna_feature.shape)


## visualization of the distribution of each feature
#fig = plt.figure(figsize=[10, 500])
#for (i, col) in enumerate(hd_train_nonna_feature.columns, start=1):
#    axes = fig.add_subplot(27, 1, i)
#    axes.tick_params(labelsize=10)    
#    sns.distplot(hd_train_nonna_feature[col].values)
#    axes.set_ylabel('Density', fontsize=15)
#    axes.set_title(col)
#    fig.tight_layout()

# the graph above cannot be shown, we turn to the statistical test


## statistical test on features and targets
# Kruskal Wallis H-test

feature_name_li = hd_train_nonna_feature.columns

from scipy.stats import kruskal

kw_test_val_dict = {}
for col in feature_name_li:
    kw_test_val_dict[col] = list(kruskal(hd_train_nonna_feature[col][hd_train_nonna['failure']==0],
                                hd_train_nonna_feature[col][hd_train_nonna['failure']==1]))

print(kw_test_val_dict)
# the result is horrible


## correlation between features
corr_mat = hd_train_nonna_feature.corr()

plt.figure(figsize=[20, 20])
sns.heatmap(corr_mat, square=True, annot=True, cmap="RdBu")
#plt.title('Correlations between each Feature', fontsize=20)
plt.suptitle('Correlations between each Feature', fontsize=20, y=0.94, horizontalalignment='center')
# find some features with almost all of the values are fairly small (very close to 0)
# we will try to remove them 

# remove the columns with those small values
for col in hd_train_nonna_feature.columns:
    if np.max(hd_train_nonna_feature[col].values) < 1e-10:
        hd_train_nonna_feature.drop([col], axis=1, inplace=True)

print(hd_train_nonna_feature.shape)

# correlation again
cor_mat = np.round(hd_train_nonna_feature.corr(), 2)

plt.figure(figsize=[20, 20])
sns.heatmap(cor_mat, square=True, annot=True, cmap="RdBu")
#plt.title('Correlations between each Feature', fontsize=20)
plt.suptitle('Correlations between each Feature', fontsize=20, y=0.94, horizontalalignment='center')


pair_li = []

for i in cor_mat.index:
    for j in cor_mat.index:
        if i == j:
            continue
        else:
            if np.round(np.abs(cor_mat.ix[i, j]), 1) >= 0.5:
                pair_li.append(tuple(sorted([i, j])))
            else:
                continue

print('\n')                
pair_li = list(set(pair_li))
print('Features pairs that have high correlations:')
print(pair_li)

intersect_li = []

for i in pair_li:
    temp_li = []
    temp_li.append(i)
    for j in pair_li:
        if i == j:
            continue
        else:
            if len(np.intersect1d(i, j)) != 0:
                temp_li.append(j)
            else:
                continue
    temp_li = tuple(sorted(temp_li))
    intersect_li.append(temp_li)

intersect_li = list(set(intersect_li))
print('\n')
print('Features pairs with high correlations that have overlapping values')
for i, pair in enumerate(intersect_li, start=1):
    print(str(i)+':', pair)