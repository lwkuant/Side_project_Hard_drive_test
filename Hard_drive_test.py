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



### EDA

## column: date
print(hd['date'].head())
print(hd['date'].tail())
print(hd['date'].value_counts().shape) # it only contains Jan and April data

# are there duplicated drives i different months
from datetime import datetime 
hd['date'] = hd['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
print(hd['date'].head())   

hd['month'] = hd['date'].apply(lambda x: x.month)
print(hd['month'].value_counts())

month_1_serial = np.array(hd.ix[hd['month'] == 1, ]['serial_number'].value_counts().index)
month_4_serial = np.array(hd.ix[hd['month'] == 4, ]['serial_number'].value_counts().index)

print(len(np.intersect1d(month_1_serial, month_4_serial))) # most of the drives survive through Jan to April
double_month_drive = np.intersect1d(month_1_serial, month_4_serial)

#hd['double_month'] = hd['serial_number'].apply(lambda x: 1 if x in double_month_drive else 0)



## column: serial_number
print(hd['serial_number'].value_counts())
print(hd['serial_number'].value_counts().shape)


## column: model
print(hd['model'].value_counts())
print(hd['model'].value_counts().shape)
# extract the brand names
import re
brand_name = hd['model']
# split by the whitespace
brand_name = brand_name.apply(lambda x: re.split(r'\s', x)[0] if ' ' in x else x)
print(brand_name.value_counts()) # the brand names starting with ST is the same brand Seagate
brand_name = brand_name.apply(lambda x: x[:2] if x.startswith('ST') else x)
print(brand_name.value_counts())
# add the brand names as new column to the original dataset
hd['brand_name'] = brand_name


## column: capacity_bytes
print(hd['capacity_bytes'].value_counts())
# transform the capacity to the sylte we often encounter
import struct
def unmangle_float(x):
    return struct.unpack('>Q', struct.pack('>d', x))[0]
hd['capacity_bytes'] = hd['capacity_bytes'].map(unmangle_float)
print(hd['capacity_bytes'].value_counts())

## column: failure (label)
print(hd['failure'].value_counts()) # the number of failures is fairly small relatie to the normal type
print('failure proportion: %.6f %%' % (hd['failure'].value_counts()[1]/np.sum(hd['failure'].value_counts())*100))
### does this can be treated as a anomaly detection problem?

