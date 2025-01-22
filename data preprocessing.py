import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv(r"C:\Users\RAMYA\Downloads\Data.csv")

x = dataset.iloc[:,:-1].values #iloc - index location x = independent variable

y = dataset.iloc[:,3].values  # y = dependent variable

from sklearn.impute import SimpleImputer #impute fills missing value
#impute is a transformer
imputer = SimpleImputer()

imputer = imputer.fit(x[:,1:3]) 

x[:,1:3] = imputer.transform(x[:,1:3]) #strategy = mean or we can also change to median
#parameter tuning

from sklearn.preprocessing import LabelEncoder

LabelEncoder_x = LabelEncoder()

LabelEncoder_x.fit_transform(x[:,0])

x[:,0] = LabelEncoder_x.fit_transform(x[:,0])

labelencode_y = LabelEncoder()

y = labelencode_y.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x ,y , test_size = 0.2 , random_state = 0)
        
from sklearn.preprocessing import Normalizer  #feature scaling

sc_x = Normalizer()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)
