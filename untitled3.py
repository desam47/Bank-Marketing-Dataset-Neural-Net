#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 21:52:12 2019

@author: dipesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://github.com/duaraanalytics/bankmarketing/blob/master/Analyzing%20Employee%20Churn%20with%20Keras.ipynb

"""
Created on Wed Mar  6 08:51:19 2019

@author: dipesh
"""

# Import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

#from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from keras.layers import Dense, Dropout
from keras.models import Sequential
#from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.wrappers.scikit_learn import KerasClassifier

# Load dataset
df = pd.read_csv('bank-additional-full.csv')


# View a sample of the loaded data
df.sample(5)

df.info()

pd.set_option('display.max_columns', 200)
df.describe(include='all')

#check for any missing values
df.apply(lambda x: sum(x.isnull()),axis=0)

#From the Feature selection of our last project it is decided to remove the 'euribor3m' and 'duration' variable
df = df.drop('euribor3m', axis=1)

df = df.drop('duration', axis=1)

df.describe(include='all')

df.info()

labels = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']
for l in labels:
    print(l+":")
    print(df[l].value_counts())
    print('-------------------------------')

# Visualize the number of customer who subscribe the term deposite
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot('y', data=df)

df['y'].value_counts()

# Encode text values to indexes

l_encoder = LabelEncoder()
labels = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']
for l in labels:
    df[l] = l_encoder.fit_transform(df[l])

df.sample(5)

# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)


X, y = to_xy(df, "y")
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))


# Split dataset into train, validation and test sets
#X_trainval, X_test, y_trainval, y_test= train_test_split(X, y, test_size = 0.15, random_state = 1)
#X_train, X_valid, y_train, y_valid= train_test_split(X_trainval, y_trainval, test_size = 0.15, random_state = 1)
#print('Train set:', X_trainval.shape, y_trainval.shape)
#print('Test set:', X_test.shape, y_test.shape)
#print('Validation set:', X_valid.shape, y_valid.shape)


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

#Data scalling

scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(100, input_dim=18, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_test, y_test, epochs=150, batch_size=25, validation_split=0.2, verbose=2, shuffle=True)
# evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

from ann_visualizer.visualize import ann_viz;

ann_viz(model, title="Bank Marketing:Neaural Net")