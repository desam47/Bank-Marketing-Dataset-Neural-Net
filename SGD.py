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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=7)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)


#Data scalling

scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)






# Start

def build_nn(activation='relu',dropout_rate=0.2, optimizer = 'SGD',X=X):
#initialize the Neural network
    model = Sequential()
    #Add the different layers of the ANN
    model.add(Dense(200, input_shape=(X_train.shape[1],), 
                    kernel_initializer='uniform')) #Input
    #model.add(Dropout(0.2))
    model.add(Dense(100, activation=activation, 
                    kernel_initializer='uniform')) #First hidden layer
    model.add(Dropout(dropout_rate))  #Droupout layer
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation=activation, 
                    kernel_initializer='uniform'))  #Second hidden layer
    #model.add(Dropout(0.2))
    model.add(Dense(25, activation=activation, 
                    kernel_initializer='uniform'))  #Third hidden layer
    model.add(Dense(10, activation=activation, 
                    kernel_initializer='uniform'))  #Fourth hidden layer
    model.add(Dense(2, activation='softmax')) #Output layer
    #model.add(Dropout(0.2))
    #model.add(Dense(2, activation='softmax'))
    model.summary()

# Compiling the model
    model.compile(optimizer= optimizer, loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Validation monitor
#monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')

# Save best model
#checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True)



#classifier.fit(X_train, y_train, batch_size =10, epochs = 20)
#Wrap the ANN 

model = KerasClassifier(build_fn = build_nn, epochs = 50, batch_size=25,validation_split=0.2, verbose=0)

#history= model.fit(X_train,y_train,validation_split=0.2, verbose=0)
history= model.fit(X_train,y_train)

#plt.plot(history.epoch, history.history['val_loss'], 'r',
 #       history.epoch, history.history['loss'], 'g')


# Model Loss over time
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# Model Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy of Model')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()


#k-fold cross-validation
#from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv=3)
test_acc=accuracies.mean()
print('Test accuracy:', test_acc*100)



pred = model.predict(X_test)
#pred = np.argmax(pred, axis =1)
y_compare = np.argmax(y_test, axis=1)
accsco = accuracy_score(y_compare, pred)
print("Final accuracy: {}".format(accsco*100))


# Plot a confusion matrix
y_compare = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_compare, pred)
np.set_printoptions(precision=2)
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels='NY', yticklabels='NY')

# Display the Classification Report

print('Classification report: \n\n',classification_report(y_compare, pred))


