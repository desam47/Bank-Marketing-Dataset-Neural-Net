#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, History

# Load dataset
df = pd.read_csv('bank-additional-full.csv')


# View a sample of the loaded data
df.sample(5)

df.info()

# Visualize the number of customer who subscribe the term deposite
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot('y', data=df)

df['y'].value_counts()


# Encode text values to indexes
le = preprocessing.LabelEncoder()

df['job'] = le.fit_transform(df['job'])
df['marital'] = le.fit_transform(df['marital'])
df['education'] = le.fit_transform(df['education'])
df['default'] = le.fit_transform(df['default'])
df['housing'] = le.fit_transform(df['housing'])
df['loan'] = le.fit_transform(df['loan'])
df['contact'] = le.fit_transform(df['contact'])
df['month'] = le.fit_transform(df['month'])
df['day_of_week'] = le.fit_transform(df['day_of_week'])
df['poutcome'] = le.fit_transform(df['poutcome'])
df['y'] = le.fit_transform(df['y'])


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


x, y = to_xy(df, "y")
x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))


# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=7)
print('Train set:', x_train.shape, y_train.shape)
print('Test set:', x_test.shape, y_test.shape)


# Defining the Multilayer Perceptron for Binary Classification
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# Model architecture and parameters
model.summary()


# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Validation monitor
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')

# Save best model
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True)

# Fitting/Training the model
history = model.fit(x_train, y_train, validation_split=0.2, callbacks=[monitor, checkpointer], verbose=0, epochs=1000, batch_size=64)

# Load weights from best model
model.load_weights('best_weights.hdf5')

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


pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
y_compare = np.argmax(y_test, axis=1)
score = metrics.accuracy_score(y_compare, pred)
print("Final accuracy: {}".format(score*100))


# Plot a confusion matrix

cm = confusion_matrix(y_compare, pred)
np.set_printoptions(precision=2)
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels='NY', yticklabels='NY')


# Display the Classification Report

print('Classification report: \n\n',classification_report(y_compare, pred))
