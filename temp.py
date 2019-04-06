import time
toe=time.time()

#Loading necessory dependencies
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


#importing dataset
data = pd.read_csv('bank-additional-full.csv')
#print(data.head())


#Combining some education classes and call it basic
data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

#Dropping some data which is not necessory according to dataset, or is not contributing in any prediction.
data = data.drop(['marital'],axis=1)
data = data.drop(['day_of_week'],axis=1)

#changing object data type to category
data['job'] = data['job'].astype('category')
data['education'] = data['education'].astype('category')
data['default'] = data['default'].astype('category')
data['housing'] = data['housing'].astype('category')
data['loan'] = data['loan'].astype('category')
data['contact'] = data['contact'].astype('category')
data['month'] = data['month'].astype('category')
data['poutcome'] = data['poutcome'].astype('category')
data['y'] = data['y'].astype('category')


#converting yes and no to 1 and 0
data['y'] = data['y'].replace({'no':0, 'yes':1})

#converting categorical dataset to numerical.
data_dum = pd.get_dummies(data, columns=['job','default','housing','contact','education','loan','month','poutcome'])

# Split train and test dataset
X = data_dum.drop('y',axis=1)
y = data_dum['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling dataset
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

#using SVC classifier
clf=SVC()
svm = clf.fit(X_train_std, y_train)
svm_predicted = svm.predict(X_test_std)

#Accuracy
Accuracy= accuracy_score(y_test, svm_predicted)

#Runtime
tac=time.time()

tic=tac-toe
print("Accuracy is: {0:.2f} %".format((Accuracy)*100))
print("Runtime: {} seconds".format(int(tic)))