import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import confusion_matrix, precision_recall_curve
from knn import KNN
from logreg import LogisticRegression
from nb import NaiveBayes
from svm import SVM
from random_forest import RandomForest

csv=pd.read_csv(r"asd.csv")
data=pd.DataFrame(csv)

#Applying one hot encoding
n=[]
for i in data["Ethnicity"]:
    n.append(i)
dat=set(n)

fdat={}
i = data[data['Ethnicity'] == 'middle eastern'].index
for i in dat:
    k=data[data['Ethnicity'] == i].index
    fdat[i]=k.to_list()
for i in dat:
    data[i]=None
for i,j in fdat.items():
    for k in j:
        data.loc[k, i]=1

data=data.fillna(0)
data=data.drop('Ethnicity', axis=1)
rep={'Yes':1, 'No': 0, 'yes':1, 'no':0, 'f':1, 'm':0}
data=data.replace(rep)

data.to_csv('finalasd.csv', index=False)

csvf=pd.read_csv(r"finalasd.csv")
df=pd.DataFrame(csvf)
X=df.drop(['Case_No', 'ASD', 'Who completed the test'], axis=1)
y=df['ASD']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=123)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for column in X_train.columns:
    if X_train[column].dtype == 'object':
        X_train[column] = label_encoder.fit_transform(X_train[column])
        X_test[column] = label_encoder.transform(X_test[column])

X_train1=X_train.values
y_train1=y_train.values
X_test1=X_test.values
y_test1=y_test.values

def accuracy(y_true, y_pred):
    acc=np.sum(y_true==y_pred)/len(y_true)
    return acc*100

s1time=time.time()
clf=KNN(k=3)
clf.fit(X_train1, y_train1)
preds=clf.predict(X_test1)
acc=accuracy(y_test1, preds)
print(f"Accuracy for KNN is {acc}")
e1time=time.time()
t1=e1time-s1time
print(f"Time taken for KNN is {t1} seconds")
print(confusion_matrix(y_test, preds))
print("\n")

s2time=time.time()
clf=SVM(lr=0.001, _lambda=0.01, n_iters=1000)
clf.fit(X_train1, y_train1)
predictions=clf._predict(X_test1)
print(f"SVM accuracy: {accuracy(y_test1, predictions)}")
e2time=time.time()
t2=e2time-s2time
print(f"Time taken for SVM is {t2} seconds")
print(confusion_matrix(y_test, predictions))
print("\n")


s3time=time.time()
clf=RandomForest(n_trees=3)
clf.fit(X_train1, y_train1)
y_pred1=clf.predict(X_test1)
print(f"RFC accuracy: {accuracy(y_test, y_pred1)}")
e3time=time.time()
t3=e3time-s3time
print(f"Time taken for RFC is {t3} seconds")
print(confusion_matrix(y_test, y_pred1))
print("\n")

stime=time.time()
nb=NaiveBayes()
nb.fit(X_train1, y_train1)
prediction=nb.predict(X_test1)
print(f"Naive Bayes classification accuracy: {accuracy(y_test1, prediction)}")
etime=time.time()
t=etime-stime
print(f"Time taken for Naive Bayes algorithm is {t} seconds")
print(confusion_matrix(y_test, prediction))
print("\n")

i=10000000#start value 10000 
# while(i<1000000000):
stime=time.time()
reg=LogisticRegression(lr=0.0001, n_iters=i)
reg.fit(X_train, y_train)
pred=reg.predict(X_test)
print(f"Accuracy for {i} iterations = {accuracy(y_test, preds)}")
etime=time.time()
t=etime-stime
print(f"Time taken for {i} iterations is {t} seconds")
print(confusion_matrix(y_test, pred))
print("\n")
# i*=10



"""
Logistic Regression-Works
KNN-Works
Naive Bayes-Doesn't
SVM-Works
RFC-Works(enakku innum doubt dhaan maams)
"""