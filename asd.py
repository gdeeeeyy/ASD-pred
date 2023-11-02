import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
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


from knn import KNN
clf=KNN(k=3)
clf.fit(X_train, y_train)
preds=clf.predict(X_test)

acc=np.sum(preds==y_test)/len(y_test)
print(acc)


def accuracy(y_true, y_pred):
    acc=np.sum(y_true==y_pred)/len(y_true)
    return acc*100

i=10000#final val at 1000000000
while(i<1000000000):
    stime=time.time()
    reg=LogisticRegression(lr=0.0001, n_iters=i)
    reg.fit(X_train, y_train)
    preds=reg.predict(X_test)
    print(f"Accuracy for {i} iterations = {accuracy(y_test, preds)*100}")
    etime=time.time()
    t=etime-stime
    print(f"Time taken for {i} iterations is {t} seconds")
    print("\n")
    i*=10
print("Mass dhaan")


stime=time.time()
nb=NaiveBayes()
nb.fit(X_train, y_train)
predictions=nb.predict(X_test)

print(f"Naive Bayes classification accuracy: {accuracy(y_test, predictions)}")
etime=time.time()
t=etime-stime
print(f"Time taken for Naive Bayes algorithm is {t} seconds")

clf=SVM(lr=0.001, _lambda=0.01, n_iters=1000)
clf.fit(X_train, y_train)
predictions=clf._predict(X_test)


clf=RandomForest(n_trees=3)
clf.fit(X_train1, y_train1)

y_pred=clf.predict(X_test1)
print(f"Accuracy: {accuracy(y_test, y_pred)}")
