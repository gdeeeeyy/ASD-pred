import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.4, random_state=123)


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for column in X_train.columns:
    if X_train[column].dtype == 'object':
        X_train[column] = label_encoder.fit_transform(X_train[column])
        X_test[column] = label_encoder.transform(X_test[column])


from knn import KNN
clf=KNN(k=3)
clf.fit(X_train, y_train)
preds=clf.predict(X_test)

acc=np.sum(preds==y_test)/len(y_test)
print(acc)


def accuracy(y_true, y_pred):
    acc=np.sum(y_true==y_pred)/len(y_true)
    return acc

from logreg import LogisticRegression
reg=LogisticRegression(lr=0.0001, n_iters=1000000000)#run these iterations(didn't run this yet)
reg.fit(X_train, y_train)
preds=reg.predict(X_test)

print("Accuracy for 1000000000 iterations = ",accuracy(y_test, preds)*100)