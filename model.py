import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:\\Users\\Purva\\Desktop\\Cancer Prediction project\\DATA.csv")
df.head()
df.info()
df.describe()
df.columns

df["diagnosis"].value_counts()

df.drop("id",axis=1,inplace=True)

df.head()

df.shape
sns.countplot(df['diagnosis'],label="Count")

df.dtypes

df['diagnosis'] = df['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')

df.tail()
print(df.groupby('diagnosis').size())

X=df.drop("diagnosis",axis=1)
y=df["diagnosis"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.svm import SVC
svc_class = SVC()
svc_class.fit(X_train, y_train)

predictions1= svc_class.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test,predictions1))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions1))

import pickle
filename = 'breast_cancer_pred.pkl'
pickle.dump(svc_class,open(filename,'wb'))
loaded_model = pickle.load(open(filename,'rb'))
y_pred = loaded_model.predict(X_test)
print('Confusion matrix of svm model: \n', confusion_matrix(y_test, y_pred), '\n')
print('Accuracy of svm model = ', accuracy_score(y_test, y_pred))

