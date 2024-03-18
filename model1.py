import pandas as pd #used for loading the dataset
import numpy as np #used for performing an array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset=pd.read_csv('Subscription_data.csv')

print(dataset.shape)
print(dataset.head(5))

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=0)


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()#loading the algorithm
model.fit(x_train,y_train)#training

age= int(input("Enter new customer's age: "))
sal= int(input("Enter new customer's salary: "))

newCust=[[age,sal]]

result=model.predict(sc.transform(newCust))

print(result)

if result==1:
  print("Customer will subscribe the streaming platform")
else:
  print("Customer won't subscribe the streaming platform")


y_pred=model.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
print("Accuracy of the model: {0}%".format(accuracy_score(y_test, y_pred)*100))
