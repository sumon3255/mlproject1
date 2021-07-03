from sys import prefix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Load Csv data

df = pd.read_csv("heart.csv")
# print(df.head())
# #shape
# print(df.shape)
#info
# print(df.info())
#Checking mising value
df1=df.isnull().sum()
# print(df1)

#statical measure of data
df1 = df.describe()
# print(df1)

#cheacking distription of target variable
target = df["target"].value_counts()
# print(target)

#spleting feature and target
X = df.drop(columns="target",axis=1)
Y = df["target"]

#test and train

X_train , X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y , random_state=2) #stratify spilit avoid all 1 or 0


#Model data
model =  LogisticRegression()

dfmodel = model.fit(X_train,Y_train)


#Train acuuracy
X_train_pred = model.predict(X_train)
train_data_acuur = accuracy_score(X_train_pred,Y_train)

print("Acuurcay Train data is: ",train_data_acuur )

#Test_data accuracy
X_test_pred = model.predict(X_test)
test_data_acuur = accuracy_score(X_test_pred,Y_test)

print("Acuurcay TEST DATA is: ",test_data_acuur )

#Building a predicted system
input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)
#change input data in numpy array
input_num = np.asarray(input_data)

#reshape that we want to add one only
input_data_resap = input_num.reshape(1,-1)

prediction = model.predict(input_data_resap)
# print(prediction)

if (prediction == 0):
    print("The person doesnot have a heart Disease")
else:
    print("The person  have a heart Disease")





