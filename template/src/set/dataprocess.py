## CREATE CLEAN DATA


import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 256)

#===download data===============================================
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#print(train_data.head())   # check data

test_id = test_data["PassengerId"]    # for submission
#===============================================================

===extract important data=====================================
labels = ["Pcclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
train_data = train_data[labels+["Survived"]]
test_data = test_data[labels]

#print(train_data.head())   # check data
#===============================================================

#===fill nulls with average=====================================

print(train_data.info())    # check nulls
print(test_data.info())     # check nulls

#Age
age_mean = train_data["Age"].mean()
train_data["Age"] = train_data["Age"].fillna(age_mean)
test_data["Age"] = test_data["Age"].fillna(age_mean)

#Fare
fare_mean = train_data["Fare"].mean()
train_data["Fare"] = train_data["Fare"].fillna(fare_mean)
test_data["Fare"] = test_data["Fare"].fillna(fare_mean)

#Embarked
embarked_mean = train_data["Embarked"].mode()
train_data["Embarked"] = train_data["Embarked"].fillna(embarked_mean)
test_data["Embarked"] = test_data["Embarked"].fillna(embarked_mean)
#===============================================================


#===get dummy variables=========================================
cat_label = ["Sex", "class", "Embarked"]
train_data = pd.get_dummies(train_data, columns=cat_label)
test_data = pd.get_dummies(test_data, columns=cat_label)

#print(train_data.head())   # check data
#===============================================================


#===split data into inputs and outputs=========================================
x_train = train_data.drop(labels=["Survived"], axis=1)  #non-destructive
t_train= train_data["Survived"]
x_test = test_data

#! みんなのKaggle

