from sklearn.model_selection import train_test_split,learning_curve,cross_val_score,ShuffleSplit
from sklearn.tree import DecisionTreeClassifier,plot_tree
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

dataset=pd.read_csv("Titanic-Dataset.csv")
for col in dataset.select_dtypes(include="number").columns:
    dataset[col].fillna(dataset[col].mean(), inplace=True)

x=dataset[['Pclass','Age','SibSp','Parch','Fare']]
y=dataset['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(max_depth=3,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy score:{accuracy}")

plot_tree(model,filled=True)
plt.show()
feature_importances=model.feature_importances_
feature_names=x.columns
for feature, importance in zip(feature_names, feature_importances):
    print(f"Feature: {feature}, Importance: {importance}")

train_sizes, train_scores, test_scores = learning_curve(
    model, x_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)


plt.figure(figsize=(10, 5))
plt.plot(train_sizes,np.mean(train_scores,axis=1),label='Training score')
plt.plot(train_sizes,np.mean(test_scores,axis=1),label='Test score')
plt.xlabel('Train size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#cross validation
cross_val=cross_val_score(model,x_train,y_train,cv=5,scoring='accuracy')
print(f"Cross validation scores:{cross_val}")
#checking overfitting
print("Avg Training Score:", np.mean(train_scores, axis=1))
print("Avg Test Score:", np.mean(test_scores, axis=1))


y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_score = accuracy_score(y_train, y_train_pred)#checking train score
test_score = accuracy_score(y_test, y_test_pred)#checking test score

print(f"Train Score: {train_score}")
print(f"Test Score: {test_score}")
random_forest_model=RandomForestClassifier(n_estimators=100,random_state=42)
random_forest_model.fit(x_train,y_train)
r_pred=random_forest_model.predict(x_test)
#cross validation
cross_val=cross_val_score(random_forest_model,x_train,y_train,cv=5,scoring='accuracy')
print(f"Cross validation scores:{cross_val}")




