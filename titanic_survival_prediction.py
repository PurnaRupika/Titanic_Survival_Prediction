import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

data=pd.read_csv('/content/train (1).csv')
data.head()

label=LabelEncoder()
data['Sex']=label.fit_transform(data['Sex'])
data['Embarked']=label.fit_transform(data['Embarked'])

data['Age'].fillna(data['Age'].median(),inplace=True)
data['Embarked'].fillna(data['Embarked'].mode(),inplace=True)

data.head()

data=data.drop(['Name','Ticket','Cabin'],axis=1)

X=data.drop('Survived',axis=1)
y=data['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy_score(y_pred,y_test)
print(f"Accuracy:{accuracy_score(y_pred,y_test) * 100:.2f}%")

print(classification_report(y_pred,y_test))

features=X.columns
importances=model.feature_importances_
sns.barplot(x=importances,y=features)
plt.title('feature Importance')
plt.show()



