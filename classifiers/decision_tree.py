from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

df = pd.read_csv('data_set.csv')

target = df['Class']
features = df.drop(columns=['Class'])

train_data, temp_data, train_target, temp_target = train_test_split(features, target, test_size=0.4, random_state=42)
val_data, test_data, val_target, test_target = train_test_split(temp_data, temp_target, test_size=0.5, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', max_depth=5)


clf.fit(train_data, train_target)

y_pred1 = clf.predict(val_data)
print(accuracy_score(val_target, y_pred1))


y_pred2 = clf.predict(test_data)
accuracy = accuracy_score(test_target, y_pred2)
print(f'Accuracy: {accuracy:.5f}')

