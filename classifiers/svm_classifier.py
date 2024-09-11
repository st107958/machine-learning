from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

df = pd.read_csv('data_set.csv')

target = df['Class']
features = df.drop(columns=['Class'])

train_data, temp_data, train_target, temp_target = train_test_split(features, target, test_size=0.4, random_state=42)
val_data, test_data, val_target, test_target = train_test_split(temp_data, temp_target, test_size=0.5, random_state=42)

# clf = SVC(kernel='linear', C=0.5)
clf = SVC(kernel='poly', C=0.5, degree=1, gamma=1, coef0=0.0)

#с - регуляризации (если меньше то больше обобщает)

clf.fit(train_data, train_target)

y_pred1 = clf.predict(val_data)
acc1 = accuracy_score(val_target, y_pred1)
print(acc1)

y_pred2 = clf.predict(test_data)
acc2 = accuracy_score(test_target, y_pred2)
print(f'Accuracy:{acc2:.5f}')

