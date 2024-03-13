import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split as splitData
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

df = pd.read_csv('data_park.data')

matrix = df.drop(['name','status'], axis=1)
labels = df['status']

scaler = StandardScaler()
matrix = scaler.fit_transform(matrix)

train_X, test_X, train_Y, test_Y = splitData(matrix, labels, test_size=0.2, random_state=42)

model = XGBClassifier(eval_metric='error')
model.set_params(early_stopping_rounds=10)
model.fit(train_X, train_Y, verbose=True, eval_set=[(test_X, test_Y)])

predict = model.predict(test_X)
score = accuracy_score(test_Y, predict)

print('Точность модели на тестовой выборке: %.02f' % score)
