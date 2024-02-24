import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression

data = pd.read_csv('accident_data.csv')

X = data[['year']]
y = data['deaths']

model = LinearRegression()

model.fit(X, y)
joblib.dump(model, 'accident_model.joblib')
