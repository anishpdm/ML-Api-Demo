from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('accident_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        year = int(data['year'])
        prediction = model.predict([[year]])

        return jsonify({'year': year, 'predicted_deaths': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
