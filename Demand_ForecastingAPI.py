import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify
from math import sqrt
import io

app = Flask(__name__)

@app.route('/demandprediction', methods=['POST'])
def predict_demand():
  """
  API endpoint to predict demand using ARIMA model.

  Returns:
      JSON: A JSON object containing formatted predictions with demand percentages.
  """

  # Read the CSV file from the request body
  try:
      data = request.files['data']
      df = pd.read_csv(io.BytesIO(data.read()))
  except Exception as e:
      return jsonify({'error': str(e)}), 400

  # Preprocess data (if needed)

  # Train ARIMA model
  model = ARIMA(df['Sales'], order=(5, 0, 5))
  model_fit = model.fit()

  # Make predictions
  predictions = model_fit.predict(start=0, end=len(df)-1, typ='levels')

  # Evaluate model performance
  error = mean_squared_error(df['Sales'], predictions)
  rmse = sqrt(error)

  # Prepare the formatted response data
  formatted_response = []
  for i in range(len(predictions)):
      month_name = df.index[i]  # Assuming the index contains month names
      prediction_value = predictions[i]
      demand_percentage = round(prediction_value / df['Sales'].iloc[i] * 100, 2)
      formatted_response.append({
          "month": month_name,
          "prediction_value": prediction_value,
          "demand_percentage": demand_percentage  # Added demand percentage
      })

  return jsonify({
      'predictions': formatted_response,
      'rmse': rmse
  })

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
