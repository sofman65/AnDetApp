from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from runOCSVM import runOCSVM

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    train_data = pd.DataFrame(request.json['train_data'])
    test_data = pd.DataFrame(request.json['test_data'])
    # Dummy prediction logic (replace with actual function calls)
    metrics = {'F1 Score': 0.95, 'Precision': 0.96, 'Recall': 0.94}
    return jsonify(metrics)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
