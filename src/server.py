from flask import Flask, jsonify
import os
import pandas as pd
import traceback
import joblib
from functions import explainn, get_employee_shap_values

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'empid_test_set.csv')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'xgb_model.joblib')

model = joblib.load(MODEL_PATH)
dataset = pd.read_csv(DATA_PATH)

@app.route('/')
def home():
    return jsonify({'message': 'Attrition explanation API is running', 'routes': ['/global', '/local']})

@app.route('/global')
def global_explanations():
    try:
        X_test = dataset.drop(columns=['Attrition'])
        class_names = ['No Attrition', 'Attrition']
        explanations = explainn(model, X_test, feature_names=X_test.columns.tolist(), class_names=class_names)
        out = {str(k): [[feat, float(w)] for feat, w in v] for k, v in explanations.items()}
        return jsonify({'status': 'success', 'global_explanations': out})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/local')
def local_explanations():
    try:
        employee_id = 1780761
        shap_values = get_employee_shap_values(model, employee_id, dataset)
        return jsonify({'status': 'success', 'employee_id': employee_id, 'local_shap': shap_values})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
