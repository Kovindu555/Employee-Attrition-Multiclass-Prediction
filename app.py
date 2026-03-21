from flask import Flask, render_template, Response, jsonify, request
import numpy as np
import pandas as pd
import risk_profiling_be as rpb
import os
import feature_interpretation_be as fib
import strategic_analysis_be as sab
# ------------------------------ Library Corner -----------------------------

app = Flask(__name__)

UPLOAD_FOLDER = 'test_csv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------- App Functions --------------------------------
@app.route('/')
def index():
    return render_template('home-page/index.html')


# file upload from the Browse button
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400

    filename = os.path.basename(file.filename)
    saved_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(saved_path)
    dataset_name = filename[:-4]

    try:
        df = rpb.sample_predict(dataset_name)
        records = df.to_dict(orient='records')
        columns = list(df.columns)
        # Return the full filename so the frontend can build the correct URL
        return jsonify({'dataset': filename, 'columns': columns, 'rows': records})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def product_page():
    return render_template('product-page/product.html')


@app.route('/<employee_dataset>/Raw', methods=['GET', 'POST'])
def show_raw(employee_dataset):
    raw_dataset_name = employee_dataset + '.csv'
    raw_dataset_directory = os.path.join(os.path.dirname(__file__), 'test_csv', raw_dataset_name)
    raw_dataset = pd.read_csv(raw_dataset_directory)
    return raw_dataset.sample(7).to_html()


@app.route('/<path:employee_dataset>/Dashboard', methods=['GET'])
def show_dashboard(employee_dataset):
    if 'application/json' in request.headers.get('Accept', ''):
        dataset_name = employee_dataset[:-4] if employee_dataset.endswith('.csv') else employee_dataset
        try:
            df = rpb.sample_predict(dataset_name)
            records = df.to_dict(orient='records')
            columns = list(df.columns)
            return jsonify({'dataset': employee_dataset, 'columns': columns, 'rows': records})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('product-page/product.html')

@app.route('/<path:employee_dataset>/global_insights', methods=['GET'])
def get_global_insights(employee_dataset):
    import pickle
    dataset_name = employee_dataset[:-4] if employee_dataset.endswith('.csv') else employee_dataset
    try:
        df_clean = rpb.clean_dataset(dataset_name)

        model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
        model = pickle.load(open(model_path, 'rb'))

        feature_cols = [c for c in df_clean.columns if c not in ['EmployeeID', 'Attrition']]
        X_test = df_clean[feature_cols].copy()

        group_explanations = fib.explainn(
            X_test=X_test,
            model=model,
            class_names=['Stay', 'Attrite'],
            group_col='JobRole'
        )

        jobrole_map = {0: 'Other', 1: 'Nurse', 2: 'Therapist', 3: 'Administrative', 4: 'Admin'}
        serializable = {jobrole_map.get(k, str(k)): v for k, v in group_explanations.items()}
        return jsonify({'explanations': serializable})
    except Exception as e:
        return jsonify({'error': str(e)}), 500  


@app.route('/<path:employee_dataset>/view_dataset', methods=['GET'])
def show_view_dataset(employee_dataset):
    try:
        df = pd.read_json(os.path.join(os.path.dirname(__file__), 'jsons', 'combined_df_table.json'))
        records = df.to_dict(orient='records')
        columns = list(df.columns)
        return jsonify({'dataset': employee_dataset, 'columns': columns, 'rows': records})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/<path:employee_dataset>/select_employee_id=<search_id>', methods=['GET'])
def show_search_employee_id(employee_dataset, search_id):
    # Browser refresh → redirect to Dashboard so JS restores cleanly
    if 'application/json' not in request.headers.get('Accept', ''):
        return render_template('product-page/product.html')
    dataset_name = employee_dataset[:-4] if employee_dataset.endswith('.csv') else employee_dataset
    result = rpb.search_employee_id(dataset_name, search_id)
    if result is None:
        return jsonify({'error': f'Employee ID {search_id} not found.'}), 404
    record = result['Employee Details'].to_dict()
    return jsonify({'dataset': employee_dataset, 'employee_id': search_id, 'record': record})
# --------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)