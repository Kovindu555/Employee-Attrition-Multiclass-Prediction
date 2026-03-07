import pandas as pd
from flask import Flask, render_template, Response
import pickle
# ------------------------------ Library Corner -----------------------------

# ---------------------------------- Tools ----------------------------------
def risk_pred(p):
    if p < 0.25:
        return 'Low Risk'
    elif p < 0.50:
        return 'Medium Risk'
    elif p < 0.75:
        return 'High Risk'
    else:
        return 'Critical Risk'

def results(y_test_proba, y_pred_binary, df, raw_df):
    result_table = pd.DataFrame({
        'PersonID': df['EmployeeID'],
        'Age': raw_df['Age'],
        'JobRole': raw_df['JobRole'],
        'Department': raw_df['Department'],
        'Attrition_Probability': [f"{p:.2%}" for p in y_test_proba],
        'Predicted_Risk': [risk_pred(p) for p in y_test_proba],
        'Attrition_Binary': y_pred_binary
    })

    result_table.to_json('jsons/results_table.json', orient='records', indent=4)
    return result_table

def clean_dataset(employee_dataset):

    dataset_name = employee_dataset + '.csv'
    dataset_directory = 'test_csv/' + dataset_name
    df = pd.read_csv(dataset_directory)

    # ------------- DroppingUnwantedColumns -------------
    dropping_columns = ['Over18', 'EmployeeCount', 'StandardHours']
    df.drop(dropping_columns, axis=1, inplace=True)

    # ------------- Encoding Department and Gender -------------
    df['Department_Cardiology'] = (df['Department'] == 'Cardiology') * 1
    df['Department_Maternity']  = (df['Department'] == 'Maternity') * 1
    df['Department_Neurology'] = (df['Department'] == 'Neurology') * 1

    df['Gender_Female'] = (df['Gender'] == 'Female') * 1
    df['Gender_Male'] = (df['Gender'] == 'Male') * 1

    df.drop(columns=['Gender', 'Department'], inplace=True)

    # ------------- Encoding BusinessTravel and MaritalStatus -------------

    df['BusinessTravel'] = df['BusinessTravel'].map({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
    df['MaritalStatus'] = df['MaritalStatus'].map({'Single': 0, 'Married': 1, 'Divorced': 2})

    # ------------- Mapping -------------
    df['OverTime'] = (df['OverTime'].str.lower()).map({'no': 0, 'yes': 1})
    df['EducationField'] = df['EducationField'].map({'Other': 0, 'Life Sciences': 1, 'Medical': 2, 'Marketing': 3, 'Technical Degree': 4, 'Human Resources': 5})
    df['is_other_JobRole'] = (df['JobRole'] == 'Other').astype(int)
    df['JobRole'] = df['JobRole'].map({'Other': 0, 'Nurse': 1, 'Therapist': 2, 'Administrative': 3, 'Admin': 4})

    return df
# -----------------------------------------------------------------------------


app = Flask(__name__)

# ------------------------------- App Functions -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<employee_dataset>', methods=['GET' ,'POST'])
def sample_predict(employee_dataset):

    raw_dataset_name = employee_dataset + '.csv'
    raw_dataset_directory = 'test_csv/' + raw_dataset_name
    raw_dataset = pd.read_csv(raw_dataset_directory)

    dataset = clean_dataset(employee_dataset)

    model = pickle.load(open('models/model.pkl','rb'))

    df = dataset.drop(columns=['EmployeeID']).copy()

    y_test_proba = model.predict_proba(df)[:, 1]
    y_pred_binary = (y_test_proba >= 0.5).astype(int)

    table = results(y_test_proba, y_pred_binary, dataset, raw_dataset)

    return Response(table.to_html(), mimetype='text/html')

@app.route('/ViewInsights/<employee_dataset>', methods=['GET', 'POST'])
def combined_dataset(employee_dataset):
    raw_dataset_name = employee_dataset + '.csv'
    raw_dataset_directory = 'test_csv/' + raw_dataset_name
    raw_dataset = pd.read_csv(raw_dataset_directory)

    sample_predict(employee_dataset)

    results_table = pd.read_json('jsons/results_table.json')
    cols = results_table.iloc[:, [4, 5, 6]]
    combined_df = pd.concat([raw_dataset, cols], axis=1)

    return Response(combined_df.to_html(), mimetype='text/html')
# --------------------------------------------------------------------------------


# ------------------------------- Testing Purposes -------------------------------
@app.route('/show/<df>', methods=['GET', 'POST'])
def show(df):
    df = clean_dataset(df)
    return df.sample(7).to_html()


# --------------------------------------------------------------------------------


if __name__ == '__main__':
    app.run(debug=True)