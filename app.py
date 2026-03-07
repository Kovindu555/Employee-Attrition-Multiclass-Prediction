import pandas as pd
from flask import Flask, render_template, Response
import pickle
# +++++++++++++++++++++++++++ Library Corner +++++++++++++++++++++++++++

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/<employee_dataset>', methods = ['GET', 'POST'])
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

def risk_pred(p):
    if p < 0.25:
        return 'low risk'
    elif p < 0.50:
        return 'medium risk'
    elif p < 0.75:
        return 'high risk'
    else:
        return 'critical risk'

def results(y_test_proba, y_pred_binary, df):
    result_table = pd.DataFrame({
        'PersonID': df['EmployeeID'],
        'Attrition_Probability': [f"{p:.2%}" for p in y_test_proba],
        'Predicted_Risk': [risk_pred(p) for p in y_test_proba],
        'Binary_Prediction': y_pred_binary
    })
    return result_table

@app.route('/predict/<employee_dataset>', methods=['GET' ,'POST'])
def predict(employee_dataset):

    dataset = clean_dataset(employee_dataset)

    model = pickle.load(open('models/model.pkl','rb'))

    df = dataset.drop(columns=['EmployeeID']).copy()

    y_test_proba = model.predict_proba(df)[:, 1]
    y_pred_binary = (y_test_proba >= 0.5).astype(int)

    table = results(y_test_proba, y_pred_binary, dataset)

    return Response(table.to_html(), mimetype='text/html')





if __name__ == '__main__':
    app.run(debug=True)