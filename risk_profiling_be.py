import pandas as pd
import pickle
# ------------------------------ Library Corner -----------------------------

# ------------------------------ Secondary Tools ----------------------------
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
        'EmployeeID': df['EmployeeID'],
        'Age': raw_df['Age'],
        'JobRole': raw_df['JobRole'],
        'Department': raw_df['Department'],
        'Attrition_Probability': [f"{p:.2%}" for p in y_test_proba],
        'Predicted_Risk': [risk_pred(p) for p in y_test_proba],
        'Attrition_Binary': y_pred_binary
    })

    # result_table.to_json('jsons/results_table.json', orient='records', indent=4)
    return result_table

def clean_dataset(employee_dataset):

    dataset_name = employee_dataset + '.csv'
    dataset_directory = 'test_csv/' + dataset_name
    df = pd.read_csv(dataset_directory)

    # -- DroppingUnwantedColumns --
    dropping_columns = ['Over18', 'EmployeeCount', 'StandardHours']
    df.drop(dropping_columns, axis=1, inplace=True)

    # -- Encoding Department and Gender --
    df['Department_Cardiology'] = (df['Department'] == 'Cardiology') * 1
    df['Department_Maternity']  = (df['Department'] == 'Maternity') * 1
    df['Department_Neurology'] = (df['Department'] == 'Neurology') * 1

    df['Gender_Female'] = (df['Gender'] == 'Female') * 1
    df['Gender_Male'] = (df['Gender'] == 'Male') * 1

    df.drop(columns=['Gender', 'Department'], inplace=True)

    # -- Encoding BusinessTravel and MaritalStatus --

    df['BusinessTravel'] = df['BusinessTravel'].map({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
    df['MaritalStatus'] = df['MaritalStatus'].map({'Single': 0, 'Married': 1, 'Divorced': 2})

    # -- Mapping --
    df['OverTime'] = (df['OverTime'].str.lower()).map({'no': 0, 'yes': 1})
    df['EducationField'] = df['EducationField'].map({'Other': 0, 'Life Sciences': 1, 'Medical': 2, 'Marketing': 3, 'Technical Degree': 4, 'Human Resources': 5})
    df['is_other_JobRole'] = (df['JobRole'] == 'Other').astype(int)
    df['JobRole'] = df['JobRole'].map({'Other': 0, 'Nurse': 1, 'Therapist': 2, 'Administrative': 3, 'Admin': 4})

    return df
# -----------------------------------------------------------------------------


# ------------------------------- Primary Tools -------------------------------
def sample_predict(employee_dataset):

    combined_dataset(employee_dataset)
    combined_table = pd.read_json('jsons/combined_df_table.json')

    columns = [0, 1, 14, 4, 34, 35]
    dashboard_pred = combined_table.iloc[:, columns].copy()
    return dashboard_pred

def combined_dataset(employee_dataset):

    raw_dataset_name = employee_dataset + '.csv'
    raw_dataset_directory = 'test_csv/' + raw_dataset_name
    raw_dataset = pd.read_csv(raw_dataset_directory)

    dataset = clean_dataset(employee_dataset)

    model = pickle.load(open('models/model.pkl', 'rb'))

    df = dataset.drop(columns=['EmployeeID']).copy()

    y_test_proba = model.predict_proba(df)[:, 1]
    y_pred_binary = (y_test_proba >= 0.5).astype(int)

    results_table = results(y_test_proba, y_pred_binary, dataset, raw_dataset)
    cols = results_table.iloc[:, [4, 5]]
    combined_df = pd.concat([raw_dataset, cols], axis=1)
    combined_df.to_json('jsons/combined_df_table.json', orient='records', indent=4)

    return combined_df


def search_employee_id(employee_dataset, search_id):

    combined_dataset(employee_dataset)
    employee_data = pd.read_json('jsons/combined_df_table.json')

    if employee_data.index.name != 'EmployeeID':
        employee_data = employee_data.set_index('EmployeeID')

    try:
        if not isinstance(search_id, int):
            search_id = int(search_id)

        employee_info = employee_data.loc[[search_id]]
        table = employee_info.T
        table.columns = ['Employee Details']

        return table

    except (KeyError, ValueError):
        return None
# --------------------------------------------------------------------------------