# import libraries and suppress warnings
import os
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import warnings
import shap
import pickle

warnings.filterwarnings('ignore')

model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
model = pickle.load(open(model_path, 'rb'))


def get_jobrole_shap_values(dataset, model=model, group_col='JobRole', num_features=10):
    """
    Computes mean absolute SHAP values per JobRole group.

    Parameters:
        dataset     : pd.DataFrame — the cleaned dataset (must include JobRole, EmployeeID, Attrition)
        model       : trained sklearn-compatible model
        group_col   : column name to group by (default: 'JobRole')
        num_features: how many top features to return per group

    Returns:
        dict — { group_value: [(feature_name, mean_shap_value), ...] }
    """
    df = dataset.copy()
    feature_cols = [c for c in df.columns if c not in ['EmployeeID', 'Attrition']]
    X_full = df[feature_cols].values

    # Build SHAP explainer — prefer TreeExplainer, fall back to KernelExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values_all = explainer.shap_values(X_full)
        
        if isinstance(shap_values_all, list):
            shap_matrix = shap_values_all[1]   # class 1 = Attrition happens
        else:
            shap_matrix = shap_values_all
    except Exception:
        background_indices = np.random.choice(X_full.shape[0], min(30, X_full.shape[0]), replace=False)
        background_data = X_full[background_indices]

        def model_predict(X):
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[:, 1]
            return model.predict(X)

        masker = shap.maskers.Independent(background_data)
        explainer = shap.Explainer(model_predict, masker, feature_names=feature_cols)
        shap_obj = explainer(X_full)
        shap_matrix = shap_obj.values   # shape: (n_samples, n_features)

    # Attach SHAP matrix back to the dataframe so we can group it
    shap_df = pd.DataFrame(shap_matrix, columns=feature_cols, index=df.index)
    shap_df[group_col] = df[group_col].values

    group_explanations = {}
    for group_val, group_shap in shap_df.groupby(group_col):
        display_cols = [c for c in feature_cols if c != group_col]
        feature_means = group_shap[display_cols].mean()         # signed mean per feature
        top_features = (
            feature_means
            .reindex(feature_means.abs().sort_values(ascending=False).index)
            .head(num_features)
        )
        group_explanations[group_val] = [(feat, float(val)) for feat, val in top_features.items()]

    return group_explanations


def get_employee_lime_values(employee_id, dataset, model=model,
                              num_samples=200, num_features=10, random_state=42):
    """
    Computes LIME feature importances for a single employee.

    Parameters:
        employee_id : value to look up in the 'EmployeeID' column
        dataset     : pd.DataFrame — the cleaned dataset
        model       : trained sklearn-compatible model
        num_samples : LIME perturbation samples
        num_features: top features to return
        random_state: reproducibility seed

    Returns:
        list of (feature_description, weight) tuples — sorted by |weight| descending
    """
    df = dataset.copy()

    employee_row = df[df['EmployeeID'] == employee_id]
    if employee_row.empty:
        raise ValueError(f"EmployeeID {employee_id} not found in dataset")

    feature_cols = [c for c in df.columns if c not in ['EmployeeID', 'Attrition']]
    X_full = df[feature_cols].values

    explainer = LimeTabularExplainer(
        X_full,
        feature_names=feature_cols,
        class_names=['Stay', 'Attrite'],
        discretize_continuous=True,
        random_state=random_state
    )

    instance = employee_row[feature_cols].values[0]

    exp = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_samples=num_samples,
        num_features=num_features
    )

    result = exp.as_list()                              # [(feature_desc, weight), ...]
    result.sort(key=lambda x: abs(x[1]), reverse=True)
    return result