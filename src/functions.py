# import libraries and suppress warnings
import pandas as pd
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer
import warnings
import shap

warnings.filterwarnings('ignore')

def explainn(model, X_test, feature_names=None, class_names=None, 
                                 group_col='JobRole', num_samples=1000, 
                                 num_features=10, random_state=42):
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=feature_names)
    
    feature_cols = [c for c in X_test.columns if c != group_col]
    X_features = X_test[feature_cols].values
    
    explainer = LimeTabularExplainer(
        X_features,
        feature_names=feature_cols,
        class_names=class_names,
        discretize_continuous=True,
        random_state=random_state
    )
    
    groups = X_test.groupby(group_col)
    group_explanations = {}
    
    for group_val, group_df in groups:
        group_features = group_df[feature_cols].values
        all_explanations = []
        
        for idx in range(len(group_features)):
            instance = group_features[idx]
            exp = explainer.explain_instance(
                instance, 
                model.predict_proba,
                num_samples=num_samples,
                num_features=num_features
            )
            all_explanations.append(exp.as_list())
        
        # Aggregate by feature
        feature_weights = {}
        for exp_list in all_explanations:
            for feature_desc, weight in exp_list:
                if feature_desc not in feature_weights:
                    feature_weights[feature_desc] = []
                feature_weights[feature_desc].append(weight)
        
        # Calculate mean and sort by absolute value
        aggregated = [(feat, float(np.mean(weights))) for feat, weights in feature_weights.items()]
        aggregated.sort(key=lambda x: abs(x[1]), reverse=True)
        
        group_explanations[group_val] = aggregated[:num_features]
    
    return group_explanations


def get_employee_shap_values(model, employee_id, dataset):
    # Load dataset
    df = dataset.copy()
    
    # Find employee by ID
    employee_row = df[df['EmployeeID'] == employee_id]
    
    if employee_row.empty:
        raise ValueError(f"EmployeeID {employee_id} not found in dataset")
    
    # Get feature columns (exclude EmployeeID and target Attrition)
    feature_cols = [col for col in df.columns if col not in ['EmployeeID', 'Attrition']]
    
    # Extract employee features as numpy array
    employee_features = employee_row[feature_cols].values
    
    # Get the full dataset features (excluding ID and target)
    X_full = df[feature_cols].values
    
    # Create SHAP explainer based on model type
    try:
        explainer = shap.TreeExplainer(model)
    except:
        def model_predict(X):
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[:, 1]
            else:
                return model.predict(X)
        
        # Use shap.Explainer with a masker (modern API that avoids the base_score issue)
        background_indices = np.random.choice(X_full.shape[0], min(100, X_full.shape[0]), replace=False)
        background_data = X_full[background_indices]
        
        masker = shap.maskers.Independent(background_data)
        
        explainer = shap.Explainer(model_predict, masker, feature_names=feature_cols)
        
        shap_values = explainer(employee_features)
        
        shap_values_for_employee = shap_values.values[0]
        
    result = {}
    for feature_name, shap_value in zip(feature_cols, shap_values_for_employee):
        result[feature_name] = float(shap_value)
    
    return result