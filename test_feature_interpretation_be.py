import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
import pickle

import feature_interpretation_be as fi


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "EmployeeID": range(1, n + 1),
        "Attrition": np.random.randint(0, 2, n),
        "JobRole": np.random.randint(0, 3, n),
        "Age": np.random.randint(25, 60, n),
        "MonthlyIncome": np.random.randint(3000, 15000, n),
        "JobSatisfaction": np.random.randint(1, 5, n),
        "WorkLifeBalance": np.random.randint(1, 5, n),
        "YearsAtCompany": np.random.randint(0, 20, n),
        "OverTime": np.random.randint(0, 2, n),
        "DistanceFromHome": np.random.randint(1, 30, n),
    })


@pytest.fixture
def mock_model(sample_df):
    feature_cols = [c for c in sample_df.columns if c not in ["EmployeeID", "Attrition"]]
    n_features = len(feature_cols)
    model = MagicMock()
    model.predict_proba.side_effect = lambda X: np.column_stack(
        [np.full(len(X), 0.4), np.full(len(X), 0.6)]
    )
    return model


class TestGetJobRoleShapValues:
    def test_returns_dict(self, sample_df, mock_model):
        with patch("shap.TreeExplainer") as mock_explainer_cls:
            mock_explainer = MagicMock()
            feature_cols = [c for c in sample_df.columns if c not in ["EmployeeID", "Attrition"]]
            n = len(sample_df)
            mock_explainer.shap_values.return_value = [
                np.zeros((n, len(feature_cols))),
                np.random.randn(n, len(feature_cols)),
            ]
            mock_explainer_cls.return_value = mock_explainer

            result = fi.get_jobrole_shap_values(sample_df, model=mock_model)

        assert isinstance(result, dict)

    def test_keys_match_job_roles(self, sample_df, mock_model):
        with patch("shap.TreeExplainer") as mock_explainer_cls:
            mock_explainer = MagicMock()
            feature_cols = [c for c in sample_df.columns if c not in ["EmployeeID", "Attrition"]]
            n = len(sample_df)
            mock_explainer.shap_values.return_value = [
                np.zeros((n, len(feature_cols))),
                np.random.randn(n, len(feature_cols)),
            ]
            mock_explainer_cls.return_value = mock_explainer

            result = fi.get_jobrole_shap_values(sample_df, model=mock_model)

        expected_roles = set(sample_df["JobRole"].unique())
        assert set(result.keys()) == expected_roles

    def test_values_are_lists_of_tuples(self, sample_df, mock_model):
        with patch("shap.TreeExplainer") as mock_explainer_cls:
            mock_explainer = MagicMock()
            feature_cols = [c for c in sample_df.columns if c not in ["EmployeeID", "Attrition"]]
            n = len(sample_df)
            mock_explainer.shap_values.return_value = [
                np.zeros((n, len(feature_cols))),
                np.random.randn(n, len(feature_cols)),
            ]
            mock_explainer_cls.return_value = mock_explainer

            result = fi.get_jobrole_shap_values(sample_df, model=mock_model)

        for role, features in result.items():
            assert isinstance(features, list)
            for item in features:
                assert isinstance(item, tuple)
                assert len(item) == 2
                assert isinstance(item[0], str)
                assert isinstance(item[1], float)

    def test_respects_num_features(self, sample_df, mock_model):
        with patch("shap.TreeExplainer") as mock_explainer_cls:
            mock_explainer = MagicMock()
            feature_cols = [c for c in sample_df.columns if c not in ["EmployeeID", "Attrition"]]
            n = len(sample_df)
            mock_explainer.shap_values.return_value = [
                np.zeros((n, len(feature_cols))),
                np.random.randn(n, len(feature_cols)),
            ]
            mock_explainer_cls.return_value = mock_explainer

            result = fi.get_jobrole_shap_values(sample_df, model=mock_model, num_features=3)

        for features in result.values():
            assert len(features) <= 3

    def test_jobrole_excluded_from_features_in_output(self, sample_df, mock_model):
        with patch("shap.TreeExplainer") as mock_explainer_cls:
            mock_explainer = MagicMock()
            feature_cols = [c for c in sample_df.columns if c not in ["EmployeeID", "Attrition"]]
            n = len(sample_df)
            mock_explainer.shap_values.return_value = [
                np.zeros((n, len(feature_cols))),
                np.random.randn(n, len(feature_cols)),
            ]
            mock_explainer_cls.return_value = mock_explainer

            result = fi.get_jobrole_shap_values(sample_df, model=mock_model)

        for features in result.values():
            feature_names = [f[0] for f in features]
            assert "JobRole" not in feature_names

    def test_shap_values_as_ndarray_not_list(self, sample_df, mock_model):
        with patch("shap.TreeExplainer") as mock_explainer_cls:
            mock_explainer = MagicMock()
            feature_cols = [c for c in sample_df.columns if c not in ["EmployeeID", "Attrition"]]
            n = len(sample_df)
            mock_explainer.shap_values.return_value = np.random.randn(n, len(feature_cols))
            mock_explainer_cls.return_value = mock_explainer

            result = fi.get_jobrole_shap_values(sample_df, model=mock_model)

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_fallback_to_kernel_explainer(self, sample_df, mock_model):
        with patch("shap.TreeExplainer", side_effect=Exception("no tree")):
            with patch("shap.maskers.Independent") as mock_masker_cls:
                with patch("shap.Explainer") as mock_explainer_cls:
                    feature_cols = [c for c in sample_df.columns if c not in ["EmployeeID", "Attrition"]]
                    n = len(sample_df)

                    mock_shap_obj = MagicMock()
                    mock_shap_obj.values = np.random.randn(n, len(feature_cols))
                    mock_explainer_cls.return_value = MagicMock(return_value=mock_shap_obj)

                    result = fi.get_jobrole_shap_values(sample_df, model=mock_model)

        assert isinstance(result, dict)

    def test_custom_group_col(self, sample_df, mock_model):
        df = sample_df.copy()
        df["Department"] = np.random.randint(0, 2, len(df))

        with patch("shap.TreeExplainer") as mock_explainer_cls:
            mock_explainer = MagicMock()
            feature_cols = [c for c in df.columns if c not in ["EmployeeID", "Attrition"]]
            n = len(df)
            mock_explainer.shap_values.return_value = [
                np.zeros((n, len(feature_cols))),
                np.random.randn(n, len(feature_cols)),
            ]
            mock_explainer_cls.return_value = mock_explainer

            result = fi.get_jobrole_shap_values(df, model=mock_model, group_col="Department")

        assert set(result.keys()) == set(df["Department"].unique())


class TestGetEmployeeLimeValues:
    def test_returns_list_of_tuples(self, sample_df, mock_model):
        result = fi.get_employee_lime_values(1, sample_df, model=mock_model, num_samples=50)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_result_sorted_by_abs_weight_descending(self, sample_df, mock_model):
        result = fi.get_employee_lime_values(1, sample_df, model=mock_model, num_samples=50)
        weights = [abs(w) for _, w in result]
        assert weights == sorted(weights, reverse=True)

    def test_respects_num_features(self, sample_df, mock_model):
        result = fi.get_employee_lime_values(1, sample_df, model=mock_model, num_samples=50, num_features=3)
        assert len(result) <= 3

    def test_invalid_employee_id_raises(self, sample_df, mock_model):
        with pytest.raises(ValueError, match="9999"):
            fi.get_employee_lime_values(9999, sample_df, model=mock_model, num_samples=50)

    def test_does_not_mutate_dataset(self, sample_df, mock_model):
        original = sample_df.copy()
        fi.get_employee_lime_values(1, sample_df, model=mock_model, num_samples=50)
        pd.testing.assert_frame_equal(sample_df, original)

    def test_feature_descriptions_are_strings(self, sample_df, mock_model):
        result = fi.get_employee_lime_values(1, sample_df, model=mock_model, num_samples=50)
        for desc, _ in result:
            assert isinstance(desc, str)

    def test_weights_are_floats(self, sample_df, mock_model):
        result = fi.get_employee_lime_values(1, sample_df, model=mock_model, num_samples=50)
        for _, weight in result:
            assert isinstance(weight, float)

    def test_reproducible_with_same_random_state(self, sample_df, mock_model):
        r1 = fi.get_employee_lime_values(1, sample_df, model=mock_model, num_samples=50, random_state=0)
        r2 = fi.get_employee_lime_values(1, sample_df, model=mock_model, num_samples=50, random_state=0)
        assert r1 == r2

    def test_different_employees_can_differ(self, sample_df, mock_model):
        r1 = fi.get_employee_lime_values(1, sample_df, model=mock_model, num_samples=50, random_state=42)
        r2 = fi.get_employee_lime_values(2, sample_df, model=mock_model, num_samples=50, random_state=42)
        assert r1 != r2



# pytest test_feature_interpretation_be.py -v     -> to run
