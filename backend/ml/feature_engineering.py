"""
Feature engineering module - extracted from the training notebook.
These functions must match exactly what was used during model training.
"""
import pandas as pd


def engineer_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering transformations to input data.
    Must match the exact logic from churn_shield_model_final.ipynb

    Args:
        X_df: Input DataFrame with raw features

    Returns:
        DataFrame with engineered features added
    """
    df_eng = X_df.copy()

    # Derived features from notebook
    df_eng['avg_amt_per_txn'] = df_eng['Total_Trans_Amt'] / (df_eng['Total_Trans_Ct'] + 1e-9)
    df_eng['engagement_score'] = df_eng['Total_Trans_Ct'] * df_eng['Avg_Utilization_Ratio']

    return df_eng


# Numeric features that go through StandardScaler
NUMERIC_FEATURES = [
    'Customer_Age',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Total_Revolving_Bal',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
]

# Categorical features that get OneHotEncoded
CATEGORICAL_FEATURES = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category',
]

# All input features expected by the model
ALL_INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns to drop (from training notebook)
DROP_COLUMNS = [
    'CLIENTNUM',
    'Credit_Limit',
    'Dependent_count',
    'Avg_Open_To_Buy',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
]
