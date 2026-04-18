"""
DRF Serializers for the ChurnShield API.
"""
from rest_framework import serializers

from ml.feature_engineering import ALL_INPUT_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES


class PredictionInputSerializer(serializers.Serializer):
    """
    Serializer for prediction input data.
    Validates all 16 required features.
    """

    # Numeric features
    Customer_Age = serializers.IntegerField(min_value=18, max_value=100)
    Months_on_book = serializers.IntegerField(min_value=12, max_value=60)
    Total_Relationship_Count = serializers.IntegerField(min_value=1, max_value=8)
    Months_Inactive_12_mon = serializers.IntegerField(min_value=0, max_value=6)
    Contacts_Count_12_mon = serializers.IntegerField(min_value=0, max_value=6)
    Total_Revolving_Bal = serializers.IntegerField(min_value=0, max_value=5000)
    Total_Amt_Chng_Q4_Q1 = serializers.FloatField(min_value=0.0, max_value=5.0)
    Total_Trans_Amt = serializers.IntegerField(min_value=0, max_value=50000)
    Total_Trans_Ct = serializers.IntegerField(min_value=0, max_value=150)
    Total_Ct_Chng_Q4_Q1 = serializers.FloatField(min_value=0.0, max_value=5.0)
    Avg_Utilization_Ratio = serializers.FloatField(min_value=0.0, max_value=1.0)

    # Categorical features are left open-ended so unseen values can pass through
    # to the preprocessing pipeline, which is expected to handle unknown labels.
    Gender = serializers.CharField()
    Education_Level = serializers.CharField()
    Marital_Status = serializers.CharField()
    Income_Category = serializers.CharField()
    Card_Category = serializers.CharField()

    def validate(self, data):
        """Additional cross-field validation if needed."""
        return data


class PredictionResultSerializer(serializers.Serializer):
    """
    Serializer for single prediction results.
    """
    success = serializers.BooleanField()
    model_key = serializers.CharField()
    model_name = serializers.CharField()
    prediction = serializers.IntegerField()
    prediction_label = serializers.CharField()
    probability = serializers.FloatField()
    probability_pct = serializers.FloatField()
    risk_level = serializers.CharField()
    threshold = serializers.FloatField()
    error = serializers.CharField(required=False)


class MultiModelPredictionSerializer(serializers.Serializer):
    """
    Serializer for multi-model comparison results.
    """
    success = serializers.BooleanField()
    predictions = serializers.DictField(child=serializers.DictField())
    consensus = serializers.DictField()
    error = serializers.CharField(required=False)


class SHAPIndividualSerializer(serializers.Serializer):
    """
    Serializer for individual SHAP explanation results.
    """
    success = serializers.BooleanField()
    probability = serializers.FloatField()
    prediction = serializers.IntegerField()
    base_value = serializers.FloatField()
    shap_values = serializers.ListField(child=serializers.FloatField())
    feature_names = serializers.ListField(child=serializers.CharField())
    importance_ranking = serializers.ListField(child=serializers.DictField())
    top_features = serializers.ListField(child=serializers.DictField())
    error = serializers.CharField(required=False)


class SHAPGlobalSerializer(serializers.Serializer):
    """
    Serializer for global SHAP analysis results.
    """
    success = serializers.BooleanField()
    n_samples = serializers.IntegerField()
    base_value = serializers.FloatField()
    feature_importance = serializers.ListField(child=serializers.DictField())
    shap_values_shape = serializers.ListField(child=serializers.IntegerField())
    shap_values = serializers.ListField()  # Nested list
    feature_names = serializers.ListField(child=serializers.CharField())
    error = serializers.CharField(required=False)


class ModelInfoSerializer(serializers.Serializer):
    """
    Serializer for model metadata.
    """
    key = serializers.CharField()
    name = serializers.CharField()
    exists = serializers.BooleanField()
    size_mb = serializers.FloatField()


class DashboardStatsSerializer(serializers.Serializer):
    """
    Serializer for dashboard statistics.
    """
    total_customers = serializers.IntegerField()
    churn_rate = serializers.FloatField()
    avg_churn_probability = serializers.FloatField()
    risk_distribution = serializers.DictField()
    top_risk_factors = serializers.ListField(child=serializers.DictField())


class BatchPredictionInputSerializer(serializers.Serializer):
    """
    Serializer for batch prediction input.
    """
    customers = serializers.ListField(
        child=PredictionInputSerializer()
    )
    model_key = serializers.CharField(required=False, default='stacking_lr_meta')
