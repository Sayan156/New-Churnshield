"""
Prediction service for churn prediction.
Handles feature engineering, prediction, and risk assessment.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .feature_engineering import engineer_features, ALL_INPUT_FEATURES, DROP_COLUMNS
from .model_loader import get_model_loader

logger = logging.getLogger(__name__)

# Default threshold from training
DEFAULT_THRESHOLD = 0.50


def get_risk_label(probability: float) -> str:
    """Convert probability to human-readable risk label."""
    if probability >= 0.75:
        return "Very High Risk"
    elif probability >= 0.50:
        return "High Risk"
    elif probability >= 0.25:
        return "Medium Risk"
    else:
        return "Low Risk"


def prepare_input_data(input_dict: Dict) -> pd.DataFrame:
    """
    Convert input dictionary to DataFrame with proper column ordering.

    Args:
        input_dict: Dictionary with feature names and values

    Returns:
        DataFrame with features in correct order
    """
    # Create DataFrame and ensure correct column order
    X_input = pd.DataFrame([input_dict])

    # Ensure all expected columns exist and are in correct order
    for col in ALL_INPUT_FEATURES:
        if col not in X_input.columns:
            logger.warning(f"Missing column: {col}")

    # Reindex to match training order
    X_input = X_input.reindex(columns=ALL_INPUT_FEATURES)

    return X_input


def predict_churn(
    input_data: Dict,
    model_key: str = 'stacking_lr_meta',
    include_shap: bool = False
) -> Dict:
    """
    Make a churn prediction for a single customer.

    Args:
        input_data: Dictionary with customer features
        model_key: Which model to use for prediction
        include_shap: Whether to compute SHAP values (slower)

    Returns:
        Dictionary with prediction results
    """
    loader = get_model_loader()
    model = loader.load_model(model_key)

    if model is None:
        return {
            'success': False,
            'error': f'Model {model_key} not found',
        }

    try:
        # Prepare input
        X_input = prepare_input_data(input_data)

        # Make prediction
        proba = model.predict_proba(X_input)[0]
        probability = float(proba[1])  # P(churn)
        prediction = int(probability >= DEFAULT_THRESHOLD)

        # Build response
        result = {
            'success': True,
            'model_key': model_key,
            'model_name': loader.MODEL_NAMES.get(model_key, model_key),
            'prediction': prediction,
            'prediction_label': 'Churned' if prediction == 1 else 'Retained',
            'probability': probability,
            'probability_pct': round(probability * 100, 1),
            'risk_level': get_risk_label(probability),
            'threshold': DEFAULT_THRESHOLD,
        }

        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            'success': False,
            'error': str(e),
        }


def predict_churn_multi(
    input_data: Dict,
    model_keys: Optional[List[str]] = None
) -> Dict:
    """
    Get predictions from multiple models for comparison.

    Args:
        input_data: Customer features
        model_keys: List of model keys to use (default: all available)

    Returns:
        Dictionary with predictions from all models
    """
    loader = get_model_loader()

    if model_keys is None:
        model_keys = list(loader.MODEL_NAMES.keys())

    predictions = {}
    for key in model_keys:
        result = predict_churn(input_data, model_key=key)
        if result['success']:
            predictions[key] = {
                'model_name': result['model_name'],
                'probability': result['probability'],
                'prediction': result['prediction'],
                'prediction_label': result['prediction_label'],
                'risk_level': result['risk_level'],
            }

    # Calculate consensus
    if predictions:
        avg_prob = np.mean([p['probability'] for p in predictions.values()])
        churn_votes = sum(1 for p in predictions.values() if p['prediction'] == 1)
        total_models = len(predictions)

        return {
            'success': True,
            'predictions': predictions,
            'consensus': {
                'average_probability': float(avg_prob),
                'average_probability_pct': round(avg_prob * 100, 1),
                'consensus_label': 'Churned' if avg_prob >= DEFAULT_THRESHOLD else 'Retained',
                'churn_votes': churn_votes,
                'total_models': total_models,
                'agreement_pct': round(max(churn_votes, total_models - churn_votes) / total_models * 100, 1),
            }
        }

    return {'success': False, 'error': 'No models available'}


def predict_batch(
    data: List[Dict],
    model_key: str = 'stacking_lr_meta'
) -> List[Dict]:
    """
    Make predictions for multiple customers.

    Args:
        data: List of customer feature dictionaries
        model_key: Model to use

    Returns:
        List of prediction results
    """
    results = []
    for input_data in data:
        result = predict_churn(input_data, model_key=model_key)
        results.append(result)
    return results
