"""
SHAP explainability service.
Computes SHAP values for model interpretability.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import shap
from django.conf import settings

from .feature_engineering import ALL_INPUT_FEATURES
from .model_loader import get_model_loader
from .predictor import prepare_input_data, DEFAULT_THRESHOLD

logger = logging.getLogger(__name__)
SHAP_L1_REG = f"num_features({len(ALL_INPUT_FEATURES)})"

def load_reference_data() -> pd.DataFrame:
    """Load the reference dataset used for SHAP background/global analysis."""
    data_path = Path(settings.DATA_DIR) / 'BankChurners.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"Background data not found: {data_path}")

    df = pd.read_csv(data_path)
    missing_columns = [col for col in ALL_INPUT_FEATURES if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Background data is missing required columns: {', '.join(missing_columns)}"
        )

    return df.loc[:, ALL_INPUT_FEATURES].copy()


def get_background_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Load real background data for SHAP explainer.

    SHAP must see valid categorical values because the model pipeline contains
    one-hot encoding. Synthetic Gaussian data breaks the encoder contract.
    """
    reference_data = load_reference_data()
    background = reference_data.sample(
        n=min(n_samples, len(reference_data)),
        random_state=42,
    )
    logger.info("Loaded %s background rows for SHAP", len(background))
    return background.reset_index(drop=True)


class SHAPExplainer:
    """
    Manages SHAP explanation generation.
    Uses KernelExplainer for model-agnostic explanations.
    """

    def __init__(self, model, background_data: Optional[pd.DataFrame] = None):
        self.model = model
        self.background_data = background_data
        self._explainer = None

    def _predict_wrapper(self, X):
        """Wrapper for model prediction that handles DataFrame conversion."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=ALL_INPUT_FEATURES)
        else:
            X = X.reindex(columns=ALL_INPUT_FEATURES)

        return self.model.predict_proba(X)[:, 1]

    def get_explainer(self):
        """Get or create the SHAP explainer."""
        if self._explainer is None:
            if self.background_data is None:
                self.background_data = get_background_data(n_samples=100)

            self._explainer = shap.KernelExplainer(
                self._predict_wrapper,
                self.background_data,
                link='logit'
            )

        return self._explainer

    def explain_individual(
        self,
        input_data: Dict,
        n_samples: int = 100
    ) -> Dict:
        """
        Compute SHAP values for a single prediction.

        Args:
            input_data: Customer features
            n_samples: Number of samples for KernelExplainer

        Returns:
            Dictionary with SHAP values and visualization data
        """
        try:
            # Prepare input
            X_input = prepare_input_data(input_data)

            # Get prediction
            proba = self.model.predict_proba(X_input)[0, 1]

            # Compute SHAP
            explainer = self.get_explainer()
            shap_values = explainer.shap_values(
                X_input,
                nsamples=n_samples,
                l1_reg=SHAP_L1_REG,
            )

            shap_values = normalize_shap_values(shap_values)

            # Create feature importance ranking
            feature_names = ALL_INPUT_FEATURES
            abs_shap = np.abs(shap_values.flatten())
            importance_ranking = sorted(
                zip(feature_names, abs_shap, shap_values.flatten()),
                key=lambda x: abs(x[2]),
                reverse=True
            )

            return {
                'success': True,
                'probability': float(proba),
                'prediction': int(proba >= DEFAULT_THRESHOLD),
                'base_value': float(explainer.expected_value),
                'shap_values': shap_values.flatten().tolist(),
                'feature_names': feature_names,
                'importance_ranking': [
                    {'feature': name, 'abs_shap': float(abs_val), 'shap_value': float(val)}
                    for name, abs_val, val in importance_ranking
                ],
                'top_features': [
                    {'feature': name, 'shap_value': float(val)}
                    for name, _, val in importance_ranking[:10]
                ],
            }

        except Exception as e:
            logger.error(f"SHAP explanation error: {e}")
            return {
                'success': False,
                'error': str(e),
            }

    def explain_global(
        self,
        data: pd.DataFrame,
        n_samples: int = 100,
        shap_nsamples: int = 100,
    ) -> Dict:
        """
        Compute global SHAP values for a dataset sample.

        Args:
            data: DataFrame with customer features
            n_samples: Number of samples to explain

        Returns:
            Dictionary with global SHAP analysis
        """
        try:
            # Sample data
            X_sample = data.sample(n=min(n_samples, len(data)), random_state=42)

            # Compute SHAP values
            explainer = self.get_explainer()
            shap_values = explainer.shap_values(
                X_sample,
                nsamples=shap_nsamples,
                l1_reg=SHAP_L1_REG,
            )

            shap_values = normalize_shap_values(shap_values)

            # Calculate mean absolute SHAP values (global importance)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            feature_importance = sorted(
                zip(ALL_INPUT_FEATURES, mean_abs_shap),
                key=lambda x: x[1],
                reverse=True
            )

            return {
                'success': True,
                'n_samples': len(X_sample),
                'base_value': float(explainer.expected_value),
                'shap_values_shape': list(shap_values.shape),
                'feature_importance': [
                    {'feature': name, 'importance': float(importance)}
                    for name, importance in feature_importance
                ],
                'shap_values': shap_values.tolist(),
                'feature_names': ALL_INPUT_FEATURES,
            }

        except Exception as e:
            logger.error(f"Global SHAP error: {e}")
            return {
                'success': False,
                'error': str(e),
            }


def normalize_shap_values(shap_values) -> np.ndarray:
    """
    Normalize SHAP outputs across library/model variants to a 2D array of
    shape (n_rows, n_features) for the positive churn class.
    """
    if isinstance(shap_values, list):
        if not shap_values:
            raise ValueError("SHAP returned an empty list of values")
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 1:
        return shap_values.reshape(1, -1)

    if shap_values.ndim == 2:
        return shap_values

    if shap_values.ndim == 3:
        if shap_values.shape[1] == len(ALL_INPUT_FEATURES):
            return shap_values[:, :, -1]
        if shap_values.shape[2] == len(ALL_INPUT_FEATURES):
            return shap_values[:, -1, :]

    raise ValueError(f"Unexpected SHAP output shape: {shap_values.shape}")


def compute_shap_individual(
    input_data: Dict,
    model_key: str = 'stacking_lr_meta',
    n_samples: int = 100,
) -> Dict:
    """
    Compute SHAP values for a single prediction.
    Convenience function that handles model loading.
    """
    loader = get_model_loader()
    model = loader.load_model(model_key)

    if model is None:
        return {'success': False, 'error': f'Model {model_key} not found'}

    explainer = SHAPExplainer(model)
    return explainer.explain_individual(input_data, n_samples=n_samples)


def compute_shap_global(
    data: pd.DataFrame,
    model_key: str = 'stacking_lr_meta',
    n_samples: int = 100,
    shap_nsamples: int = 100,
) -> Dict:
    """
    Compute global SHAP values for a dataset.
    """
    loader = get_model_loader()
    model = loader.load_model(model_key)

    if model is None:
        return {'success': False, 'error': f'Model {model_key} not found'}

    explainer = SHAPExplainer(model)
    return explainer.explain_global(
        data,
        n_samples=n_samples,
        shap_nsamples=shap_nsamples,
    )
