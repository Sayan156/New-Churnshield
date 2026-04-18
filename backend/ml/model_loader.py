"""
Model loading service with caching.
Handles loading pre-trained pickle models from the models directory.
"""
import os
import logging
from functools import lru_cache
from pathlib import Path
import cloudpickle

from django.conf import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Manages loading and caching of ML models.
    Uses LRU cache to avoid reloading models on every prediction.
    """

    # Model name mappings
    MODEL_NAMES = {
        'stacking_lr_meta': 'Stacking (LR Meta)',
        'stacking_xgb_meta': 'Stacking (XGB Meta)',
        'xgboost': 'XGBoost',
    }

    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or settings.MODEL_DIR

    def get_model_path(self, model_key: str) -> Path:
        """Get the full path to a model file."""
        return self.model_dir / f"{model_key}.pkl"

    def model_exists(self, model_key: str) -> bool:
        """Check if a model file exists."""
        return self.get_model_path(model_key).exists()

    def list_available_models(self) -> list:
        """Return list of available models with their metadata."""
        available = []
        for key, name in self.MODEL_NAMES.items():
            path = self.get_model_path(key)
            available.append({
                'key': key,
                'name': name,
                'exists': path.exists(),
                'size_mb': round(path.stat().st_size / 1e6, 2) if path.exists() else 0,
            })
        return available

    @lru_cache(maxsize=3)
    def load_model(self, model_key: str):
        """
        Load a model from disk with caching.

        Args:
            model_key: The model identifier (e.g., 'stacking_lr_meta')

        Returns:
            Loaded sklearn pipeline or None if not found
        """
        path = self.get_model_path(model_key)

        if not path.exists():
            logger.warning(f"Model not found: {path}")
            return None

        try:
            with open(path, 'rb') as f:
                model = cloudpickle.load(f)
            logger.info(f"Loaded model: {model_key} ({path.stat().st_size / 1e6:.1f} MB)")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            return None

    def get_primary_model(self):
        """Get the primary (best) model for predictions."""
        return self.load_model('stacking_lr_meta')

    def get_all_loaded_models(self) -> dict:
        """Get all available and loaded models."""
        models = {}
        for key in self.MODEL_NAMES.keys():
            model = self.load_model(key)
            if model is not None:
                models[key] = {
                    'model': model,
                    'name': self.MODEL_NAMES[key],
                }
        return models


# Global loader instance
_loader = None


def get_model_loader() -> ModelLoader:
    """Get or create the global model loader instance."""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader


def load_primary_model():
    """Load the primary model for predictions."""
    return get_model_loader().get_primary_model()


def load_model(model_key: str):
    """Load a specific model by key."""
    return get_model_loader().load_model(model_key)
