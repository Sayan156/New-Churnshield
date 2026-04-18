"""
DRF Views for the ChurnShield API.
"""
import logging
import pandas as pd
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import JSONParser

from ml.predictor import (
    predict_churn,
    predict_churn_multi,
    predict_batch,
    DEFAULT_THRESHOLD,
)
from ml.shap_service import compute_shap_individual, compute_shap_global, load_reference_data
from ml.model_loader import get_model_loader
from ml.feature_engineering import engineer_features, ALL_INPUT_FEATURES

from .serializers import (
    PredictionInputSerializer,
    PredictionResultSerializer,
    MultiModelPredictionSerializer,
    SHAPIndividualSerializer,
    SHAPGlobalSerializer,
    ModelInfoSerializer,
    BatchPredictionInputSerializer,
)

logger = logging.getLogger(__name__)


class HealthCheckView(APIView):
    """Simple health check endpoint."""

    def get(self, request):
        return Response({
            'status': 'healthy',
            'service': 'ChurnShield API',
        })


class PredictView(APIView):
    """
    Single customer churn prediction endpoint.

    POST /api/predict/
    {
        "Customer_Age": 45,
        "Gender": "M",
        ...
    }
    """

    parser_classes = [JSONParser]

    def post(self, request):
        # Validate input
        serializer = PredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                'success': False,
                'error': 'Validation error',
                'details': serializer.errors,
            }, status=status.HTTP_400_BAD_REQUEST)

        # Get model from request or use default
        model_key = request.data.get('model_key', 'stacking_lr_meta')

        # Make prediction
        result = predict_churn(
            input_data=serializer.validated_data,
            model_key=model_key,
        )

        if result['success']:
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PredictMultiView(APIView):
    """
    Multi-model prediction comparison endpoint.

    POST /api/predict/compare/
    {
        "Customer_Age": 45,
        "Gender": "M",
        ...
    }

    Returns predictions from all available models.
    """

    parser_classes = [JSONParser]

    def post(self, request):
        # Validate input
        serializer = PredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                'success': False,
                'error': 'Validation error',
                'details': serializer.errors,
            }, status=status.HTTP_400_BAD_REQUEST)

        # Get optional model list
        model_keys = request.data.get('model_keys', None)

        # Get multi-model predictions
        result = predict_churn_multi(
            input_data=serializer.validated_data,
            model_keys=model_keys,
        )

        if result['success']:
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BatchPredictView(APIView):
    """
    Batch prediction endpoint for multiple customers.

    POST /api/predict/batch/
    {
        "customers": [
            {"Customer_Age": 45, "Gender": "M", ...},
            {"Customer_Age": 32, "Gender": "F", ...},
        ],
        "model_key": "stacking_lr_meta"
    }
    """

    parser_classes = [JSONParser]

    def post(self, request):
        # Validate batch input
        serializer = BatchPredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                'success': False,
                'error': 'Validation error',
                'details': serializer.errors,
            }, status=status.HTTP_400_BAD_REQUEST)

        customers = serializer.validated_data['customers']
        model_key = serializer.validated_data.get('model_key', 'stacking_lr_meta')

        # Make batch predictions
        results = predict_batch(customers, model_key=model_key)

        return Response({
            'success': True,
            'count': len(results),
            'predictions': results,
        }, status=status.HTTP_200_OK)


class ModelsView(APIView):
    """
    List available models endpoint.

    GET /api/models/
    """

    def get(self, request):
        loader = get_model_loader()
        models = loader.list_available_models()

        serializer = ModelInfoSerializer(models, many=True)
        return Response({
            'success': True,
            'models': serializer.data,
            'primary_model': 'stacking_lr_meta',
        })


class SHAPIndividualView(APIView):
    """
    Compute SHAP values for individual prediction.

    POST /api/shap/individual/
    {
        "Customer_Age": 45,
        "Gender": "M",
        ...
    }
    """

    parser_classes = [JSONParser]

    def post(self, request):
        # Validate input
        serializer = PredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                'success': False,
                'error': 'Validation error',
                'details': serializer.errors,
            }, status=status.HTTP_400_BAD_REQUEST)

        model_key = request.data.get('model_key', 'stacking_lr_meta')
        n_samples = int(request.data.get('n_samples', 100))

        # Compute SHAP
        result = compute_shap_individual(
            input_data=serializer.validated_data,
            model_key=model_key,
            n_samples=n_samples,
        )

        if result['success']:
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SHAPGlobalView(APIView):
    """
    Compute global SHAP values for dataset sample.

    POST /api/shap/global/
    {
        "data": [...],  // Array of customer records
        "n_samples": 100
    }

    Note: This endpoint requires data to be sent.
    For production, you might want to load from database instead.
    """

    parser_classes = [JSONParser]

    def post(self, request):
        data = request.data.get('data', [])
        n_samples = request.data.get('n_samples', 100)
        model_key = request.data.get('model_key', 'stacking_lr_meta')

        try:
            n_samples = max(1, int(n_samples))

            if data:
                df = pd.DataFrame(data)

                # Ensure all required columns exist
                for col in ALL_INPUT_FEATURES:
                    if col not in df.columns:
                        return Response({
                            'success': False,
                            'error': f'Missing required column: {col}',
                        }, status=status.HTTP_400_BAD_REQUEST)
            else:
                df = load_reference_data()

            result = compute_shap_global(
                data=df,
                model_key=model_key,
                n_samples=n_samples,
                shap_nsamples=n_samples,
            )

            if result['success']:
                return Response(result, status=status.HTTP_200_OK)
            else:
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            logger.error(f"Global SHAP error: {e}")
            return Response({
                'success': False,
                'error': str(e),
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DashboardStatsView(APIView):
    """
    Get dashboard statistics.

    GET /api/dashboard/stats/

    Returns aggregated metrics for the dashboard overview.
    """

    def get(self, request):
        # In production, this would query the database
        # For now, return static metrics from the training results

        stats = {
            'success': True,
            'metrics': {
                'model_accuracy': 0.9589,
                'model_recall': 0.9447,
                'model_precision': 0.8247,
                'model_f1': 0.8806,
                'model_roc_auc': 0.9930,
                'threshold': DEFAULT_THRESHOLD,
            },
            'model_comparison': [
                {'model': 'Random Forest', 'accuracy': 0.9457, 'precision': 0.7942, 'recall': 0.8934, 'f1': 0.8409, 'roc_auc': 0.9817},
                {'model': 'Gradient Boosting', 'accuracy': 0.9500, 'precision': 0.7887, 'recall': 0.9406, 'f1': 0.8579, 'roc_auc': 0.9910},
                {'model': 'XGBoost', 'accuracy': 0.9645, 'precision': 0.8571, 'recall': 0.9344, 'f1': 0.8941, 'roc_auc': 0.9934},
                {'model': 'CatBoost', 'accuracy': 0.9658, 'precision': 0.9324, 'recall': 0.8484, 'f1': 0.8884, 'roc_auc': 0.9915},
                {'model': 'Stacking (LR Meta)', 'accuracy': 0.9589, 'precision': 0.8247, 'recall': 0.9447, 'f1': 0.8806, 'roc_auc': 0.9930},
                {'model': 'Stacking (XGB Meta)', 'accuracy': 0.9585, 'precision': 0.8255, 'recall': 0.9406, 'f1': 0.8793, 'roc_auc': 0.9926},
            ],
            'primary_model': {
                'key': 'stacking_lr_meta',
                'name': 'Stacking (LR Meta)',
            },
        }

        return Response(stats, status=status.HTTP_200_OK)
