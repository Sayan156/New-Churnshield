"""
URL Configuration for the ChurnShield API.
"""
from django.urls import path
from . import views

urlpatterns = [
    # Health check
    path('health/', views.HealthCheckView.as_view(), name='health'),

    # Prediction endpoints
    path('predict/', views.PredictView.as_view(), name='predict'),
    path('predict/compare/', views.PredictMultiView.as_view(), name='predict-multi'),
    path('predict/batch/', views.BatchPredictView.as_view(), name='batch-predict'),

    # Model information
    path('models/', views.ModelsView.as_view(), name='models'),

    # SHAP explainability
    path('shap/individual/', views.SHAPIndividualView.as_view(), name='shap-individual'),
    path('shap/global/', views.SHAPGlobalView.as_view(), name='shap-global'),

    # Dashboard
    path('dashboard/stats/', views.DashboardStatsView.as_view(), name='dashboard-stats'),
]
