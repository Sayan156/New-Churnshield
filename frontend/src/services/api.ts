/**
 * API client for ChurnShield backend
 */
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout for SHAP computations
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available (future)
    // const token = localStorage.getItem('token');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export default api;

// API endpoints
export const endpoints = {
  // Health
  health: '/health/',

  // Predictions
  predict: '/predict/',
  predictCompare: '/predict/compare/',
  predictBatch: '/predict/batch/',

  // Models
  models: '/models/',

  // SHAP
  shapIndividual: '/shap/individual/',
  shapGlobal: '/shap/global/',

  // Dashboard
  dashboardStats: '/dashboard/stats/',
};

// API functions
export const apiService = {
  // Health check
  healthCheck: () => api.get(endpoints.health),

  // Single prediction
  predict: (data: PredictionInput) => api.post(endpoints.predict, data),

  // Multi-model comparison
  predictCompare: (data: PredictionInput, modelKeys?: string[]) =>
    api.post(endpoints.predictCompare, { ...data, model_keys: modelKeys }),

  // Batch prediction
  predictBatch: (customers: PredictionInput[], modelKey?: string) =>
    api.post(endpoints.predictBatch, { customers, model_key: modelKey }),

  // Get available models
  getModels: () => api.get(endpoints.models),

  // SHAP individual
  getShapIndividual: (data: PredictionInput, modelKey?: string) =>
    api.post(endpoints.shapIndividual, { ...data, model_key: modelKey }),

  // SHAP global
  getShapGlobal: (data: any[], nSamples?: number, modelKey?: string) =>
    api.post(endpoints.shapGlobal, { data, n_samples: nSamples, model_key: modelKey }),

  // Dashboard stats
  getDashboardStats: () => api.get(endpoints.dashboardStats),
};

// Types
export interface PredictionInput {
  Customer_Age: number;
  Gender: 'M' | 'F';
  Education_Level: string;
  Marital_Status: string;
  Income_Category: string;
  Card_Category: string;
  Months_on_book: number;
  Total_Relationship_Count: number;
  Months_Inactive_12_mon: number;
  Contacts_Count_12_mon: number;
  Total_Revolving_Bal: number;
  Total_Amt_Chng_Q4_Q1: number;
  Total_Trans_Amt: number;
  Total_Trans_Ct: number;
  Total_Ct_Chng_Q4_Q1: number;
  Avg_Utilization_Ratio: number;
}

export interface PredictionResponse {
  success: boolean;
  model_key: string;
  model_name: string;
  prediction: number;
  prediction_label: string;
  probability: number;
  probability_pct: number;
  risk_level: string;
  threshold: number;
  error?: string;
}

export interface MultiModelResponse {
  success: boolean;
  predictions: Record<string, {
    model_name: string;
    probability: number;
    prediction: number;
    prediction_label: string;
    risk_level: string;
  }>;
  consensus: {
    average_probability: number;
    average_probability_pct: number;
    consensus_label: string;
    churn_votes: number;
    total_models: number;
    agreement_pct: number;
  };
}

export interface SHAPResponse {
  success: boolean;
  probability: number;
  prediction: number;
  base_value: number;
  shap_values: number[];
  feature_names: string[];
  importance_ranking: Array<{
    feature: string;
    abs_shap: number;
    shap_value: number;
  }>;
  top_features: Array<{
    feature: string;
    shap_value: number;
  }>;
}

export interface SHAPGlobalResponse {
  success: boolean;
  n_samples: number;
  base_value: number;
  shap_values_shape: number[];
  shap_values: number[][];
  feature_names: string[];
  feature_importance: Array<{
    feature: string;
    importance: number;
  }>;
  error?: string;
}

export interface DashboardStats {
  success: boolean;
  metrics: {
    model_accuracy: number;
    model_recall: number;
    model_precision: number;
    model_f1: number;
    model_roc_auc: number;
    threshold: number;
  };
  model_comparison: Array<{
    model: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number;
  }>;
  primary_model: {
    key: string;
    name: string;
  };
}

export interface BatchPredictionResponse {
  success: boolean;
  count: number;
  predictions: PredictionResponse[];
}
