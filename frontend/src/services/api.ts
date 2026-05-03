import axios from 'axios'

const API_BASE_URL = '/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

export interface PredictionRequest {
  lat: number
  lon: number
  country: string
  family: string
  body_length_mm: number
  color: string
  head_feature: string
  species?: string
  actual_species?: string
}

export interface ModelPrediction {
  model: string
  prediction: string
  confidence: number
  actual_species?: string
  correct?: boolean
  top_3_predictions: Array<{
    species: string
    probability: number
  }>
}

export interface PredictionResponse {
  success: boolean
  actual_species?: string
  model_predictions?: ModelPrediction[]
  error?: string
}

export interface ValidationSample {
  id: number
  species: string
  features: PredictionRequest
}

export interface ValidationSamplesResponse {
  success: boolean
  page: number
  page_size: number
  total: number
  species_options: string[]
  samples: ValidationSample[]
  error?: string
}

export interface ModelMetric {
  accuracy: number
  balanced_accuracy?: number
  precision: number
  recall: number
  f1_score: number
  macro_f1?: number
  top_3_accuracy?: number
}

export interface ModelInfo {
  best_model: string
  best_score: number
  metric_used: string
  all_results: Record<string, ModelMetric>
  dataset?: {
    class_count?: number
    validation_samples?: number
  }
}

export const predictStonefly = async (data: PredictionRequest): Promise<PredictionResponse> => {
  const response = await apiClient.post('/predict', data)
  return response.data
}

export const getValidationSamples = async (params: {
  page: number
  page_size: number
  keyword?: string
  species?: string
}): Promise<ValidationSamplesResponse> => {
  const response = await apiClient.get('/validation-samples', { params })
  return response.data
}

export const getFeatures = async () => {
  const response = await apiClient.get('/features')
  return response.data
}

export const getModelInfo = async (): Promise<{ success: boolean; data?: ModelInfo; error?: string }> => {
  const response = await apiClient.get('/model-info')
  return response.data
}

export const getStats = async () => {
  const response = await apiClient.get('/stats')
  return response.data
}
