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
}

export interface PredictionResponse {
  success: boolean
  prediction?: string
  confidence?: number
  top_3_predictions?: Array<{
    species: string
    probability: number
  }>
  error?: string
}

export const predictStonefly = async (data: PredictionRequest): Promise<PredictionResponse> => {
  const response = await apiClient.post('/predict', data)
  return response.data
}

export const getFeatures = async () => {
  const response = await apiClient.get('/features')
  return response.data
}

export const getModelInfo = async () => {
  const response = await apiClient.get('/model-info')
  return response.data
}

export const getStats = async () => {
  const response = await apiClient.get('/stats')
  return response.data
}
