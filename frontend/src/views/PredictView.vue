<template>
  <div class="predict">
    <el-row :gutter="20">
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><Edit /></el-icon>
              <span>石蝇特征输入</span>
            </div>
          </template>
          <el-form :model="formData" label-position="top" :rules="rules" ref="formRef">
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="纬度 (Latitude)" prop="lat">
                  <el-input-number v-model="formData.lat" :min="-90" :max="90" :precision="6" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="经度 (Longitude)" prop="lon">
                  <el-input-number v-model="formData.lon" :min="-180" :max="180" :precision="6" style="width: 100%" />
                </el-form-item>
              </el-col>
            </el-row>

            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="国家" prop="country">
                  <el-select v-model="formData.country" placeholder="请选择国家" style="width: 100%">
                    <el-option label="美国 (US)" value="US" />
                    <el-option label="新西兰 (NZ)" value="NZ" />
                    <el-option label="加拿大 (CA)" value="CA" />
                    <el-option label="德国 (DE)" value="DE" />
                    <el-option label="荷兰 (NL)" value="NL" />
                    <el-option label="西班牙 (ES)" value="ES" />
                    <el-option label="法国 (FR)" value="FR" />
                  </el-select>
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="科 (Family)" prop="family">
                  <el-select v-model="formData.family" placeholder="请选择科" style="width: 100%">
                    <el-option label="Perlidae" value="Perlidae" />
                    <el-option label="Austroperlidae" value="Austroperlidae" />
                    <el-option label="Pteronarcyidae" value="Pteronarcyidae" />
                    <el-option label="Capniidae" value="Capniidae" />
                    <el-option label="Leuctridae" value="Leuctridae" />
                    <el-option label="Taeniopterygidae" value="Taeniopterygidae" />
                    <el-option label="Nemouridae" value="Nemouridae" />
                    <el-option label="Eustheniidae" value="Eustheniidae" />
                    <el-option label="Gripopterygidae" value="Gripopterygidae" />
                    <el-option label="Perlodidae" value="Perlodidae" />
                  </el-select>
                </el-form-item>
              </el-col>
            </el-row>

            <el-form-item label="体长 (mm)" prop="body_length_mm">
              <el-slider v-model="formData.body_length_mm" :min="0" :max="50" :step="0.1" show-input />
            </el-form-item>

            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="颜色" prop="color">
                  <el-select v-model="formData.color" placeholder="请选择颜色" style="width: 100%">
                    <el-option label="棕色 (Brown)" value="brown" />
                    <el-option label="深褐色 (Dark)" value="dark" />
                    <el-option label="黄色 (Yellow)" value="yellow" />
                    <el-option label="黑色 (Black)" value="black" />
                  </el-select>
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="头部特征" prop="head_feature">
                  <el-select v-model="formData.head_feature" placeholder="请选择头部特征" style="width: 100%">
                    <el-option label="小触角 (Small antenna)" value="small antenna" />
                    <el-option label="圆形 (Rounded)" value="rounded" />
                    <el-option label="大眼睛 (Large eye)" value="large eye" />
                  </el-select>
                </el-form-item>
              </el-col>
            </el-row>

            <el-form-item>
              <el-button type="primary" @click="submitPrediction" :loading="loading" size="large" style="width: 100%">
                <el-icon><Search /></el-icon>
                开始分类预测
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card v-if="predictionResult" class="result-card">
          <template #header>
            <div class="card-header">
              <el-icon><Check /></el-icon>
              <span>分类结果</span>
            </div>
          </template>
          <div class="result-content">
            <div class="prediction-main">
              <h2>{{ predictionResult.prediction }}</h2>
              <el-progress 
                :percentage="Math.round(predictionResult.confidence * 100)" 
                :color="getProgressColor(predictionResult.confidence)"
                :stroke-width="20"
                status="success"
              />
              <p class="confidence-text">置信度: {{ (predictionResult.confidence * 100).toFixed(2) }}%</p>
            </div>
            
            <el-divider />
            
            <div class="top-predictions">
              <h4>Top 3 预测结果</h4>
              <el-table :data="predictionResult.top_3_predictions" style="width: 100%">
                <el-table-column prop="species" label="石蝇种类" />
                <el-table-column prop="probability" label="概率">
                  <template #default="{ row }">
                    <el-progress :percentage="Math.round(row.probability * 100)" />
                  </template>
                </el-table-column>
              </el-table>
            </div>
          </div>
        </el-card>

        <el-card v-else class="empty-card">
          <el-empty description="请输入特征数据并点击预测按钮" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import { predictStonefly } from '../services/api'

const loading = ref(false)
const predictionResult = ref(null)
const formRef = ref()

const formData = reactive({
  lat: 30.5,
  lon: -95.0,
  country: 'US',
  family: 'Perlidae',
  body_length_mm: 15.0,
  color: 'brown',
  head_feature: 'small antenna'
})

const rules = {
  lat: [{ required: true, message: '请输入纬度', trigger: 'blur' }],
  lon: [{ required: true, message: '请输入经度', trigger: 'blur' }],
  country: [{ required: true, message: '请选择国家', trigger: 'change' }],
  family: [{ required: true, message: '请选择科', trigger: 'change' }],
  body_length_mm: [{ required: true, message: '请输入体长', trigger: 'blur' }],
  color: [{ required: true, message: '请选择颜色', trigger: 'change' }],
  head_feature: [{ required: true, message: '请选择头部特征', trigger: 'change' }]
}

const getProgressColor = (confidence: number) => {
  if (confidence >= 0.8) return '#67C23A'
  if (confidence >= 0.6) return '#E6A23C'
  return '#F56C6C'
}

const submitPrediction = async () => {
  try {
    await formRef.value.validate()
    loading.value = true
    
    const result = await predictStonefly(formData)
    if (result.success) {
      predictionResult.value = result
      ElMessage.success('预测成功！')
    } else {
      ElMessage.error(result.error || '预测失败')
    }
  } catch (error) {
    ElMessage.error('请求失败: ' + error.message)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.predict {
  padding: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: bold;
}

.result-card {
  min-height: 500px;
}

.result-content {
  padding: 20px 0;
}

.prediction-main {
  text-align: center;
  padding: 20px;
}

.prediction-main h2 {
  color: #409eff;
  margin-bottom: 20px;
  font-size: 28px;
}

.confidence-text {
  margin-top: 15px;
  color: #666;
  font-size: 16px;
}

.top-predictions {
  padding: 20px 0;
}

.top-predictions h4 {
  margin-bottom: 15px;
  color: #333;
}

.empty-card {
  min-height: 500px;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
