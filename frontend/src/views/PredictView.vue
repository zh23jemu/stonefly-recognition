<template>
  <div class="predict">
    <el-row :gutter="20">
      <el-col :xs="24" :lg="11">
        <el-card class="panel-card">
          <template #header>
            <div class="card-header">
              <el-icon><List /></el-icon>
              <span>验证集样本选择</span>
            </div>
          </template>

          <div class="filters">
            <el-input
              v-model="keyword"
              clearable
              placeholder="搜索物种、国家或科"
              @keyup.enter="loadSamples(1)"
              @clear="loadSamples(1)"
            />
            <el-select
              v-model="speciesFilter"
              clearable
              filterable
              placeholder="按真实物种筛选"
              @change="loadSamples(1)"
            >
              <el-option
                v-for="item in speciesOptions"
                :key="item"
                :label="item"
                :value="item"
              />
            </el-select>
            <el-button type="primary" :loading="sampleLoading" @click="loadSamples(1)">
              <el-icon><Search /></el-icon>
              查询
            </el-button>
          </div>

          <el-table
            v-loading="sampleLoading"
            :data="samples"
            border
            highlight-current-row
            class="sample-table"
            @row-click="selectSample"
          >
            <el-table-column prop="species" label="真实物种" min-width="180" show-overflow-tooltip />
            <el-table-column label="国家" width="80">
              <template #default="{ row }">{{ row.features.country }}</template>
            </el-table-column>
            <el-table-column label="科" min-width="130" show-overflow-tooltip>
              <template #default="{ row }">{{ row.features.family }}</template>
            </el-table-column>
            <el-table-column label="体长" width="90">
              <template #default="{ row }">
                {{ Number(row.features.body_length_mm).toFixed(1) }} mm
              </template>
            </el-table-column>
          </el-table>

          <div class="pagination-wrap">
            <el-pagination
              v-model:current-page="page"
              v-model:page-size="pageSize"
              layout="prev, pager, next, sizes, total"
              :page-sizes="[5, 10, 20, 50]"
              :total="total"
              @current-change="loadSamples"
              @size-change="handleSizeChange"
            />
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="13">
        <el-card class="panel-card result-card">
          <template #header>
            <div class="card-header">
              <el-icon><Check /></el-icon>
              <span>科级分类预测</span>
            </div>
          </template>

          <template v-if="selectedSample">
            <div class="sample-summary">
              <div>
                <span class="summary-label">真实物种</span>
                <strong>{{ selectedSample.species }}</strong>
              </div>
              <div>
                <span class="summary-label">真实科</span>
                <strong>{{ selectedSample.features.family }}</strong>
              </div>
              <el-tag effect="plain">验证集样本 #{{ selectedSample.id + 1 }}</el-tag>
            </div>

            <el-descriptions :column="2" border class="feature-summary">
              <el-descriptions-item label="纬度">{{ selectedSample.features.lat }}</el-descriptions-item>
              <el-descriptions-item label="经度">{{ selectedSample.features.lon }}</el-descriptions-item>
              <el-descriptions-item label="国家">{{ selectedSample.features.country }}</el-descriptions-item>
              <el-descriptions-item label="体长">
                {{ Number(selectedSample.features.body_length_mm).toFixed(1) }} mm
              </el-descriptions-item>
              <el-descriptions-item label="颜色">{{ selectedSample.features.color }}</el-descriptions-item>
              <el-descriptions-item label="头部特征">
                {{ selectedSample.features.head_feature }}
              </el-descriptions-item>
            </el-descriptions>

            <el-button
              type="primary"
              :loading="predictionLoading"
              class="predict-button"
              @click="submitPrediction"
            >
              <el-icon><DataAnalysis /></el-icon>
              预测所属科
            </el-button>

            <div v-if="predictionResult?.predicted_family" class="hierarchical-result">
              <div class="result-grid">
                <div class="result-block">
                  <span class="summary-label">预测科</span>
                  <strong>{{ predictionResult.predicted_family }}</strong>
                  <el-progress
                    :percentage="toPercent(predictionResult.family_confidence)"
                    :stroke-width="8"
                    status="success"
                  />
                </div>
                <div class="result-block">
                  <span class="summary-label">使用模型</span>
                  <strong>{{ (predictionResult.model_used || '-').toUpperCase() }}</strong>
                  <el-tag type="info" effect="plain">
                    Top-1 置信度 {{ toPercent(predictionResult.family_confidence) }}%
                  </el-tag>
                </div>
                <div class="result-status">
                  <el-tag :type="predictionResult.family_correct ? 'success' : 'danger'">
                    科{{ predictionResult.family_correct ? '正确' : '错误' }}
                  </el-tag>
                  <el-tag v-if="predictionResult.actual_family" effect="plain">
                    真实科：{{ predictionResult.actual_family }}
                  </el-tag>
                </div>
              </div>

              <el-table
                :data="predictionResult.top_3_family_predictions || []"
                border
                class="prediction-table"
              >
                <el-table-column type="index" label="#" width="60" />
                <el-table-column prop="family" label="Top-3候选科" min-width="220" show-overflow-tooltip />
                <el-table-column label="概率" width="180">
                  <template #default="{ row }">
                    <el-progress :percentage="toPercent(row.probability)" />
                  </template>
                </el-table-column>
                <el-table-column label="命中" width="90">
                  <template #default="{ row }">
                    <el-tag :type="row.family === selectedSample?.features.family ? 'success' : 'info'">
                      {{ row.family === selectedSample?.features.family ? '是' : '否' }}
                    </el-tag>
                  </template>
                </el-table-column>
              </el-table>
            </div>
          </template>

          <el-empty v-else description="请选择一条验证集样本" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import {
  getValidationSamples,
  predictStonefly,
  type PredictionResponse,
  type ValidationSample
} from '../services/api'

const samples = ref<ValidationSample[]>([])
const selectedSample = ref<ValidationSample | null>(null)
const predictionResult = ref<PredictionResponse | null>(null)
const speciesOptions = ref<string[]>([])
const keyword = ref('')
const speciesFilter = ref('')
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)
const sampleLoading = ref(false)
const predictionLoading = ref(false)

const loadSamples = async (targetPage = page.value) => {
  sampleLoading.value = true
  page.value = targetPage
  try {
    const result = await getValidationSamples({
      page: page.value,
      page_size: pageSize.value,
      keyword: keyword.value,
      species: speciesFilter.value
    })
    if (!result.success) {
      ElMessage.error(result.error || '验证集样本加载失败')
      return
    }
    samples.value = result.samples
    total.value = result.total
    speciesOptions.value = result.species_options
  } catch (error) {
    ElMessage.error(`验证集样本加载失败: ${(error as Error).message}`)
  } finally {
    sampleLoading.value = false
  }
}

const handleSizeChange = (size: number) => {
  pageSize.value = size
  loadSamples(1)
}

const selectSample = (sample: ValidationSample) => {
  selectedSample.value = sample
  predictionResult.value = null
}

const submitPrediction = async () => {
  if (!selectedSample.value) return

  predictionLoading.value = true
  try {
    const result = await predictStonefly({
      ...selectedSample.value.features,
      species: selectedSample.value.species
    })
    if (result.success) {
      predictionResult.value = result
      ElMessage.success('科级分类预测完成')
    } else {
      ElMessage.error(result.error || '预测失败')
    }
  } catch (error) {
    ElMessage.error(`预测失败: ${(error as Error).message}`)
  } finally {
    predictionLoading.value = false
  }
}

const toPercent = (value?: number) => {
  return Math.round((value || 0) * 100)
}

onMounted(() => {
  loadSamples()
})
</script>

<style scoped>
.predict {
  padding: 20px;
}

.panel-card {
  min-height: 620px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: bold;
}

.filters {
  display: grid;
  grid-template-columns: minmax(180px, 1fr) minmax(180px, 1fr) auto;
  gap: 12px;
  margin-bottom: 16px;
}

.sample-table,
.prediction-table {
  width: 100%;
}

.hierarchical-result {
  display: grid;
  gap: 16px;
}

.result-grid {
  display: grid;
  grid-template-columns: minmax(180px, 1fr) minmax(180px, 1fr) auto;
  gap: 14px;
  align-items: stretch;
}

.result-block {
  display: grid;
  gap: 8px;
  min-width: 0;
}

.result-block strong {
  overflow-wrap: anywhere;
  line-height: 1.35;
}

.result-status {
  display: flex;
  flex-direction: column;
  gap: 8px;
  justify-content: center;
}

.pagination-wrap {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}

.sample-summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.summary-label {
  display: block;
  color: #909399;
  font-size: 13px;
  margin-bottom: 6px;
}

.feature-summary {
  margin-bottom: 16px;
}

.predict-button {
  width: 100%;
  margin-bottom: 18px;
}

.top-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  line-height: 1.4;
}

@media (max-width: 900px) {
  .filters {
    grid-template-columns: 1fr;
  }

  .result-card {
    margin-top: 20px;
  }

  .sample-summary {
    align-items: flex-start;
    flex-direction: column;
  }

  .result-grid {
    grid-template-columns: 1fr;
  }

  .result-status {
    align-items: flex-start;
  }
}
</style>
