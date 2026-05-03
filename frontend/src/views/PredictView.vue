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
              <span>四模型预测对比</span>
            </div>
          </template>

          <template v-if="selectedSample">
            <div class="sample-summary">
              <div>
                <span class="summary-label">真实物种</span>
                <strong>{{ selectedSample.species }}</strong>
              </div>
              <el-tag effect="plain">验证集样本 #{{ selectedSample.id + 1 }}</el-tag>
            </div>

            <el-descriptions :column="2" border class="feature-summary">
              <el-descriptions-item label="纬度">{{ selectedSample.features.lat }}</el-descriptions-item>
              <el-descriptions-item label="经度">{{ selectedSample.features.lon }}</el-descriptions-item>
              <el-descriptions-item label="国家">{{ selectedSample.features.country }}</el-descriptions-item>
              <el-descriptions-item label="科">{{ selectedSample.features.family }}</el-descriptions-item>
              <el-descriptions-item label="体长">
                {{ Number(selectedSample.features.body_length_mm).toFixed(1) }} mm
              </el-descriptions-item>
              <el-descriptions-item label="颜色">{{ selectedSample.features.color }}</el-descriptions-item>
            </el-descriptions>

            <el-button
              type="primary"
              :loading="predictionLoading"
              class="predict-button"
              @click="submitPrediction"
            >
              <el-icon><DataAnalysis /></el-icon>
              使用四个模型预测
            </el-button>

            <el-table
              v-if="predictionResult?.model_predictions?.length"
              :data="predictionResult.model_predictions"
              border
              class="prediction-table"
            >
              <el-table-column label="模型" width="150">
                <template #default="{ row }">{{ formatModelName(row.model) }}</template>
              </el-table-column>
              <el-table-column prop="prediction" label="预测物种" min-width="190" show-overflow-tooltip />
              <el-table-column label="置信度" width="150">
                <template #default="{ row }">
                  <el-progress :percentage="Math.round(row.confidence * 100)" />
                </template>
              </el-table-column>
              <el-table-column label="结果" width="90">
                <template #default="{ row }">
                  <el-tag :type="row.correct ? 'success' : 'danger'">
                    {{ row.correct ? '正确' : '错误' }}
                  </el-tag>
                </template>
              </el-table-column>
              <el-table-column label="Top-3" min-width="220">
                <template #default="{ row }">
                  <div class="top-list">
                    <span v-for="item in row.top_3_predictions" :key="`${row.model}-${item.species}`">
                      {{ item.species }} {{ (item.probability * 100).toFixed(1) }}%
                    </span>
                  </div>
                </template>
              </el-table-column>
            </el-table>
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
      ElMessage.success('四模型预测完成')
    } else {
      ElMessage.error(result.error || '预测失败')
    }
  } catch (error) {
    ElMessage.error(`预测失败: ${(error as Error).message}`)
  } finally {
    predictionLoading.value = false
  }
}

const formatModelName = (name: string) => {
  const modelNames: Record<string, string> = {
    knn: 'KNN',
    random_forest: 'Random Forest',
    svm: 'SVM',
    xgboost: 'XGBoost'
  }
  return modelNames[name] || name
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
}
</style>
