<template>
  <div class="dashboard">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><PieChart /></el-icon>
              <span>模型性能监控大屏</span>
            </div>
          </template>
          
          <div v-if="modelInfo" class="dashboard-content">
            <el-row :gutter="20" class="stats-row">
              <el-col :span="6">
                <el-card class="stat-card">
                  <h3>最优模型</h3>
                  <p class="stat-value">{{ modelInfo.best_model.toUpperCase() }}</p>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card class="stat-card">
                  <h3>最优模型得分</h3>
                  <p class="stat-value">{{ (modelInfo.best_score * 100).toFixed(2) }}%</p>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card class="stat-card">
                  <h3>评估指标</h3>
                  <p class="stat-value">{{ modelInfo.metric_used.toUpperCase() }}</p>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card class="stat-card">
                  <h3>模型数量</h3>
                  <p class="stat-value">4</p>
                </el-card>
              </el-col>
            </el-row>

            <el-row :gutter="20" class="stats-row">
              <el-col :span="6">
                <el-card class="stat-card">
                  <h3>宏平均F1</h3>
                  <p class="stat-value">{{ formatPercent(bestModelMetrics.macro_f1) }}</p>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card class="stat-card">
                  <h3>平衡准确率</h3>
                  <p class="stat-value">{{ formatPercent(bestModelMetrics.balanced_accuracy) }}</p>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card class="stat-card">
                  <h3>Top-3准确率</h3>
                  <p class="stat-value">{{ formatPercent(bestModelMetrics.top_3_accuracy) }}</p>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card class="stat-card">
                  <h3>已知物种数</h3>
                  <p class="stat-value">{{ datasetSummary.class_count }}</p>
                </el-card>
              </el-col>
            </el-row>

            <el-row :gutter="20" class="stats-row">
              <el-col :span="6">
                <el-card class="stat-card">
                  <h3>验证集样本</h3>
                  <p class="stat-value">{{ datasetSummary.validation_samples }}</p>
                </el-card>
              </el-col>
              <el-col :span="18">
                <el-card class="stat-card">
                  <h3>指标说明</h3>
                  <p class="stat-desc">
                    当前训练流程已过滤 Unknown Stonefly，并使用完整7个输入特征训练模型；准确率为严格Top-1预测命中率。
                  </p>
                </el-card>
              </el-col>
            </el-row>

            <el-divider />

            <h3 class="section-title">模型性能对比</h3>
            <el-table :data="modelComparisonData" style="width: 100%" border>
              <el-table-column prop="model" label="模型" width="150" />
              <el-table-column prop="accuracy" label="准确率">
                <template #default="{ row }">
                  <el-progress :percentage="Math.round(row.accuracy * 100)" />
                </template>
              </el-table-column>
              <el-table-column prop="balanced_accuracy" label="平衡准确率">
                <template #default="{ row }">
                  <el-progress :percentage="Math.round(row.balanced_accuracy * 100)" />
                </template>
              </el-table-column>
              <el-table-column prop="precision" label="精确率">
                <template #default="{ row }">
                  <el-progress :percentage="Math.round(row.precision * 100)" />
                </template>
              </el-table-column>
              <el-table-column prop="recall" label="召回率">
                <template #default="{ row }">
                  <el-progress :percentage="Math.round(row.recall * 100)" />
                </template>
              </el-table-column>
              <el-table-column prop="f1_score" label="F1分数">
                <template #default="{ row }">
                  <el-progress :percentage="Math.round(row.f1_score * 100)" />
                </template>
              </el-table-column>
              <el-table-column prop="macro_f1" label="宏平均F1">
                <template #default="{ row }">
                  <el-progress :percentage="Math.round(row.macro_f1 * 100)" />
                </template>
              </el-table-column>
              <el-table-column prop="top_3_accuracy" label="Top-3准确率">
                <template #default="{ row }">
                  <el-progress :percentage="Math.round(row.top_3_accuracy * 100)" />
                </template>
              </el-table-column>
              <el-table-column prop="top_4_accuracy" label="Top-4准确率">
                <template #default="{ row }">
                  <el-progress :percentage="Math.round(row.top_4_accuracy * 100)" />
                </template>
              </el-table-column>
              <el-table-column prop="top_5_accuracy" label="Top-5准确率">
                <template #default="{ row }">
                  <el-progress :percentage="Math.round(row.top_5_accuracy * 100)" />
                </template>
              </el-table-column>
            </el-table>

            <el-divider />

            <h3 class="section-title">可视化图表</h3>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-card class="chart-card">
                  <template #header>模型对比图</template>
                  <img v-if="imageExists.model_comparison" 
                       :src="'/api/visualization/model_comparison.png'" 
                       alt="Model Comparison"
                       class="chart-image" />
                  <el-empty v-else description="图表加载中..." />
                </el-card>
              </el-col>
              <el-col :span="12">
                <el-card class="chart-card">
                  <template #header>类别分布</template>
                  <img v-if="imageExists.class_distribution" 
                       :src="'/api/visualization/class_distribution.png'" 
                       alt="Class Distribution"
                       class="chart-image" />
                  <el-empty v-else description="图表加载中..." />
                </el-card>
              </el-col>
            </el-row>

            <el-row :gutter="20" class="chart-row">
              <el-col :span="12">
                <el-card class="chart-card">
                  <template #header>特征分布</template>
                  <img v-if="imageExists.feature_distributions" 
                       :src="'/api/visualization/feature_distributions.png'" 
                       alt="Feature Distributions"
                       class="chart-image" />
                  <el-empty v-else description="图表加载中..." />
                </el-card>
              </el-col>
              <el-col :span="12">
                <el-card class="chart-card">
                  <template #header>相关矩阵</template>
                  <img v-if="imageExists.correlation_matrix" 
                       :src="'/api/visualization/correlation_matrix.png'" 
                       alt="Correlation Matrix"
                       class="chart-image" />
                  <el-empty v-else description="图表加载中..." />
                </el-card>
              </el-col>
            </el-row>
          </div>

          <div v-else class="loading-container">
            <el-empty description="暂无模型数据，请先训练模型">
              <el-button type="primary" @click="loadModelInfo">刷新数据</el-button>
            </el-empty>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { getModelInfo, type ModelInfo, type ModelMetric } from '../services/api'
import { ElMessage } from 'element-plus'

const modelInfo = ref<ModelInfo | null>(null)
const imageExists = ref({
  model_comparison: false,
  class_distribution: false,
  feature_distributions: false,
  correlation_matrix: false
})

const modelComparisonData = computed(() => {
  if (!modelInfo.value || !modelInfo.value.all_results) return []
  
  return Object.entries(modelInfo.value.all_results).map(([model, results]: [string, ModelMetric]) => ({
    model: model.toUpperCase(),
    accuracy: results.accuracy,
    balanced_accuracy: results.balanced_accuracy ?? 0,
    precision: results.precision,
    recall: results.recall,
    f1_score: results.f1_score,
    macro_f1: results.macro_f1 ?? 0,
    top_3_accuracy: results.top_3_accuracy ?? 0,
    top_4_accuracy: results.top_4_accuracy ?? results.top_3_accuracy ?? 0,
    top_5_accuracy: results.top_5_accuracy ?? results.top_4_accuracy ?? 0
  }))
})

const bestModelMetrics = computed(() => {
  if (!modelInfo.value || !modelInfo.value.all_results) {
    return {
      macro_f1: 0,
      balanced_accuracy: 0,
      top_3_accuracy: 0,
      top_4_accuracy: 0
    }
  }

  const key = modelInfo.value.best_model
  const result = modelInfo.value.all_results[key] || {}
  return {
    macro_f1: result.macro_f1 ?? 0,
    balanced_accuracy: result.balanced_accuracy ?? 0,
    top_3_accuracy: result.top_3_accuracy ?? 0,
    top_4_accuracy: result.top_4_accuracy ?? result.top_3_accuracy ?? 0
  }
})

const datasetSummary = computed(() => {
  const dataset = modelInfo.value?.dataset || {}
  return {
    class_count: dataset.class_count ?? 0,
    validation_samples: dataset.validation_samples ?? 0
  }
})

const formatPercent = (value: number) => {
  return `${(value * 100).toFixed(2)}%`
}

const loadModelInfo = async () => {
  try {
    const result = await getModelInfo()
    if (result.success && result.data) {
      modelInfo.value = result.data
      checkImages()
    }
  } catch (error) {
    ElMessage.warning('模型信息尚未生成，请先运行训练流程')
  }
}

const checkImages = () => {
  const images = [
    'model_comparison.png',
    'class_distribution.png',
    'feature_distributions.png',
    'correlation_matrix.png'
  ]
  
  images.forEach(img => {
    const key = img.replace('.png', '')
    fetch(`/api/visualization/${img}`)
      .then(res => {
        imageExists.value[key] = res.ok
      })
      .catch(() => {
        imageExists.value[key] = false
      })
  })
}

onMounted(() => {
  loadModelInfo()
})
</script>

<style scoped>
.dashboard {
  padding: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 20px;
  font-weight: bold;
}

.stats-row {
  margin-bottom: 20px;
}

.stat-card {
  text-align: center;
  padding: 20px;
}

.stat-card h3 {
  color: #666;
  font-size: 14px;
  margin-bottom: 10px;
}

.stat-value {
  color: #409eff;
  font-size: 24px;
  font-weight: bold;
  margin: 0;
}

.stat-desc {
  color: #909399;
  font-size: 13px;
  line-height: 1.6;
  margin: 0;
}

.section-title {
  margin: 20px 0;
  color: #333;
  font-size: 18px;
}

.chart-row {
  margin-top: 20px;
}

.chart-card {
  text-align: center;
}

.chart-image {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
}

.loading-container {
  padding: 60px;
  text-align: center;
}
</style>
