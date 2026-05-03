<template>
  <div class="home">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="welcome-card">
          <template #header>
            <div class="card-header">
              <el-icon><TrendCharts /></el-icon>
              <span>欢迎使用石蝇分类系统</span>
            </div>
          </template>
          <div class="intro-content">
            <p>本系统基于机器学习算法，通过分析石蝇的形态特征数据，实现对不同石蝇种类的智能分类。</p>
            <el-row :gutter="20" class="stats-row">
              <el-col :span="6">
                <el-statistic title="总样本数" :value="99901" />
              </el-col>
              <el-col :span="6">
                <el-statistic title="特征维度" :value="7" />
              </el-col>
              <el-col :span="6">
                <el-statistic title="分类类别" :value="25" />
              </el-col>
              <el-col :span="6">
                <el-statistic title="模型准确率" :value="95" suffix="%" />
              </el-col>
            </el-row>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="feature-row">
      <el-col :span="8">
        <el-card class="feature-card" @click="$router.push('/predict')">
          <el-icon class="feature-icon"><Aim /></el-icon>
          <h3>智能分类</h3>
          <p>输入石蝇特征数据，快速获得分类结果</p>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card class="feature-card" @click="$router.push('/dashboard')">
          <el-icon class="feature-icon"><DataAnalysis /></el-icon>
          <h3>数据可视化</h3>
          <p>查看模型性能和数据分析可视化图表</p>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card class="feature-card">
          <el-icon class="feature-icon"><Cpu /></el-icon>
          <h3>多模型对比</h3>
          <p>集成随机森林、SVM、XGBoost等多种算法</p>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
</script>

<style scoped>
.home {
  padding: 20px;
}

.welcome-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 20px;
  font-weight: bold;
}

.intro-content {
  text-align: center;
  padding: 20px 0;
}

.intro-content p {
  font-size: 16px;
  color: #666;
  margin-bottom: 30px;
}

.stats-row {
  margin-top: 20px;
}

.feature-row {
  margin-top: 20px;
}

.feature-card {
  text-align: center;
  padding: 30px;
  cursor: pointer;
  transition: transform 0.3s, box-shadow 0.3s;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
}

.feature-icon {
  font-size: 48px;
  color: #409eff;
  margin-bottom: 15px;
}

.feature-card h3 {
  margin: 10px 0;
  color: #333;
}

.feature-card p {
  color: #666;
  font-size: 14px;
}
</style>
