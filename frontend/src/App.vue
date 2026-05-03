<template>
  <div id="app">
    <el-container>
      <el-header>
        <div class="header-content">
          <h1>
            <el-icon><Monitor /></el-icon>
            石蝇分类系统
          </h1>
          <nav>
            <router-link to="/">首页</router-link>
            <router-link to="/predict">分类预测</router-link>
            <router-link to="/dashboard">数据大屏</router-link>
          </nav>
        </div>
      </el-header>
      <el-main>
        <router-view />
      </el-main>
      <el-footer>
        <p>石蝇分类系统 © 2026 - 基于机器学习的智能分类平台</p>
      </el-footer>
    </el-container>
  </div>
</template>

<script setup lang="ts">
</script>

<style scoped>
.el-header {
  background-color: #409eff;
  color: white;
  padding: 0 20px;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 100%;
}

.header-content h1 {
  margin: 0;
  font-size: 24px;
  display: flex;
  align-items: center;
  gap: 10px;
}

nav {
  display: flex;
  gap: 30px;
}

nav a {
  color: white;
  text-decoration: none;
  font-size: 16px;
  padding: 5px 15px;
  border-radius: 4px;
  transition: background-color 0.3s;
}

nav a:hover {
  background-color: rgba(255, 255, 255, 0.2);
}

nav a.router-link-active {
  background-color: rgba(255, 255, 255, 0.3);
  font-weight: bold;
}

.el-footer {
  background-color: #f5f7fa;
  color: #666;
  text-align: center;
  padding: 20px;
}

.el-main {
  min-height: calc(100vh - 120px);
  background-color: #f0f2f5;
}
</style>
