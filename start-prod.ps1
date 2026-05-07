# 石蝇分类系统生产启动脚本
# 用途：
# 1. 检查项目关键目录与本地虚拟环境
# 2. 如缺少 waitress，则自动安装到项目本地 .venv
# 3. 构建前端生产包
# 4. 后台启动 Flask(Waitress) 与前端 Vite Preview
# 5. 把日志输出到 logs/prod，便于排查问题

param(
    [int]$BackendPort = 5000,
    [int]$FrontendPort = 4173,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ProjectRoot "backend"
$FrontendDir = Join-Path $ProjectRoot "frontend"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvPip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"
$WaitressExe = Join-Path $ProjectRoot ".venv\Scripts\waitress-serve.exe"
$LogsDir = Join-Path $ProjectRoot "logs\prod"
$BackendStdout = Join-Path $LogsDir "backend.stdout.log"
$BackendStderr = Join-Path $LogsDir "backend.stderr.log"
$FrontendStdout = Join-Path $LogsDir "frontend.stdout.log"
$FrontendStderr = Join-Path $LogsDir "frontend.stderr.log"
$PidFile = Join-Path $LogsDir "prod-processes.json"

Write-Step "检查项目目录"
if (-not (Test-Path $BackendDir)) {
    throw "未找到后端目录：$BackendDir"
}
if (-not (Test-Path $FrontendDir)) {
    throw "未找到前端目录：$FrontendDir"
}
if (-not (Test-Path $VenvPython)) {
    throw "未找到项目本地 Python：$VenvPython"
}
if (-not (Test-Path $VenvPip)) {
    throw "未找到项目本地 pip：$VenvPip"
}

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

Write-Step "检查 waitress"
if (-not (Test-Path $WaitressExe)) {
    & $VenvPip install waitress
}

if (-not $SkipBuild) {
    Write-Step "构建前端生产包"
    Push-Location $FrontendDir
    try {
        & npm.cmd run build
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Step "跳过前端构建（--SkipBuild）"
}

Write-Step "后台启动后端服务"
$BackendProcess = Start-Process `
    -FilePath $WaitressExe `
    -ArgumentList @("--host=0.0.0.0", "--port=$BackendPort", "run:app") `
    -WorkingDirectory $BackendDir `
    -RedirectStandardOutput $BackendStdout `
    -RedirectStandardError $BackendStderr `
    -WindowStyle Hidden `
    -PassThru

Write-Step "后台启动前端预览服务"
$FrontendProcess = Start-Process `
    -FilePath "npm.cmd" `
    -ArgumentList @("run", "preview", "--", "--host", "0.0.0.0", "--port", "$FrontendPort") `
    -WorkingDirectory $FrontendDir `
    -RedirectStandardOutput $FrontendStdout `
    -RedirectStandardError $FrontendStderr `
    -WindowStyle Hidden `
    -PassThru

$PidInfo = [ordered]@{
    backend_pid = $BackendProcess.Id
    frontend_pid = $FrontendProcess.Id
    backend_port = $BackendPort
    frontend_port = $FrontendPort
    started_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    backend_stdout_log = $BackendStdout
    backend_stderr_log = $BackendStderr
    frontend_stdout_log = $FrontendStdout
    frontend_stderr_log = $FrontendStderr
}
$PidInfo | ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8

Write-Step "启动完成"
Write-Host "前端地址: http://localhost:$FrontendPort" -ForegroundColor Green
Write-Host "后端地址: http://localhost:$BackendPort" -ForegroundColor Green
Write-Host "进程信息: $PidFile"
Write-Host "后端日志: $BackendStdout"
Write-Host "前端日志: $FrontendStdout"
Write-Host ""
Write-Host "停止服务可执行以下命令：" -ForegroundColor Yellow
Write-Host "Stop-Process -Id $($BackendProcess.Id),$($FrontendProcess.Id)"
