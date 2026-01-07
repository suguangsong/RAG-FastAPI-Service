#!/bin/bash

# Qdrant 安装脚本 (Ubuntu)
# 用于在 Ubuntu 系统上安装和配置 Qdrant 向量数据库

set -e

echo "🚀 开始安装 Qdrant..."

# 检查是否为 root 用户
if [ "$EUID" -eq 0 ]; then
    echo "❌ 请不要使用 root 用户运行此脚本"
    exit 1
fi

# 检查系统版本
if ! grep -q "Ubuntu" /etc/os-release; then
    echo "⚠️  警告: 此脚本针对 Ubuntu 系统设计，其他系统可能不兼容"
fi

# 安装 Docker（如果未安装）
if ! command -v docker &> /dev/null; then
    echo "📦 安装 Docker..."
    sudo apt-get update
    sudo apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # 将当前用户添加到 docker 组
    sudo usermod -aG docker $USER
    echo "✅ Docker 安装完成，请重新登录以使 docker 组权限生效"
fi

# 创建 Qdrant 数据目录
QDRANT_DATA_DIR="${HOME}/qdrant_storage"
mkdir -p "$QDRANT_DATA_DIR"
echo "📁 Qdrant 数据目录: $QDRANT_DATA_DIR"

# 检查 Qdrant 容器是否已存在
if docker ps -a | grep -q qdrant; then
    echo "⚠️  Qdrant 容器已存在，正在停止并删除..."
    docker stop qdrant 2>/dev/null || true
    docker rm qdrant 2>/dev/null || true
fi

# 启动 Qdrant 容器
echo "🚀 启动 Qdrant 容器..."
docker run -d \
    --name qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -v "${QDRANT_DATA_DIR}:/qdrant/storage" \
    --restart unless-stopped \
    qdrant/qdrant:latest

# 等待 Qdrant 启动
echo "⏳ 等待 Qdrant 启动..."
sleep 5

# 检查 Qdrant 健康状态
max_attempts=10
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo "✅ Qdrant 启动成功！"
        echo ""
        echo "📊 Qdrant 信息:"
        echo "   - Web UI: http://localhost:6333/dashboard"
        echo "   - API: http://localhost:6333"
        echo "   - 数据目录: $QDRANT_DATA_DIR"
        break
    fi
    attempt=$((attempt + 1))
    echo "   尝试 $attempt/$max_attempts..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Qdrant 启动失败，请检查日志: docker logs qdrant"
    exit 1
fi

echo ""
echo "🎉 Qdrant 安装完成！"
