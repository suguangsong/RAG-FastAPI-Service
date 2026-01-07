#!/bin/bash

# Qdrant 启动脚本
# 用于启动已安装的 Qdrant 容器

set -e

echo "🚀 启动 Qdrant..."

# 检查 Qdrant 容器是否存在
if ! docker ps -a | grep -q qdrant; then
    echo "❌ Qdrant 容器不存在，请先运行 install_qdrant.sh"
    exit 1
fi

# 检查容器是否已在运行
if docker ps | grep -q qdrant; then
    echo "✅ Qdrant 已在运行中"
    docker ps | grep qdrant
    exit 0
fi

# 启动容器
echo "📦 启动 Qdrant 容器..."
docker start qdrant

# 等待启动
sleep 3

# 检查健康状态
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo "✅ Qdrant 启动成功！"
    echo ""
    echo "📊 Qdrant 信息:"
    echo "   - Web UI: http://localhost:6333/dashboard"
    echo "   - API: http://localhost:6333"
else
    echo "❌ Qdrant 启动失败，请检查日志: docker logs qdrant"
    exit 1
fi
