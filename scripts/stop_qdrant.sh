#!/bin/bash

# Qdrant 停止脚本
# 用于停止 Qdrant 容器

set -e

echo "🛑 停止 Qdrant..."

if docker ps | grep -q qdrant; then
    docker stop qdrant
    echo "✅ Qdrant 已停止"
else
    echo "ℹ️  Qdrant 未在运行"
fi
