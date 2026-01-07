#!/bin/bash

# 启动脚本

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "复制环境变量模板..."
    cp env.example .env
    echo "请编辑 .env 文件配置 API Key 等信息"
fi

# 启动服务
echo "启动服务..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

