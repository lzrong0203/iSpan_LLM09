#!/bin/bash

# 環境設置腳本
# 用於設置 LLM 開發環境

echo "開始設置 LLM 開發環境..."

# 建議使用Python 3.9+
# 創建虛擬環境
python -m venv llm_env

# 啟動虛擬環境
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source llm_env/Scripts/activate
else
    # Linux/Mac
    source llm_env/bin/activate  
fi

# 安裝必要套件
echo "安裝必要的Python套件..."
pip install torch torchvision transformers
pip install numpy pandas matplotlib
pip install sentencepiece protobuf
pip install accelerate bitsandbytes

echo "環境設置完成！"