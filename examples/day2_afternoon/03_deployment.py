# Day 2 下午：部署方案
# 15:30-16:30 實際部署

import os
import json
from datetime import datetime

# === Step 1: 部署選項總覽 ===
print("=== AI應用部署方案 ===")
print()

deployment_options = {
    "Gradio Share": {
        "難度": "⭐",
        "費用": "免費",
        "特點": "一行程式碼分享",
        "適合": "快速原型、展示"
    },
    "Hugging Face Spaces": {
        "難度": "⭐⭐",
        "費用": "免費/付費",
        "特點": "永久託管、自動部署",
        "適合": "開源專案、Portfolio"
    },
    "Streamlit Cloud": {
        "難度": "⭐⭐",
        "費用": "免費",
        "特點": "GitHub整合",
        "適合": "資料應用、儀表板"
    },
    "Docker": {
        "難度": "⭐⭐⭐",
        "費用": "依平台",
        "特點": "容器化、可移植",
        "適合": "企業部署"
    },
    "雲端服務": {
        "難度": "⭐⭐⭐⭐",
        "費用": "按用量計費",
        "特點": "可擴展、高可用",
        "適合": "生產環境"
    }
}

for name, info in deployment_options.items():
    print(f"{name}:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

# === Step 2: Gradio快速分享 ===
print("=== Gradio快速分享 ===")
print()

gradio_code = '''import gradio as gr

def chatbot(message):
    return f"你說了：{message}"

# 建立介面
demo = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="我的AI應用"
)

# 啟動並分享
demo.launch(share=True)  # share=True 產生公開連結
'''

print("Gradio分享程式碼：")
print(gradio_code)
print()
print("執行後會得到：")
print("• 本地連結：http://127.0.0.1:7860")
print("• 公開連結：https://xxxxx.gradio.live (72小時有效)")
print()

# === Step 3: Hugging Face Spaces部署 ===
print("=== Hugging Face Spaces部署 ===")
print()

def create_hf_space_files():
    """建立Hugging Face Space所需檔案"""
    
    # app.py
    app_content = '''import gradio as gr
from transformers import pipeline

# 載入模型
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = classifier(text)[0]
    return f"{result['label']}: {result['score']:.2f}"

# 建立Gradio介面
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="輸入文字"),
    outputs=gr.Textbox(label="情感分析結果"),
    title="情感分析工具",
    description="分析文字的情感傾向"
)

if __name__ == "__main__":
    demo.launch()
'''
    
    # requirements.txt
    requirements = '''gradio==4.0.0
transformers==4.35.0
torch==2.1.0
'''
    
    # README.md
    readme = '''---
title: 情感分析工具
emoji: 😊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 情感分析工具

這是一個使用Transformers的情感分析應用。
'''
    
    print("需要建立的檔案：")
    print("\n1. app.py (主程式)")
    print("-" * 40)
    print(app_content[:200] + "...")
    
    print("\n2. requirements.txt (依賴)")
    print("-" * 40)
    print(requirements)
    
    print("3. README.md (配置)")
    print("-" * 40)
    print(readme[:150] + "...")
    
    return {
        "app.py": app_content,
        "requirements.txt": requirements,
        "README.md": readme
    }

# 建立檔案
hf_files = create_hf_space_files()

print("\n部署步驟：")
print("1. 在 huggingface.co 建立新Space")
print("2. 選擇Gradio SDK")
print("3. 上傳以上檔案")
print("4. 自動部署完成！")
print()

# === Step 4: Docker容器化 ===
print("=== Docker容器化 ===")
print()

dockerfile_content = '''# 使用Python基礎映像
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 複製依賴檔案
COPY requirements.txt .

# 安裝依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY . .

# 暴露端口
EXPOSE 7860

# 啟動應用
CMD ["python", "app.py"]
'''

docker_compose = '''version: '3.8'

services:
  ai-app:
    build: .
    ports:
      - "7860:7860"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
'''

print("Dockerfile:")
print("-" * 40)
print(dockerfile_content)

print("\ndocker-compose.yml:")
print("-" * 40)
print(docker_compose)

print("\nDocker指令：")
print("• 建立映像：docker build -t my-ai-app .")
print("• 執行容器：docker run -p 7860:7860 my-ai-app")
print("• 使用compose：docker-compose up -d")
print()

# === Step 5: 環境變數管理 ===
print("=== 環境變數管理 ===")
print()

def create_env_file():
    """建立環境變數檔案"""
    env_template = '''# OpenAI設定
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-3.5-turbo

# 應用設定
APP_PORT=7860
APP_HOST=0.0.0.0
DEBUG=False

# 資料庫設定（如果需要）
DATABASE_URL=sqlite:///app.db

# 安全設定
SECRET_KEY=your-secret-key-here
'''
    
    print(".env檔案範例：")
    print(env_template)
    
    print("\n在程式中使用：")
    print("```python")
    print("from dotenv import load_dotenv")
    print("import os")
    print()
    print("load_dotenv()")
    print("api_key = os.getenv('OPENAI_API_KEY')")
    print("```")

create_env_file()
print()

# === Step 6: 部署檢查清單 ===
print("=== 部署檢查清單 ===")
print()

checklist = [
    "□ API金鑰已設定為環境變數",
    "□ 依賴套件已列在requirements.txt",
    "□ 錯誤處理已完善",
    "□ 已設定適當的超時時間",
    "□ 已加入使用量限制",
    "□ 已準備監控和日誌",
    "□ 已測試各種輸入情況",
    "□ 已準備備援方案"
]

for item in checklist:
    print(item)

print()

# === Step 7: 成本估算 ===
print("=== 部署成本估算 ===")
print()

cost_estimation = {
    "開發階段": {
        "Gradio Share": "$0 (免費)",
        "HF Spaces": "$0 (免費版)",
        "本地Ollama": "$0 (自己的電腦)"
    },
    "小規模 (<100用戶/天)": {
        "HF Spaces": "$0-9/月",
        "Streamlit": "$0",
        "Render": "$7/月"
    },
    "中規模 (100-1000用戶/天)": {
        "AWS EC2": "$20-50/月",
        "Google Cloud": "$25-60/月",
        "Azure": "$30-70/月"
    },
    "大規模 (>1000用戶/天)": {
        "Kubernetes": "$100+/月",
        "Auto-scaling": "$200+/月",
        "CDN + 負載均衡": "$500+/月"
    }
}

for stage, options in cost_estimation.items():
    print(f"{stage}:")
    for platform, cost in options.items():
        print(f"  • {platform}: {cost}")
    print()

# === Step 8: 快速部署腳本 ===
print("=== 快速部署腳本 ===")
print()

deploy_script = '''#!/bin/bash

echo "🚀 開始部署AI應用..."

# 1. 檢查環境
echo "檢查Python版本..."
python --version

# 2. 建立虛擬環境
echo "建立虛擬環境..."
python -m venv venv
source venv/bin/activate

# 3. 安裝依賴
echo "安裝依賴..."
pip install -r requirements.txt

# 4. 設定環境變數
echo "設定環境變數..."
cp .env.example .env
echo "請編輯 .env 檔案設定API金鑰"

# 5. 測試應用
echo "測試應用..."
python test_app.py

# 6. 啟動應用
echo "啟動應用..."
python app.py

echo "✅ 部署完成！"
echo "訪問 http://localhost:7860"
'''

print("deploy.sh:")
print(deploy_script)

# === Step 9: 監控和維護 ===
print("\n=== 監控和維護 ===")
print()

monitoring_tips = [
    "📊 監控指標：回應時間、錯誤率、使用量",
    "📝 日誌記錄：所有API呼叫、錯誤、用戶行為",
    "🔔 警報設定：API額度、錯誤閾值、系統資源",
    "💾 備份策略：定期備份用戶資料和對話歷史",
    "🔄 更新計畫：模型更新、安全修補、功能迭代",
    "📈 效能優化：快取、批次處理、非同步處理"
]

for tip in monitoring_tips:
    print(tip)

# === Step 10: 總結 ===
print("\n" + "="*50)
print("=== 部署建議總結 ===")
print()

print("初學者路線：")
print("1️⃣ Gradio本地測試")
print("2️⃣ Gradio Share分享")
print("3️⃣ Hugging Face Spaces永久託管")
print()

print("進階路線：")
print("1️⃣ Docker容器化")
print("2️⃣ 雲端平台部署")
print("3️⃣ Kubernetes自動擴展")
print()

print("💡 記住：先讓它能運作，再讓它完美！")