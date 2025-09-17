# Day 1 上午：設定OpenAI API
# 10:30-11:00 環境設定與第一個API呼叫

import os

from dotenv import load_dotenv

# === Step 1: 載入環境變數 ===
print("=== 設定OpenAI API ===")
print()

# 載入.env檔案
load_dotenv()

# === Step 2: 設定API Key ===
# 方法1：從環境變數讀取（推薦）
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("✅ API Key已載入")
    print(f"   Key開頭：{api_key[:10]}...")
else:
    print("❌ 找不到API Key")
    print("請在.env檔案中設定：")
    print("OPENAI_API_KEY=sk-your-key-here")
print()

# === Step 3: 安裝套件 ===
print("=== 需要安裝的套件 ===")
print("在終端機執行：")
print("pip install openai python-dotenv")
print()

# === Step 4: 測試連線 ===
try:
    from openai import OpenAI

    # 初始化客戶端
    client = OpenAI(api_key=api_key)

    print("=== 測試API連線 ===")

    # 簡單的測試請求
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "說『你好』"}],
        max_tokens=10,
    )

    print("✅ 連線成功！")
    print(f"AI回應：{response.choices[0].message.content}")

except Exception as e:
    print("❌ 連線失敗")
    print(f"錯誤訊息：{e}")
    print()
    print("可能的原因：")
    print("1. API Key錯誤")
    print("2. 網路連線問題")
    print("3. 額度用完")

print()

# === Step 5: 建立.env檔案範例 ===
print("=== .env檔案範例 ===")
print("在專案根目錄建立.env檔案：")
print()
print("# OpenAI設定")
print("OPENAI_API_KEY=sk-your-api-key-here")
print()
print("# 其他設定（選用）")
print("OPENAI_MODEL=gpt-3.5-turbo")
print("MAX_TOKENS=1000")
print()

# === Step 6: 安全提醒 ===
print("=== 安全提醒 ===")
print("⚠️  永遠不要把API Key寫在程式碼裡")
print("⚠️  .env檔案要加入.gitignore")
print("⚠️  不要分享你的API Key給別人")
print()

# === Step 7: 檢查.gitignore ===
print("=== 檢查.gitignore ===")
gitignore_content = """
# 環境變數
.env
.env.local

# Python
__pycache__/
*.py[cod]
*$py.class

# 虛擬環境
venv/
env/
"""

print("確保.gitignore包含：")
print(gitignore_content)
