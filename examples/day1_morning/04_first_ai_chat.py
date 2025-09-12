# Day 1 上午：第一個AI對話
# 11:00-11:30 實作簡單對話

import os
from dotenv import load_dotenv
from openai import OpenAI

# 載入環境變數
load_dotenv()

# === Step 1: 初始化OpenAI ===
print("=== 第一個AI對話程式 ===")
print()

# 建立客戶端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# === Step 2: 基本對話函數 ===
def chat_with_ai(user_message, model="gpt-3.5-turbo"):
    """
    與AI對話的函數
    
    參數：
    - user_message: 使用者的訊息
    - model: 使用的模型
    
    回傳：
    - AI的回應
    """
    try:
        # 呼叫API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,  # 創意程度 (0-1)
            max_tokens=150    # 最大回應長度
        )
        
        # 取得回應
        return response.choices[0].message.content
        
    except Exception as e:
        return f"錯誤：{e}"

# === Step 3: 測試基本對話 ===
print("=== 測試基本對話 ===")
print()

# 測試問題
test_questions = [
    "台灣最高的山是什麼？",
    "1+1等於多少？",
    "用一句話介紹Python"
]

for question in test_questions:
    print(f"👤 問：{question}")
    answer = chat_with_ai(question)
    print(f"🤖 答：{answer}")
    print("-" * 50)
    print()

# === Step 4: 加入系統角色 ===
def chat_with_role(user_message, system_role="你是一個友善的助手"):
    """
    帶有角色設定的對話
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"錯誤：{e}"

print("=== 測試不同角色 ===")
print()

# 測試不同角色
roles_and_questions = [
    ("你是一個Python老師", "什麼是變數？"),
    ("你是一個搞笑藝人", "為什麼雞要過馬路？"),
    ("你是一個詩人", "描述今天的天氣")
]

for role, question in roles_and_questions:
    print(f"🎭 角色：{role}")
    print(f"👤 問：{question}")
    answer = chat_with_role(question, role)
    print(f"🤖 答：{answer}")
    print("-" * 50)
    print()

# === Step 5: 互動式對話 ===
def interactive_chat():
    """
    互動式對話模式
    """
    print("=== 互動式對話模式 ===")
    print("輸入'quit'結束對話")
    print()
    
    # 對話歷史
    messages = [
        {"role": "system", "content": "你是一個友善的AI助手"}
    ]
    
    while True:
        # 取得使用者輸入
        user_input = input("👤 你：")
        
        if user_input.lower() == 'quit':
            print("👋 再見！")
            break
        
        # 加入使用者訊息
        messages.append({"role": "user", "content": user_input})
        
        try:
            # 呼叫API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7
            )
            
            # 取得回應
            ai_response = response.choices[0].message.content
            print(f"🤖 AI：{ai_response}")
            print()
            
            # 加入AI回應到歷史
            messages.append({"role": "assistant", "content": ai_response})
            
        except Exception as e:
            print(f"❌ 錯誤：{e}")
            break

# === Step 6: 參數調整實驗 ===
print("=== Temperature參數實驗 ===")
print("Temperature控制創意程度（0=保守, 1=創意）")
print()

prompt = "寫一個關於貓的句子"

for temp in [0.1, 0.5, 0.9]:
    print(f"Temperature = {temp}")
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=50
    )
    
    print(f"回應：{response.choices[0].message.content}")
    print()

# === Step 7: 執行互動模式 ===
print("\n" + "="*50)
print("要開始互動對話嗎？")
start = input("輸入 'yes' 開始，其他鍵跳過：")

if start.lower() == 'yes':
    interactive_chat()