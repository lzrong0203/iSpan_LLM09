# Day 2 下午：Ollama本地部署
# 13:30-14:30 本地LLM部署

import subprocess
import requests
import json
import time

# === Step 1: Ollama介紹 ===
print("=== Ollama本地LLM部署 ===")
print()
print("Ollama = 本地運行的ChatGPT")
print()
print("優點：")
print("✅ 完全免費")
print("✅ 資料不外流")
print("✅ 離線可用")
print("✅ 支援多種模型")
print()

# === Step 2: 安裝指南 ===
print("=== 安裝Ollama ===")
print()
print("1. Windows/Mac：")
print("   訪問 https://ollama.ai 下載安裝檔")
print()
print("2. Linux：")
print("   curl -fsSL https://ollama.ai/install.sh | sh")
print()
print("3. 驗證安裝：")
print("   在終端機執行：ollama --version")
print()

# === Step 3: 下載模型 ===
print("=== 下載模型 ===")
print()
print("常用模型：")
models = [
    {"name": "llama2", "size": "3.8GB", "desc": "Meta的開源模型"},
    {"name": "mistral", "size": "4.1GB", "desc": "輕量但強大"},
    {"name": "codellama", "size": "3.8GB", "desc": "程式碼專用"},
    {"name": "phi", "size": "1.6GB", "desc": "微軟的小模型"},
    {"name": "gemma", "size": "1.4GB", "desc": "Google的輕量模型"}
]

for model in models:
    print(f"• {model['name']:10} ({model['size']:5}) - {model['desc']}")

print()
print("下載模型指令：")
print("ollama pull llama2")
print()

# === Step 4: Ollama API客戶端 ===
class OllamaClient:
    """Ollama API客戶端"""
    
    def __init__(self, host="http://localhost:11434"):
        self.host = host
        self.api_generate = f"{host}/api/generate"
        self.api_chat = f"{host}/api/chat"
        self.api_tags = f"{host}/api/tags"
    
    def check_connection(self) -> bool:
        """檢查Ollama是否運行"""
        try:
            response = requests.get(self.api_tags)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> list:
        """列出已安裝的模型"""
        try:
            response = requests.get(self.api_tags)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            print(f"錯誤：{e}")
            return []
    
    def generate(self, model: str, prompt: str, stream: bool = False) -> str:
        """生成回應"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        try:
            response = requests.post(self.api_generate, json=payload)
            if response.status_code == 200:
                data = response.json()
                return data.get('response', '')
            return f"錯誤：{response.status_code}"
        except Exception as e:
            return f"錯誤：{e}"
    
    def chat(self, model: str, messages: list, stream: bool = False) -> str:
        """對話模式"""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        try:
            response = requests.post(self.api_chat, json=payload)
            if response.status_code == 200:
                data = response.json()
                return data.get('message', {}).get('content', '')
            return f"錯誤：{response.status_code}"
        except Exception as e:
            return f"錯誤：{e}"

# === Step 5: 測試Ollama ===
print("=== 測試Ollama連線 ===")
print()

ollama = OllamaClient()

if ollama.check_connection():
    print("✅ Ollama正在運行")
    
    # 列出模型
    models = ollama.list_models()
    if models:
        print(f"\n已安裝的模型：")
        for model in models:
            print(f"  • {model}")
    else:
        print("\n❌ 尚未安裝任何模型")
        print("請執行：ollama pull llama2")
else:
    print("❌ Ollama未運行")
    print("\n請先啟動Ollama：")
    print("1. 確保已安裝Ollama")
    print("2. 執行：ollama serve")

print()

# === Step 6: 簡單對話範例 ===
def ollama_chat_demo():
    """Ollama對話示範"""
    print("=== Ollama對話示範 ===")
    print()
    
    # 檢查連線
    if not ollama.check_connection():
        print("請先啟動Ollama服務")
        return
    
    # 選擇模型
    models = ollama.list_models()
    if not models:
        print("請先下載模型：ollama pull llama2")
        return
    
    model = models[0]  # 使用第一個可用模型
    print(f"使用模型：{model}")
    print()
    
    # 對話迴圈
    messages = []
    print("開始對話（輸入'quit'結束）")
    print("-" * 40)
    
    while True:
        user_input = input("👤 你：")
        if user_input.lower() == 'quit':
            break
        
        # 添加使用者訊息
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # 取得回應
        print("🤖 AI：", end="", flush=True)
        response = ollama.chat(model, messages)
        print(response)
        
        # 添加AI回應
        messages.append({
            "role": "assistant",
            "content": response
        })
        print()

# === Step 7: 比較不同模型 ===
def compare_models():
    """比較不同模型的回應"""
    print("=== 比較不同模型 ===")
    print()
    
    if not ollama.check_connection():
        print("Ollama未運行")
        return
    
    models = ollama.list_models()
    if len(models) < 2:
        print("需要至少2個模型來比較")
        return
    
    prompt = "用一句話解釋什麼是人工智慧"
    
    print(f"測試提示：{prompt}")
    print("=" * 50)
    
    for model in models[:3]:  # 最多比較3個模型
        print(f"\n模型：{model}")
        start_time = time.time()
        response = ollama.generate(model, prompt)
        elapsed = time.time() - start_time
        print(f"回應：{response}")
        print(f"時間：{elapsed:.2f}秒")

# === Step 8: 本地RAG系統 ===
class LocalRAG:
    """使用Ollama的本地RAG系統"""
    
    def __init__(self, model="llama2"):
        self.model = model
        self.ollama = OllamaClient()
        self.knowledge_base = []
    
    def add_knowledge(self, text: str):
        """添加知識"""
        self.knowledge_base.append(text)
    
    def answer(self, question: str) -> str:
        """回答問題"""
        if not self.knowledge_base:
            return "知識庫為空"
        
        # 建立上下文
        context = "\n".join(self.knowledge_base)
        
        # 建立提示
        prompt = f"""基於以下資料回答問題：

資料：
{context}

問題：{question}

答案："""
        
        return self.ollama.generate(self.model, prompt)

# === Step 9: 執行示範 ===
print("=== 選擇功能 ===")
print()
print("1. 測試Ollama連線")
print("2. 互動對話")
print("3. 比較模型")
print("4. 本地RAG系統")
print()

choice = input("選擇 (1-4)：")

if choice == "1":
    # 已在上面執行
    pass
elif choice == "2":
    ollama_chat_demo()
elif choice == "3":
    compare_models()
elif choice == "4":
    print("\n=== 本地RAG系統 ===")
    rag = LocalRAG()
    
    # 添加知識
    rag.add_knowledge("Python是1991年由Guido van Rossum創造的程式語言")
    rag.add_knowledge("Python強調程式碼的可讀性和簡潔性")
    
    # 測試問答
    question = "Python是什麼時候創造的？"
    print(f"問題：{question}")
    print(f"答案：{rag.answer(question)}")

# === Step 10: 部署建議 ===
print("\n" + "="*50)
print("=== Ollama部署建議 ===")
print()

tips = [
    "💾 硬碟空間：每個模型需要2-8GB",
    "💻 記憶體：建議至少8GB RAM",
    "🚀 GPU加速：支援NVIDIA顯卡加速",
    "🌐 API服務：可作為API服務供其他應用呼叫",
    "🔒 安全性：預設只監聽localhost",
    "📦 Docker：支援Docker容器部署"
]

for tip in tips:
    print(tip)

print()
print("💡 Ollama讓每個人都能運行自己的AI！")