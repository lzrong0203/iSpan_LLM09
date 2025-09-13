# Day 1 下午：AI聊天機器人 - Ollama版本
# 使用 Ollama gemma3:270m 模型

import ollama
from datetime import datetime

# === Step 1: 初始化Ollama ===
print("=== AI聊天機器人 (Ollama gemma3:270m) ===")
print()

# === Step 2: 基本AI Chatbot ===
class OllamaChatbot:
    """
    使用Ollama API的聊天機器人
    """
    def __init__(self, model="gemma3:270m"):
        self.model = model
        self.messages = []
        self.total_tokens = 0

    def add_system_message(self, content):
        """設定系統角色"""
        self.messages.append({
            "role": "system",
            "content": content
        })

    def chat(self, user_input):
        """與AI對話"""
        # 加入使用者訊息
        self.messages.append({
            "role": "user",
            "content": user_input
        })

        try:
            # 呼叫Ollama API
            response = ollama.chat(
                model=self.model,
                messages=self.messages
            )

            # 取得回應
            ai_response = response['message']['content']

            # 記錄token使用 (如果有提供)
            if 'eval_count' in response:
                self.total_tokens += response.get('prompt_eval_count', 0) + response.get('eval_count', 0)

            # 加入AI回應到歷史
            self.messages.append({
                "role": "assistant",
                "content": ai_response
            })

            tokens_used = response.get('eval_count', 0)
            return ai_response, tokens_used

        except Exception as e:
            return f"錯誤：{e}", 0

    def reset(self):
        """重置對話"""
        self.messages = []
        self.total_tokens = 0

# === Step 3: 不同角色的Chatbot ===
def create_role_chatbot(role_description):
    """建立特定角色的Chatbot"""
    bot = OllamaChatbot()
    bot.add_system_message(role_description)
    return bot

# 預設角色
roles = {
    "助手": "你是一個友善的AI助手，用繁體中文回答",
    "老師": "你是一個有耐心的程式設計老師，會用簡單的方式解釋概念",
    "翻譯": "你是一個專業翻譯，將使用者的話翻譯成英文",
    "詩人": "你是一個詩人，用優美的詩句回應"
}

# === Step 4: 互動式對話 ===
def interactive_ai_chat():
    """互動式AI對話"""
    print("選擇AI角色：")
    for i, (name, desc) in enumerate(roles.items(), 1):
        print(f"{i}. {name}: {desc[:30]}...")
    print()

    choice = input("選擇 (1-4)：")

    # 建立Chatbot
    role_name = list(roles.keys())[int(choice)-1] if choice.isdigit() else "助手"
    bot = create_role_chatbot(roles[role_name])

    print(f"\n🤖 AI {role_name} 已啟動 (使用 gemma3:270m)")
    print("輸入'quit'結束，'reset'重置對話")
    print("-" * 40)

    while True:
        user_input = input("👤 你：")

        if user_input.lower() == 'quit':
            print(f"\n總共使用約 {bot.total_tokens} tokens")
            print("👋 再見！")
            break

        if user_input.lower() == 'reset':
            bot.reset()
            bot.add_system_message(roles[role_name])
            print("✨ 對話已重置")
            continue

        # 取得AI回應
        response, tokens = bot.chat(user_input)
        print(f"🤖 AI：{response}")
        if tokens > 0:
            print(f"   (生成 {tokens} tokens)")
        print()

# === Step 5: 進階功能 ===
class AdvancedOllamaChatbot(OllamaChatbot):
    """
    進階Chatbot with 額外功能
    """
    def __init__(self, model="gemma3:270m"):
        super().__init__(model)
        self.conversation_log = []

    def save_conversation(self, filename=None):
        """儲存對話"""
        if not filename:
            filename = f"ollama_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"=== Ollama對話記錄 ===\n")
            f.write(f"時間：{datetime.now()}\n")
            f.write(f"模型：{self.model}\n")
            f.write(f"總Token：{self.total_tokens}\n")
            f.write("="*40 + "\n\n")

            for msg in self.messages:
                if msg['role'] != 'system':
                    role = "使用者" if msg['role'] == 'user' else "AI"
                    f.write(f"{role}：{msg['content']}\n\n")

        print(f"對話已儲存到 {filename}")

    def summarize_conversation(self):
        """總結對話"""
        if len(self.messages) < 3:
            return "對話太短，無法總結"

        # 建立總結請求
        summary_request = "請用3句話總結我們的對話重點"
        response, _ = self.chat(summary_request)

        return response

# === Step 6: 測試模型連接 ===
def test_model_connection():
    """測試Ollama連接和模型可用性"""
    try:
        # 測試連接
        models = ollama.list()
        model_names = [m['name'] for m in models['models']]

        if 'gemma3:270m' not in model_names:
            print("⚠️  警告：gemma3:270m 模型未安裝")
            print("請執行：ollama pull gemma3:270m")
            return False

        print("✅ Ollama連接成功，gemma3:270m 模型已就緒")
        return True

    except Exception as e:
        print(f"❌ Ollama連接失敗：{e}")
        print("請確認 Ollama 服務是否正在運行")
        return False

# === Step 7: 執行程式 ===
print("=== Ollama Chatbot (gemma3:270m) ===")
print()

# 測試連接
if not test_model_connection():
    print("\n請先啟動 Ollama 服務並安裝模型")
    exit(1)

print("\n1. 基本對話模式")
print("2. 角色扮演模式")
print("3. 進階功能模式")
print()

mode = input("選擇模式 (1-3)：")

if mode == "1":
    # 基本模式
    bot = OllamaChatbot()
    print("\n基本AI對話模式 (gemma3:270m)")
    print("-" * 40)

    while True:
        user_input = input("👤 你：")
        if user_input.lower() == 'quit':
            break
        response, _ = bot.chat(user_input)
        print(f"🤖 AI：{response}\n")

elif mode == "2":
    # 角色模式
    interactive_ai_chat()

elif mode == "3":
    # 進階模式
    bot = AdvancedOllamaChatbot()
    bot.add_system_message("你是一個友善的AI助手")

    print("\n進階AI對話模式 (gemma3:270m)")
    print("指令：quit(結束), save(儲存), summary(總結)")
    print("-" * 40)

    while True:
        user_input = input("👤 你：")

        if user_input.lower() == 'quit':
            bot.save_conversation()
            break
        elif user_input.lower() == 'save':
            bot.save_conversation()
            continue
        elif user_input.lower() == 'summary':
            summary = bot.summarize_conversation()
            print(f"📝 總結：{summary}\n")
            continue

        response, _ = bot.chat(user_input)
        print(f"🤖 AI：{response}\n")

else:
    print("無效選擇")