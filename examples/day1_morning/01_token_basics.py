# Day 1 上午：Token是什麼？
# 09:30-10:00 Token基礎概念

import tiktoken

# === Step 0: 使用 OpenAI tiktoken 實際計算 Token ===
print("=== 使用 tiktoken 實際計算 Token ===")
print("tiktoken 是 OpenAI 官方的 Token 計算工具")
print()

# 載入不同模型的編碼器
encoding_gpt35 = tiktoken.encoding_for_model("gpt-3.5-turbo")
encoding_gpt4 = tiktoken.encoding_for_model("gpt-4")

# 測試文字
test_texts = [
    "我愛台灣",
    "Hello World!",
    "今天天氣很好",
    "Machine Learning is fascinating",
    "人工智慧大型語言模型實作應用班"
]

print("不同文字的 Token 計算結果：")
print("-" * 60)
for text in test_texts:
    tokens = encoding_gpt35.encode(text)
    token_count = len(tokens)
    
    # 顯示每個 token
    token_strings = [encoding_gpt35.decode([token]) for token in tokens]
    
    print(f"文字：{text}")
    print(f"Token 數量：{token_count}")
    print(f"Token 切分：{token_strings}")
    print(f"Token IDs：{tokens}")
    print()

print("-" * 60)
print()

# === Step 1: Token就像拼圖片 ===
text = "我愛台灣"
print(f"原始文字：{text}")
print()

# 模擬Token化過程
print("=== Token化過程（簡化說明） ===")
print("Step 1: 把文字切成小塊")
tokens = ["我", "愛", "台", "灣"]
for i, token in enumerate(tokens):
    print(f"  Token {i+1}: {token}")
print(f"\n總共：{len(tokens)} 個Token（簡化版）")

# 實際的 tiktoken 結果
actual_tokens = encoding_gpt35.encode(text)
actual_token_strings = [encoding_gpt35.decode([token]) for token in actual_tokens]
print(f"\n實際 tiktoken 結果：")
print(f"  Token 切分：{actual_token_strings}")
print(f"  Token 數量：{len(actual_tokens)}")
print()

# === Step 2: 為什麼需要Token？ ===
print("=== 為什麼電腦需要Token？ ===")
print("❌ 電腦不懂：'我愛台灣'")
print("✅ 電腦懂：[1234, 5678, 9012, 3456]")
print()
print("Token就像是：")
print("• 文字的樂高積木")
print("• 每個積木有編號")
print("• 電腦用編號來理解")
print()

# === Step 3: 實際Token化範例 ===
print("=== 實際Token化範例 ===")

# 中文範例
text_zh = "今天天氣很好"
tokens_zh = ["今天", "天氣", "很", "好"]
token_ids_zh = [4170, 5929, 2523, 1962]  # 模擬的ID

print(f"中文：{text_zh}")
print(f"切分：{tokens_zh}")
print(f"編號：{token_ids_zh}")
print()

# 英文範例
text_en = "Hello World!"
tokens_en = ["Hello", " World", "!"]
token_ids_en = [15496, 4435, 0]  # 模擬的ID

print(f"英文：{text_en}")
print(f"切分：{tokens_en}")
print(f"編號：{token_ids_en}")
print()

# === Step 4: Token計費概念 ===
print("=== Token與API計費 ===")
print("OpenAI計費方式：")
print("• 1000 tokens ≈ 750個英文字")
print("• 1000 tokens ≈ 500個中文字")
print("• GPT-3.5: $0.002 / 1K tokens")
print("• GPT-4: $0.03 / 1K tokens")
print()

# 使用 tiktoken 計算實際的 token 數
message = "請幫我寫一篇文章"
actual_token_count = len(encoding_gpt35.encode(message))
cost_gpt35 = actual_token_count * 0.002 / 1000
cost_gpt4 = actual_token_count * 0.03 / 1000

print(f"訊息：'{message}'")
print(f"實際 Token 數（使用 tiktoken）：{actual_token_count}")
print(f"GPT-3.5費用：${cost_gpt35:.6f}")
print(f"GPT-4費用：${cost_gpt4:.6f}")
print()

# === Step 5: 比較不同模型的 Token 編碼 ===
print("=== 不同模型的 Token 編碼差異 ===")
sample_text = "人工智慧正在改變世界"

# GPT-3.5-turbo 編碼
tokens_gpt35 = encoding_gpt35.encode(sample_text)
print(f"文字：{sample_text}")
print(f"GPT-3.5-turbo Token 數：{len(tokens_gpt35)}")

# GPT-4 編碼
tokens_gpt4 = encoding_gpt4.encode(sample_text)
print(f"GPT-4 Token 數：{len(tokens_gpt4)}")

# 顯示差異
if len(tokens_gpt35) != len(tokens_gpt4):
    print("⚠️ 不同模型的 Token 計算可能不同！")
else:
    print("✅ 這兩個模型使用相同的 Token 編碼")
print()

# === Step 6: 實用函數：計算對話的 Token 數 ===
print("=== 實用函數：計算對話的 Token 數 ===")

def count_tokens(messages, model="gpt-3.5-turbo"):
    """計算對話列表的總 token 數"""
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0
    
    for message in messages:
        # 每個訊息有角色和內容
        role_tokens = len(encoding.encode(message["role"]))
        content_tokens = len(encoding.encode(message["content"]))
        total_tokens += role_tokens + content_tokens + 3  # 加3是因為特殊標記
    
    total_tokens += 3  # 對話開始和結束的特殊標記
    return total_tokens

# 測試對話
conversation = [
    {"role": "system", "content": "你是一個幫助使用者的助手"},
    {"role": "user", "content": "什麼是機器學習？"},
    {"role": "assistant", "content": "機器學習是人工智慧的一個分支，讓電腦能從數據中學習。"}
]

total_tokens = count_tokens(conversation)
estimated_cost = total_tokens * 0.002 / 1000

print("對話內容：")
for msg in conversation:
    print(f"  [{msg['role']}]: {msg['content']}")
print(f"\n總 Token 數：{total_tokens}")
print(f"預估費用（GPT-3.5）：${estimated_cost:.6f}")