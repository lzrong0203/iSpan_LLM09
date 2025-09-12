# Day 1 上午：Token是什麼？
# 09:30-10:00 Token基礎概念

# === Step 1: Token就像拼圖片 ===
text = "我愛台灣"
print(f"原始文字：{text}")
print()

# 模擬Token化過程
print("=== Token化過程 ===")
print("Step 1: 把文字切成小塊")
tokens = ["我", "愛", "台", "灣"]
for i, token in enumerate(tokens):
    print(f"  Token {i+1}: {token}")
print(f"\n總共：{len(tokens)} 個Token")
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

# 計算範例
message = "請幫我寫一篇文章"
token_count = 8  # 假設的token數
cost_gpt35 = token_count * 0.002 / 1000
cost_gpt4 = token_count * 0.03 / 1000

print(f"訊息：'{message}'")
print(f"Token數：{token_count}")
print(f"GPT-3.5費用：${cost_gpt35:.6f}")
print(f"GPT-4費用：${cost_gpt4:.6f}")