# Day 1 上午：Attention機制
# 10:00-10:30 理解注意力機制

import numpy as np

# === Step 1: 什麼是Attention？ ===
print("=== Attention機制是什麼？ ===")
print("像是閱讀理解時的「重點標記」")
print("AI決定要「注意」哪些字")
print()


# === Step 2: 簡單的Attention示範 ===
def simple_attention(query, keys, values):
    """
    超簡化的注意力機制
    query: 我現在想知道什麼
    keys: 所有可以看的內容
    values: 實際的資訊
    """
    # 計算相似度（這裡用簡單的點積）
    scores = []
    for key in keys:
        score = sum(q * k for q, k in zip(query, key))
        scores.append(score)

    # 轉換成機率（softmax的簡化版）
    total = sum(scores)
    weights = [s / total for s in scores]

    # 加權平均
    result = [0] * len(values[0])
    for weight, value in zip(weights, values):
        for i in range(len(value)):
            result[i] += weight * value[i]

    return weights, result


# === Step 3: 實際例子 ===
print("=== 實際例子：理解「我愛貓」===")
print()

# 假設每個字都有向量表示
words = ["我", "愛", "貓"]
word_vectors = [
    [1, 0, 0],  # 我
    [0, 1, 0],  # 愛
    [0, 0, 1],  # 貓
]

# 當AI看到「貓」時，要理解整句話
query = [0, 0, 1]  # 貓的向量
keys = word_vectors
values = word_vectors

weights, result = simple_attention(query, keys, values)

print("當AI看到「貓」這個字時：")
for word, weight in zip(words, weights):
    bar = "█" * int(weight * 20)
    print(f"  對「{word}」的注意力：{weight:.2f} {bar}")
print()

# === Step 4: Self-Attention ===
print("=== Self-Attention（自注意力）===")
print("每個字都看其他所有字")
print()

sentence = "今天天氣很好"
tokens = ["今天", "天氣", "很", "好"]

# 模擬attention矩陣
attention_matrix = [
    [0.7, 0.2, 0.05, 0.05],  # 今天 看其他字
    [0.3, 0.5, 0.1, 0.1],  # 天氣 看其他字
    [0.1, 0.3, 0.2, 0.4],  # 很 看其他字
    [0.05, 0.15, 0.3, 0.5],  # 好 看其他字
]

print(f"句子：{sentence}")
print("Attention矩陣（每個字看其他字的權重）：")
print("        ", "  ".join(tokens))
for i, token in enumerate(tokens):
    weights_str = " ".join(f"{w:.2f}" for w in attention_matrix[i])
    print(f"{token:4s}: {weights_str}")
print()

# === Step 5: 為什麼Attention很強大？ ===
print("=== Attention的優點 ===")
print("1. 長距離依賴：可以直接連接遠處的字")
print("   例：「小明」...（100個字）...「他」")
print()
print("2. 並行計算：所有字同時處理")
print("   不像RNN要一個一個處理")
print()
print("3. 可解釋性：能看出AI在注意什麼")
print("   對debug和理解AI很有幫助")

# === Step 6: 實作練習 ===
print("\n=== 實作練習：找出重要的字 ===")
text = "ChatGPT是OpenAI開發的語言模型"
important_words = ["ChatGPT", "OpenAI", "語言模型"]

print(f"句子：{text}")
print("AI認為重要的詞：")
for word in important_words:
    print(f"  • {word}")
