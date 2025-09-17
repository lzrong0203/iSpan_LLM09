# Day 2 上午：Embeddings向量化
# 10:00-10:30 文字向量化實作

import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# 載入環境變數
load_dotenv()

# === Step 1: 初始化 ===
print("=== Embeddings 文字向量化 ===")
print()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Step 2: 什麼是Embeddings？ ===
print("=== 什麼是Embeddings？ ===")
print()
print("Embeddings = 文字的數學表示")
print("例如：")
print("  '貓' → [0.1, 0.8, 0.2, ...]")
print("  '狗' → [0.2, 0.7, 0.3, ...]")
print()
print("相似的詞，向量也相似！")
print()


# === Step 3: 使用OpenAI生成Embeddings ===
def get_embedding(text: str, model="text-embedding-ada-003") -> List[float]:
    """
    使用OpenAI API生成文字向量
    """
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"錯誤：{e}")
        # 返回隨機向量作為備案
        return np.random.random(1536).tolist()


# === Step 4: 測試Embedding生成 ===
print("=== 生成Embeddings ===")
print()

# 測試文字
test_texts = ["蘋果", "香蕉", "橘子", "手機"]

embeddings = {}
for text in test_texts:
    print(f"生成 '{text}' 的向量...")
    embedding = get_embedding(text)
    embeddings[text] = embedding
    print(f"  向量維度：{len(embedding)}")
    print(f"  前5個值：{embedding[:5]}")
    print()


# === Step 5: 計算相似度 ===
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """計算兩個向量的餘弦相似度"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return dot_product / (norm1 * norm2)


print("=== 計算文字相似度 ===")
print()

# 計算所有配對的相似度
for i, text1 in enumerate(test_texts):
    for text2 in test_texts[i + 1 :]:
        sim = cosine_similarity(embeddings[text1], embeddings[text2])
        print(f"'{text1}' vs '{text2}'")
        print(f"  相似度：{sim:.3f}")
        print(f"  視覺化：{'█' * int(sim * 20)}")
        print()


# === Step 6: 語義搜尋示範 ===
class SemanticSearch:
    """語義搜尋系統"""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_document(self, text: str):
        """添加文件到搜尋庫"""
        print(f"添加文件：{text[:50]}...")
        embedding = get_embedding(text)
        self.documents.append(text)
        self.embeddings.append(embedding)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """語義搜尋"""
        print(f"\n搜尋：'{query}'")

        # 生成查詢向量
        query_embedding = get_embedding(query)

        # 計算相似度
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((self.documents[i], sim))

        # 排序並返回最相似的結果
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


# === Step 7: 建立知識庫 ===
print("=== 建立語義搜尋系統 ===")
print()

search_engine = SemanticSearch()

# 添加文件
documents = [
    "Python是一種易學易用的程式語言",
    "機器學習可以從資料中自動學習模式",
    "深度學習是機器學習的一個分支",
    "今天的天氣非常晴朗",
    "ChatGPT是OpenAI開發的對話AI",
    "向量資料庫用於儲存和搜尋embeddings",
]

for doc in documents:
    search_engine.add_document(doc)

print()

# === Step 8: 測試語義搜尋 ===
print("=== 測試語義搜尋 ===")

queries = ["什麼是AI技術？", "程式設計語言", "天氣如何？"]

for query in queries:
    results = search_engine.search(query, top_k=2)
    print(f"\n查詢：{query}")
    print("結果：")
    for doc, score in results:
        print(f"  [{score:.3f}] {doc}")

# === Step 9: 實務應用提示 ===
print("\n" + "=" * 50)
print("=== Embeddings實務應用 ===")
print()

applications = [
    "🔍 語義搜尋：找出意思相近的文件",
    "❓ 問答系統：找出最相關的答案",
    "🏷️ 文件分類：根據內容自動分類",
    "🔗 推薦系統：推薦相似內容",
    "🌐 多語言搜尋：跨語言找相似內容",
]

for app in applications:
    print(f"• {app}")

print()
print("💡 提示：Embeddings是RAG系統的核心技術！")
