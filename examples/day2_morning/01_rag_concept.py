# Day 2 上午：RAG系統概念
# 09:30-10:00 理解RAG架構

import numpy as np
from typing import List, Tuple

# === Step 1: 什麼是RAG？ ===
print("=== RAG (Retrieval Augmented Generation) ===")
print()
print("RAG = 檢索 + 生成")
print("就像是：")
print("1. 📚 先查資料（Retrieval）")
print("2. 🤖 再回答問題（Generation）")
print()

# === Step 2: 為什麼需要RAG？ ===
print("=== 為什麼需要RAG？ ===")
print()

# 沒有RAG的問題
print("❌ 沒有RAG的問題：")
print("• AI的知識有時間限制")
print("• 無法存取私有資料")
print("• 可能產生幻覺（胡說八道）")
print()

# RAG的優勢
print("✅ RAG的優勢：")
print("• 可以使用最新資料")
print("• 可以存取公司內部文件")
print("• 回答更準確（有憑有據）")
print()

# === Step 3: RAG流程示範 ===
class SimpleRAG:
    """簡單的RAG系統示範"""
    
    def __init__(self):
        # 模擬的知識庫
        self.knowledge_base = [
            {
                "id": 1,
                "content": "Python是一種高階程式語言，由Guido van Rossum在1991年創造",
                "keywords": ["Python", "程式語言", "Guido", "1991"]
            },
            {
                "id": 2,
                "content": "機器學習是AI的分支，讓電腦從資料中學習",
                "keywords": ["機器學習", "AI", "資料", "學習"]
            },
            {
                "id": 3,
                "content": "深度學習使用神經網路，可以處理圖像、語音等複雜資料",
                "keywords": ["深度學習", "神經網路", "圖像", "語音"]
            }
        ]
    
    def retrieve(self, query: str, top_k: int = 2) -> List[dict]:
        """
        步驟1：檢索相關文件
        """
        print(f"🔍 檢索：'{query}'")
        
        # 簡單的關鍵字匹配
        scores = []
        for doc in self.knowledge_base:
            score = 0
            for keyword in doc["keywords"]:
                if keyword.lower() in query.lower():
                    score += 1
            scores.append((score, doc))
        
        # 排序並返回最相關的文件
        scores.sort(key=lambda x: x[0], reverse=True)
        results = [doc for score, doc in scores[:top_k] if score > 0]
        
        print(f"   找到 {len(results)} 個相關文件")
        return results
    
    def generate(self, query: str, context: List[dict]) -> str:
        """
        步驟2：基於檢索結果生成回答
        """
        print("🤖 生成回答...")
        
        if not context:
            return "抱歉，我找不到相關資料來回答你的問題。"
        
        # 組合上下文
        context_text = "\n".join([doc["content"] for doc in context])
        
        # 模擬生成回答
        answer = f"根據我的知識庫：\n\n{context_text}\n\n"
        answer += f"因此，關於'{query}'的答案是：基於以上資料的相關內容。"
        
        return answer
    
    def answer_question(self, query: str) -> str:
        """
        完整的RAG流程
        """
        print("\n" + "="*50)
        print(f"問題：{query}")
        print("="*50)
        
        # Step 1: 檢索
        relevant_docs = self.retrieve(query)
        
        # Step 2: 生成
        answer = self.generate(query, relevant_docs)
        
        return answer

# === Step 4: 測試RAG系統 ===
print("=== 測試RAG系統 ===")

rag = SimpleRAG()

# 測試問題
questions = [
    "Python是什麼？",
    "什麼是機器學習？",
    "深度學習如何處理圖像？"
]

for q in questions:
    answer = rag.answer_question(q)
    print(f"\n答案：{answer}\n")

# === Step 5: 向量相似度概念 ===
print("=== 向量相似度（Embeddings）===")
print()

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """計算餘弦相似度"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a ** 2 for a in vec1) ** 0.5
    norm2 = sum(b ** 2 for b in vec2) ** 0.5
    return dot_product / (norm1 * norm2)

# 模擬文字的向量表示
text_vectors = {
    "貓": [1.0, 0.8, 0.2],
    "狗": [0.9, 0.7, 0.3],
    "汽車": [0.1, 0.2, 0.9],
    "飛機": [0.2, 0.1, 0.95]
}

print("文字向量相似度：")
print("（1.0 = 完全相同，0.0 = 完全不同）")
print()

# 計算相似度
pairs = [
    ("貓", "狗"),
    ("貓", "汽車"),
    ("汽車", "飛機")
]

for text1, text2 in pairs:
    sim = cosine_similarity(text_vectors[text1], text_vectors[text2])
    print(f"{text1} vs {text2}: {sim:.2f}")
    print("  " + "█" * int(sim * 20))
print()

# === Step 6: RAG架構圖解 ===
print("=== RAG完整架構 ===")
print()
print("1️⃣ 文件處理")
print("   PDF/Word/網頁 → 切分文字 → 生成向量")
print("   ↓")
print("2️⃣ 向量資料庫")
print("   儲存所有文件的向量表示")
print("   ↓")
print("3️⃣ 使用者提問")
print("   問題 → 向量化 → 搜尋相似文件")
print("   ↓")
print("4️⃣ 檢索")
print("   找出最相關的K個文件片段")
print("   ↓")
print("5️⃣ 生成")
print("   LLM + 檢索結果 → 最終答案")
print()

# === Step 7: 實務應用場景 ===
print("=== RAG實務應用 ===")
print()

applications = {
    "客服系統": "檢索產品手冊回答客戶問題",
    "知識管理": "搜尋公司文件回答員工詢問",
    "教育助手": "根據教材內容回答學生問題",
    "法律諮詢": "檢索法條提供法律建議",
    "醫療助手": "查詢醫學文獻協助診斷"
}

for app, desc in applications.items():
    print(f"• {app}：{desc}")

print()
print("💡 提示：RAG讓AI變得更實用、更可靠！")