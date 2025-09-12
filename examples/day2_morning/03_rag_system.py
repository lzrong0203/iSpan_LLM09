# Day 2 上午：完整RAG系統
# 10:30-11:30 實作完整的RAG系統

import os
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# 載入環境變數
load_dotenv()

# === Step 1: 初始化 ===
print("=== 完整RAG系統實作 ===")
print()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# === Step 2: 文件處理類 ===
class DocumentProcessor:
    """文件處理器"""
    
    @staticmethod
    def split_text(text: str, chunk_size: int = 200) -> List[str]:
        """
        將長文本切分成小塊
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

# === Step 3: 向量資料庫 ===
class VectorDatabase:
    """簡單的向量資料庫"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_document(self, text: str, embedding: List[float], metadata: Dict = None):
        """添加文件到資料庫"""
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """向量搜尋"""
        similarities = []
        
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((
                self.documents[i],
                sim,
                self.metadata[i]
            ))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """計算餘弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def save(self, filename: str):
        """儲存資料庫"""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "metadata": self.metadata
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"資料庫已儲存到 {filename}")
    
    def load(self, filename: str):
        """載入資料庫"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        print(f"從 {filename} 載入了 {len(self.documents)} 個文件")

# === Step 4: RAG系統主類 ===
class RAGSystem:
    """完整的RAG系統"""
    
    def __init__(self, model="gpt-3.5-turbo"):
        self.vector_db = VectorDatabase()
        self.model = model
        self.doc_processor = DocumentProcessor()
    
    def add_knowledge(self, text: str, source: str = "unknown"):
        """添加知識到系統"""
        print(f"添加知識來源：{source}")
        
        # 切分文本
        chunks = self.doc_processor.split_text(text)
        print(f"  切分成 {len(chunks)} 個片段")
        
        # 為每個片段生成向量並儲存
        for i, chunk in enumerate(chunks):
            try:
                # 生成向量
                response = client.embeddings.create(
                    input=chunk,
                    model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding
                
                # 添加到資料庫
                self.vector_db.add_document(
                    text=chunk,
                    embedding=embedding,
                    metadata={"source": source, "chunk_id": i}
                )
            except Exception as e:
                print(f"  錯誤處理片段 {i}: {e}")
                # 使用隨機向量作為備案
                embedding = np.random.random(1536).tolist()
                self.vector_db.add_document(
                    text=chunk,
                    embedding=embedding,
                    metadata={"source": source, "chunk_id": i}
                )
    
    def answer_question(self, question: str) -> Dict:
        """回答問題"""
        print(f"\n問題：{question}")
        
        # Step 1: 生成問題向量
        try:
            response = client.embeddings.create(
                input=question,
                model="text-embedding-ada-002"
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            print(f"生成問題向量失敗：{e}")
            query_embedding = np.random.random(1536).tolist()
        
        # Step 2: 檢索相關文件
        results = self.vector_db.search(query_embedding, top_k=3)
        
        if not results:
            return {
                "answer": "抱歉，我找不到相關資料來回答你的問題。",
                "sources": [],
                "confidence": 0.0
            }
        
        # Step 3: 準備上下文
        context_parts = []
        sources = []
        total_score = 0
        
        for doc, score, meta in results:
            context_parts.append(f"資料{len(context_parts)+1}: {doc}")
            sources.append(meta.get("source", "unknown"))
            total_score += score
        
        context = "\n\n".join(context_parts)
        avg_score = total_score / len(results)
        
        # Step 4: 生成答案
        prompt = f"""基於以下資料回答問題：

{context}

問題：{question}

請根據提供的資料準確回答。如果資料不足，請說明。"""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一個準確的問答助手，只根據提供的資料回答問題。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"生成答案時發生錯誤：{e}"
        
        return {
            "answer": answer,
            "sources": list(set(sources)),
            "confidence": avg_score,
            "retrieved_docs": len(results)
        }

# === Step 5: 建立知識庫 ===
print("=== 建立RAG系統 ===")
print()

rag = RAGSystem()

# 添加知識
knowledge_base = [
    {
        "text": """Python是一種高階程式語言，由Guido van Rossum在1991年創造。
        Python強調程式碼的可讀性，使用縮排來定義程式碼塊。
        Python支援多種程式設計範式，包括物件導向、程序式和函數式程式設計。
        Python擁有豐富的標準庫和第三方套件生態系統。""",
        "source": "Python基礎知識"
    },
    {
        "text": """機器學習是人工智慧的一個分支，讓電腦能夠從資料中學習。
        監督式學習需要標記的訓練資料，包括分類和回歸任務。
        非監督式學習不需要標記資料，常用於聚類和降維。
        強化學習透過與環境互動來學習最佳策略。""",
        "source": "機器學習概論"
    },
    {
        "text": """深度學習使用多層神經網路來學習資料的複雜表示。
        卷積神經網路(CNN)擅長處理圖像資料。
        循環神經網路(RNN)適合處理序列資料如文字和時間序列。
        Transformer架構革新了自然語言處理領域。""",
        "source": "深度學習技術"
    }
]

for kb in knowledge_base:
    rag.add_knowledge(kb["text"], kb["source"])

print()

# === Step 6: 測試問答 ===
print("=== 測試RAG問答系統 ===")

questions = [
    "Python是誰創造的？",
    "什麼是監督式學習？",
    "CNN適合處理什麼類型的資料？",
    "Transformer用於什麼領域？"
]

for q in questions:
    result = rag.answer_question(q)
    print()
    print("="*50)
    print(f"問題：{q}")
    print(f"答案：{result['answer']}")
    print(f"來源：{', '.join(result['sources'])}")
    print(f"信心度：{result['confidence']:.2%}")
    print(f"檢索文件數：{result['retrieved_docs']}")

# === Step 7: 儲存和載入 ===
print("\n" + "="*50)
print("=== 儲存RAG系統 ===")

# 儲存向量資料庫
rag.vector_db.save("rag_database.json")

# === Step 8: 互動模式 ===
print("\n" + "="*50)
print("=== 互動問答模式 ===")
print("輸入問題來測試RAG系統（輸入'quit'結束）")
print()

while True:
    user_question = input("👤 你的問題：")
    
    if user_question.lower() == 'quit':
        print("👋 再見！")
        break
    
    result = rag.answer_question(user_question)
    print(f"🤖 答案：{result['answer']}")
    print(f"📚 資料來源：{', '.join(result['sources'])}")
    print(f"📊 信心度：{result['confidence']:.2%}")
    print()