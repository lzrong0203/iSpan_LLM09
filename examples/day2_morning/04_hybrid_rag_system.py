#!/usr/bin/env python3
"""
混合檢索 RAG 系統：結合 FAISS 向量搜尋與 BM25 關鍵字搜尋
Hybrid RAG System: Combining FAISS vector search with BM25 keyword search
"""

import json
import os
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import math

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import jieba  # 中文分詞

# 載入環境變數
load_dotenv()

# === 資料結構定義 ===
@dataclass
class Document:
    """文檔資料結構"""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextProcessor:
    """文本處理器：支援中英文"""

    def __init__(self, language: str = "mixed"):
        """
        初始化文本處理器

        Args:
            language: "chinese", "english", or "mixed"
        """
        self.language = language

    def tokenize(self, text: str) -> List[str]:
        """
        文本分詞

        Args:
            text: 輸入文本

        Returns:
            分詞後的列表
        """
        if self.language == "chinese" or self.language == "mixed":
            # 使用 jieba 進行中文分詞
            tokens = list(jieba.cut(text))
            # 過濾停用詞和空白
            tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]
        else:
            # 英文分詞
            tokens = text.lower().split()
            # 簡單的停用詞過濾
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            tokens = [t for t in tokens if t not in stop_words]

        return tokens

    def split_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """
        將長文本切分成重疊的小塊

        Args:
            text: 輸入文本
            chunk_size: 每塊的字符數
            overlap: 重疊字符數

        Returns:
            文本塊列表
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]

            # 嘗試在句號處斷開
            if end < text_length:
                last_period = chunk.rfind('。')
                if last_period == -1:
                    last_period = chunk.rfind('.')
                if last_period != -1 and last_period > chunk_size * 0.5:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk.strip())
            start = end - overlap if end < text_length else text_length

        return chunks


class BM25Retriever:
    """BM25 關鍵字檢索器"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25 檢索器

        Args:
            k1: BM25 參數，控制詞頻飽和度 (通常 1.2-2.0)
            b: BM25 參數，控制文檔長度歸一化 (通常 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.text_processor = TextProcessor(language="mixed")

    def add_documents(self, documents: List[str]):
        """
        添加文檔到 BM25 索引

        Args:
            documents: 文檔列表
        """
        self.corpus = documents
        self.tokenized_corpus = [self.text_processor.tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        BM25 搜尋

        Args:
            query: 查詢文本
            top_k: 返回前 k 個結果

        Returns:
            [(文檔索引, BM25分數)] 列表
        """
        if self.bm25 is None:
            return []

        tokenized_query = self.text_processor.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 獲取前 k 個最高分的文檔
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

        return results

    def save(self, filepath: str):
        """儲存 BM25 索引"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'corpus': self.corpus,
                'tokenized_corpus': self.tokenized_corpus,
                'k1': self.k1,
                'b': self.b
            }, f)

    def load(self, filepath: str):
        """載入 BM25 索引"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.corpus = data['corpus']
        self.tokenized_corpus = data['tokenized_corpus']
        self.k1 = data['k1']
        self.b = data['b']
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)


class FAISSRetriever:
    """FAISS 向量檢索器"""

    def __init__(self, dimension: int = 1536, index_type: str = "flat"):
        """
        初始化 FAISS 檢索器

        Args:
            dimension: 向量維度
            index_type: 索引類型 ("flat", "ivf", "hnsw")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index()
        self.documents = []

    def _create_index(self) -> faiss.Index:
        """
        建立 FAISS 索引

        Returns:
            FAISS 索引對象
        """
        if self.index_type == "flat":
            # 精確搜尋，適合小規模資料
            index = faiss.IndexFlatIP(self.dimension)  # 內積（用於正規化向量）
        elif self.index_type == "ivf":
            # IVF 索引，適合中等規模
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "hnsw":
            # HNSW 索引，適合大規模且需要快速查詢
            index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def add_documents(self, embeddings: np.ndarray, documents: List[str]):
        """
        添加文檔向量到 FAISS 索引

        Args:
            embeddings: 文檔向量矩陣 (n_docs, dimension)
            documents: 對應的文檔列表
        """
        # 正規化向量（用於餘弦相似度）
        faiss.normalize_L2(embeddings)

        # 訓練索引（如果需要）
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)

        # 添加向量
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        向量搜尋

        Args:
            query_embedding: 查詢向量
            top_k: 返回前 k 個結果

        Returns:
            [(文檔索引, 相似度分數)] 列表
        """
        # 確保查詢向量是 2D 的
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # 正規化查詢向量
        faiss.normalize_L2(query_embedding)

        # 搜尋
        scores, indices = self.index.search(query_embedding, top_k)

        # 返回結果
        results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]
        return results

    def save(self, index_path: str, docs_path: str):
        """儲存 FAISS 索引和文檔"""
        faiss.write_index(self.index, index_path)
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str, docs_path: str):
        """載入 FAISS 索引和文檔"""
        self.index = faiss.read_index(index_path)
        with open(docs_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)


class HybridRAGSystem:
    """
    混合檢索 RAG 系統
    結合 FAISS 向量檢索和 BM25 關鍵字檢索
    """

    def __init__(self,
                 model: str = "gpt-3.5-turbo",
                 embedding_model: str = "text-embedding-ada-002",
                 vector_weight: float = 0.7,
                 keyword_weight: float = 0.3):
        """
        初始化混合 RAG 系統

        Args:
            model: OpenAI 生成模型
            embedding_model: OpenAI 嵌入模型
            vector_weight: 向量檢索權重
            keyword_weight: 關鍵字檢索權重
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.embedding_model = embedding_model

        # 檢索權重
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

        # 初始化檢索器
        self.faiss_retriever = FAISSRetriever(dimension=1536, index_type="flat")
        self.bm25_retriever = BM25Retriever()

        # 文本處理器
        self.text_processor = TextProcessor(language="mixed")

        # 文檔存儲
        self.documents = []
        self.doc_embeddings = []

    def add_knowledge(self, text: str, source: str = "unknown", chunk_size: int = 200):
        """
        添加知識到系統

        Args:
            text: 文本內容
            source: 來源標識
            chunk_size: 分塊大小
        """
        print(f"添加知識來源：{source}")

        # 文本分塊
        chunks = self.text_processor.split_text(text, chunk_size=chunk_size, overlap=50)
        print(f"  切分成 {len(chunks)} 個片段")

        # 生成嵌入向量
        embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                # 調用 OpenAI API 生成嵌入
                response = self.client.embeddings.create(
                    input=chunk,
                    model=self.embedding_model
                )
                embedding = np.array(response.data[0].embedding)
            except Exception as e:
                print(f"  生成嵌入失敗 (片段 {i}): {e}")
                # 使用隨機向量作為備案
                embedding = np.random.randn(1536)

            embeddings.append(embedding)

            # 創建文檔對象
            doc = Document(
                id=f"{source}_{i}",
                content=chunk,
                embedding=embedding,
                metadata={"source": source, "chunk_id": i}
            )
            self.documents.append(doc)

        # 更新檢索器
        embeddings_array = np.array(embeddings).astype('float32')
        self.faiss_retriever.add_documents(embeddings_array, chunks)
        self.bm25_retriever.add_documents(chunks)

        print(f"  成功添加 {len(chunks)} 個文檔片段")

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        混合檢索：結合向量和關鍵字搜尋

        Args:
            query: 查詢文本
            top_k: 返回前 k 個結果

        Returns:
            [(文檔內容, 綜合分數, 元數據)] 列表
        """
        # 1. 向量檢索
        try:
            response = self.client.embeddings.create(
                input=query,
                model=self.embedding_model
            )
            query_embedding = np.array(response.data[0].embedding).astype('float32')
        except Exception as e:
            print(f"生成查詢嵌入失敗: {e}")
            query_embedding = np.random.randn(1536).astype('float32')

        vector_results = self.faiss_retriever.search(query_embedding, top_k=top_k*2)

        # 2. BM25 關鍵字檢索
        keyword_results = self.bm25_retriever.search(query, top_k=top_k*2)

        # 3. 分數融合（Reciprocal Rank Fusion）
        doc_scores = {}

        # 處理向量檢索結果
        for rank, (idx, score) in enumerate(vector_results):
            if idx < len(self.documents):
                doc_id = f"doc_{idx}"
                # RRF 分數：1 / (k + rank)，k=60 是常用值
                rrf_score = 1 / (60 + rank + 1)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.vector_weight * rrf_score

        # 處理 BM25 檢索結果
        for rank, (idx, score) in enumerate(keyword_results):
            if idx < len(self.documents):
                doc_id = f"doc_{idx}"
                rrf_score = 1 / (60 + rank + 1)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.keyword_weight * rrf_score

        # 4. 排序並返回前 k 個結果
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_docs:
            idx = int(doc_id.split('_')[1])
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append((doc.content, score, doc.metadata))

        return results

    def rerank_with_cross_encoder(self, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        使用交叉編碼器重新排序（使用 GPT 進行相關性評分）

        Args:
            query: 查詢文本
            documents: 候選文檔列表
            top_k: 返回前 k 個結果

        Returns:
            重新排序後的 [(文檔, 分數)] 列表
        """
        reranked = []

        for doc in documents[:10]:  # 限制重排序的文檔數量以控制成本
            # 使用 GPT 評估相關性
            prompt = f"""請評估以下文檔與查詢的相關性，返回 0-10 的分數。
只返回數字，不要其他解釋。

查詢：{query}

文檔：{doc}

相關性分數（0-10）："""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "你是一個文檔相關性評分器。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=10
                )

                score_text = response.choices[0].message.content.strip()
                score = float(score_text) / 10.0  # 正規化到 0-1
            except:
                score = 0.5  # 預設分數

            reranked.append((doc, score))

        # 排序並返回前 k 個
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def answer_question(self,
                       question: str,
                       use_reranking: bool = False,
                       top_k: int = 5) -> Dict:
        """
        回答問題

        Args:
            question: 用戶問題
            use_reranking: 是否使用重新排序
            top_k: 檢索文檔數量

        Returns:
            包含答案和相關信息的字典
        """
        print(f"\n問題：{question}")

        # 1. 混合檢索
        search_results = self.hybrid_search(question, top_k=top_k)

        if not search_results:
            return {
                "answer": "抱歉，我找不到相關資料來回答你的問題。",
                "sources": [],
                "confidence": 0.0,
                "retrieval_method": "none"
            }

        # 2. 可選：重新排序
        if use_reranking:
            docs_to_rerank = [doc for doc, _, _ in search_results]
            reranked = self.rerank_with_cross_encoder(question, docs_to_rerank, top_k=3)

            # 使用重新排序的結果
            context_docs = [doc for doc, _ in reranked]
            sources = [search_results[i][2].get("source", "unknown") for i in range(len(reranked))]
            avg_score = np.mean([score for _, score in reranked])
        else:
            # 使用原始檢索結果
            context_docs = [doc for doc, _, _ in search_results[:3]]
            sources = [meta.get("source", "unknown") for _, _, meta in search_results[:3]]
            avg_score = np.mean([score for _, score, _ in search_results[:3]])

        # 3. 構建上下文
        context = "\n\n".join([f"資料{i+1}: {doc}" for i, doc in enumerate(context_docs)])

        # 4. 生成答案
        prompt = f"""基於以下資料回答問題。請確保答案準確且基於提供的資料。

相關資料：
{context}

問題：{question}

請提供清晰、準確的答案："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一個準確的問答助手，只根據提供的資料回答問題。如果資料不足以回答問題，請明確說明。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"生成答案時發生錯誤：{e}"

        return {
            "answer": answer,
            "sources": list(set(sources)),
            "confidence": float(avg_score),
            "retrieval_method": "hybrid_with_reranking" if use_reranking else "hybrid",
            "retrieved_docs": len(search_results)
        }

    def save_system(self, directory: str):
        """
        儲存整個系統

        Args:
            directory: 儲存目錄
        """
        os.makedirs(directory, exist_ok=True)

        # 儲存 FAISS 索引
        self.faiss_retriever.save(
            os.path.join(directory, "faiss.index"),
            os.path.join(directory, "faiss_docs.json")
        )

        # 儲存 BM25 索引
        self.bm25_retriever.save(os.path.join(directory, "bm25.pkl"))

        # 儲存文檔
        docs_data = []
        for doc in self.documents:
            docs_data.append({
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata
            })

        with open(os.path.join(directory, "documents.json"), 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

        # 儲存配置
        config = {
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "model": self.model,
            "embedding_model": self.embedding_model
        }

        with open(os.path.join(directory, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        print(f"系統已儲存到 {directory}")

    def load_system(self, directory: str):
        """
        載入整個系統

        Args:
            directory: 載入目錄
        """
        # 載入配置
        with open(os.path.join(directory, "config.json"), 'r') as f:
            config = json.load(f)

        self.vector_weight = config["vector_weight"]
        self.keyword_weight = config["keyword_weight"]
        self.model = config["model"]
        self.embedding_model = config["embedding_model"]

        # 載入 FAISS 索引
        self.faiss_retriever.load(
            os.path.join(directory, "faiss.index"),
            os.path.join(directory, "faiss_docs.json")
        )

        # 載入 BM25 索引
        self.bm25_retriever.load(os.path.join(directory, "bm25.pkl"))

        # 載入文檔
        with open(os.path.join(directory, "documents.json"), 'r', encoding='utf-8') as f:
            docs_data = json.load(f)

        self.documents = []
        for doc_data in docs_data:
            doc = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata=doc_data["metadata"]
            )
            self.documents.append(doc)

        print(f"從 {directory} 載入了 {len(self.documents)} 個文檔")


def main():
    """主函數：示範混合 RAG 系統的使用"""

    print("=== 混合檢索 RAG 系統 (FAISS + BM25) ===\n")

    # 初始化系統
    rag = HybridRAGSystem(
        model="gpt-3.5-turbo",
        vector_weight=0.6,  # 向量檢索權重
        keyword_weight=0.4   # 關鍵字檢索權重
    )

    # 添加知識庫
    knowledge_base = [
        {
            "text": """Python是一種高階程式語言，由Guido van Rossum在1991年創造。
            Python強調程式碼的可讀性，使用縮排來定義程式碼塊。
            Python支援多種程式設計範式，包括物件導向、程序式和函數式程式設計。
            Python擁有豐富的標準庫和第三方套件生態系統，如NumPy、Pandas、TensorFlow等。
            Python的應用領域包括網頁開發、資料科學、人工智慧、自動化腳本等。""",
            "source": "Python基礎知識"
        },
        {
            "text": """機器學習是人工智慧的一個分支，讓電腦能夠從資料中學習。
            監督式學習需要標記的訓練資料，包括分類和回歸任務。
            非監督式學習不需要標記資料，常用於聚類和降維。
            強化學習透過與環境互動來學習最佳策略。
            常見的機器學習演算法包括決策樹、隨機森林、支援向量機、神經網路等。
            機器學習的應用包括圖像識別、自然語言處理、推薦系統、異常檢測等。""",
            "source": "機器學習概論"
        },
        {
            "text": """深度學習使用多層神經網路來學習資料的複雜表示。
            卷積神經網路(CNN)擅長處理圖像資料，透過卷積層和池化層提取特徵。
            循環神經網路(RNN)適合處理序列資料如文字和時間序列。
            長短期記憶網路(LSTM)和門控循環單元(GRU)解決了RNN的梯度消失問題。
            Transformer架構革新了自然語言處理領域，引入了自注意力機制。
            BERT、GPT系列模型都是基於Transformer架構的預訓練語言模型。""",
            "source": "深度學習技術"
        },
        {
            "text": """向量資料庫是專門用於儲存和檢索高維向量的資料庫系統。
            FAISS是Facebook開發的高效相似度搜尋庫，支援十億級向量檢索。
            向量資料庫的應用包括語義搜尋、推薦系統、圖像檢索等。
            常見的向量資料庫包括Pinecone、Weaviate、Milvus、Chroma等。
            向量檢索通常使用餘弦相似度、歐氏距離或內積來衡量相似性。
            索引技術如LSH、HNSW、IVF可以加速大規模向量檢索。""",
            "source": "向量資料庫技術"
        },
        {
            "text": """BM25是一種經典的資訊檢索排序函數，基於概率檢索模型。
            BM25考慮了詞頻(TF)、逆文檔頻率(IDF)和文檔長度歸一化。
            相比TF-IDF，BM25引入了詞頻飽和度的概念，避免高頻詞過度影響。
            BM25的參數k1控制詞頻飽和度，b控制文檔長度歸一化程度。
            BM25在傳統搜尋引擎如Elasticsearch中廣泛使用。
            混合檢索結合了BM25的精確匹配和向量檢索的語義理解優勢。""",
            "source": "BM25檢索技術"
        }
    ]

    print("=== 建立知識庫 ===")
    for kb in knowledge_base:
        rag.add_knowledge(kb["text"], kb["source"])

    print("\n=== 測試問答（混合檢索） ===")

    # 測試問題
    test_questions = [
        "Python是誰創造的？它有哪些應用領域？",
        "什麼是監督式學習？請舉例說明。",
        "CNN和RNN分別適合處理什麼類型的資料？",
        "FAISS是什麼？它有什麼用途？",
        "BM25相比TF-IDF有什麼優勢？",
        "Transformer架構為什麼重要？",
        "向量資料庫有哪些常見的應用場景？"
    ]

    # 測試不同檢索模式
    for i, question in enumerate(test_questions[:3], 1):
        print(f"\n{'='*60}")
        print(f"測試 {i}: {question}")

        # 混合檢索（不使用重排序）
        result = rag.answer_question(question, use_reranking=False)
        print(f"\n[混合檢索]")
        print(f"答案：{result['answer']}")
        print(f"來源：{', '.join(result['sources'])}")
        print(f"信心度：{result['confidence']:.2%}")

        # 混合檢索（使用重排序）
        # result_reranked = rag.answer_question(question, use_reranking=True)
        # print(f"\n[混合檢索 + 重排序]")
        # print(f"答案：{result_reranked['answer']}")
        # print(f"信心度：{result_reranked['confidence']:.2%}")

    # 儲存系統
    print("\n=== 儲存系統 ===")
    rag.save_system("hybrid_rag_system")

    # 測試載入
    print("\n=== 測試載入系統 ===")
    new_rag = HybridRAGSystem()
    new_rag.load_system("hybrid_rag_system")

    # 使用載入的系統回答問題
    test_result = new_rag.answer_question("什麼是向量資料庫？")
    print(f"\n載入後測試問題：什麼是向量資料庫？")
    print(f"答案：{test_result['answer']}")

    # 互動模式
    print("\n" + "="*60)
    print("=== 互動問答模式 ===")
    print("輸入問題來測試混合RAG系統（輸入'quit'結束）")
    print("輸入'rerank'切換重排序模式")
    print()

    use_reranking = False

    while True:
        user_input = input("👤 你的問題：").strip()

        if user_input.lower() == 'quit':
            print("👋 再見！")
            break

        if user_input.lower() == 'rerank':
            use_reranking = not use_reranking
            print(f"📊 重排序模式：{'開啟' if use_reranking else '關閉'}")
            continue

        if user_input:
            result = rag.answer_question(user_input, use_reranking=use_reranking)
            print(f"🤖 答案：{result['answer']}")
            print(f"📚 資料來源：{', '.join(result['sources'])}")
            print(f"📊 信心度：{result['confidence']:.2%}")
            print(f"🔍 檢索方式：{result['retrieval_method']}")
            print()


if __name__ == "__main__":
    main()