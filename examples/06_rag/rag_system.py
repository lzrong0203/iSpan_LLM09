"""
RAG (Retrieval Augmented Generation) 系統實作
結合檢索和生成的完整 RAG 系統
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime


class RAGSystem:
    """完整的 RAG 系統"""
    
    def __init__(self, llm_model, embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        初始化 RAG 系統
        
        Args:
            llm_model: 語言模型
            embedding_model: 嵌入模型名稱
        """
        self.llm = llm_model
        
        # 延遲導入以避免依賴問題
        from embedding_generator import EmbeddingGenerator
        self.embedder = EmbeddingGenerator(embedding_model)
        
        self.documents = []
        self.doc_embeddings = None
        self.metadata = []
        
    def index_documents(self, documents: List[str], metadata: List[Dict] = None):
        """
        索引文檔並生成嵌入
        
        Args:
            documents: 文檔列表
            metadata: 文檔元資料列表
        """
        self.documents = documents
        self.metadata = metadata or [{}] * len(documents)
        
        print(f"開始索引 {len(documents)} 個文檔...")
        self.doc_embeddings = self.embedder.encode_texts(documents)
        print(f"已索引 {len(documents)} 個文檔片段")
        
    def retrieve_relevant_docs(self, 
                              query: str, 
                              top_k: int = 3,
                              threshold: float = 0.5) -> Tuple[List[str], List[float], List[Dict]]:
        """
        檢索最相關的文檔
        
        Args:
            query: 查詢文本
            top_k: 返回前k個文檔
            threshold: 相似度閾值
            
        Returns:
            (文檔列表, 分數列表, 元資料列表)
        """
        if self.doc_embeddings is None or len(self.documents) == 0:
            return [], [], []
        
        # 編碼查詢
        query_embedding = self.embedder.encode_texts([query])
        
        # 計算相似度
        similarities = self.embedder.compute_similarity(
            query_embedding,
            self.doc_embeddings
        )
        
        # 過濾低於閾值的結果
        valid_indices = np.where(similarities >= threshold)[0]
        
        if len(valid_indices) == 0:
            return [], [], []
        
        # 獲取top-k最相關的文檔
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])][-top_k:][::-1]
        
        relevant_docs = [self.documents[i] for i in sorted_indices]
        scores = [float(similarities[i]) for i in sorted_indices]
        metadata = [self.metadata[i] for i in sorted_indices]
        
        return relevant_docs, scores, metadata
    
    def generate_answer(self, 
                       query: str, 
                       context_docs: List[str],
                       metadata: List[Dict] = None) -> str:
        """
        基於檢索的文檔生成答案
        
        Args:
            query: 用戶問題
            context_docs: 相關文檔
            metadata: 文檔元資料
            
        Returns:
            生成的答案
        """
        if not context_docs:
            return "抱歉，我無法找到相關資訊來回答您的問題。"
        
        # 構建增強的prompt
        context = "\n\n".join([f"文檔 {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])
        
        # 如果有元資料，添加來源資訊
        if metadata:
            sources = "\n".join([f"- 來源 {i+1}: {m.get('source', 'unknown')}" 
                                for i, m in enumerate(metadata)])
        else:
            sources = ""
        
        prompt = f"""基於以下相關文檔回答問題。請確保答案準確、相關且有幫助。

相關文檔：
{context}

{sources}

問題：{query}

請根據提供的文檔內容回答。如果文檔中沒有足夠的資訊，請明確說明。

回答："""
        
        # 使用LLM生成回答（這裡是模擬）
        # response = self.llm.generate(prompt)
        response = self._simulate_llm_response(query, context_docs)
        
        return response
    
    def _simulate_llm_response(self, query: str, context_docs: List[str]) -> str:
        """
        模擬LLM回應（實際使用時替換為真實的LLM調用）
        
        Args:
            query: 查詢
            context_docs: 上下文文檔
            
        Returns:
            模擬的回應
        """
        # 簡單的模擬邏輯
        if "什麼是" in query:
            return f"根據提供的文檔，{query.replace('什麼是', '')}是一個重要的概念。{context_docs[0][:100]}..."
        elif "如何" in query:
            return f"關於{query}，文檔中提到：{context_docs[0][:150]}..."
        else:
            return f"基於相關文檔，我的理解是：{context_docs[0][:100]}..."
    
    def query(self, 
             question: str,
             top_k: int = 3,
             threshold: float = 0.5,
             return_sources: bool = True) -> Dict:
        """
        完整的RAG查詢流程
        
        Args:
            question: 用戶問題
            top_k: 檢索文檔數
            threshold: 相似度閾值
            return_sources: 是否返回來源
            
        Returns:
            包含答案和相關資訊的字典
        """
        start_time = datetime.now()
        
        # 1. 檢索相關文檔
        relevant_docs, scores, metadata = self.retrieve_relevant_docs(
            question, top_k, threshold
        )
        
        # 2. 生成答案
        answer = self.generate_answer(question, relevant_docs, metadata)
        
        # 3. 構建回應
        response = {
            "question": question,
            "answer": answer,
            "confidence": float(np.mean(scores)) if scores else 0.0,
            "processing_time": (datetime.now() - start_time).total_seconds()
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "content": doc[:200] + "..." if len(doc) > 200 else doc,
                    "score": score,
                    "metadata": meta
                }
                for doc, score, meta in zip(relevant_docs, scores, metadata)
            ]
        
        return response
    
    def add_documents(self, new_documents: List[str], new_metadata: List[Dict] = None):
        """
        添加新文檔到索引
        
        Args:
            new_documents: 新文檔列表
            new_metadata: 新文檔元資料
        """
        if not new_documents:
            return
        
        # 生成新文檔的嵌入
        new_embeddings = self.embedder.encode_texts(new_documents)
        
        # 添加到現有索引
        self.documents.extend(new_documents)
        self.metadata.extend(new_metadata or [{}] * len(new_documents))
        
        if self.doc_embeddings is None:
            self.doc_embeddings = new_embeddings
        else:
            self.doc_embeddings = np.vstack([self.doc_embeddings, new_embeddings])
        
        print(f"已添加 {len(new_documents)} 個新文檔")
    
    def save_index(self, filepath: str):
        """
        保存索引到檔案
        
        Args:
            filepath: 保存路徑
        """
        index_data = {
            "documents": self.documents,
            "metadata": self.metadata,
            "embeddings_shape": self.doc_embeddings.shape if self.doc_embeddings is not None else None
        }
        
        # 保存文檔和元資料
        with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        # 保存嵌入向量
        if self.doc_embeddings is not None:
            np.save(f"{filepath}.npy", self.doc_embeddings)
        
        print(f"索引已保存到：{filepath}")
    
    def load_index(self, filepath: str):
        """
        從檔案載入索引
        
        Args:
            filepath: 檔案路徑
        """
        # 載入文檔和元資料
        with open(f"{filepath}.json", 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        self.documents = index_data["documents"]
        self.metadata = index_data["metadata"]
        
        # 載入嵌入向量
        if index_data["embeddings_shape"]:
            self.doc_embeddings = np.load(f"{filepath}.npy")
        
        print(f"已載入索引：{len(self.documents)} 個文檔")


class HybridRAG(RAGSystem):
    """混合檢索的 RAG 系統"""
    
    def __init__(self, llm_model, embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__(llm_model, embedding_model)
        self.keyword_index = {}  # 關鍵詞索引
    
    def build_keyword_index(self):
        """建立關鍵詞索引（簡單的 BM25 替代）"""
        from collections import defaultdict
        import math
        
        self.keyword_index = defaultdict(list)
        
        # 簡單的詞頻統計
        for i, doc in enumerate(self.documents):
            words = doc.lower().split()
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # 添加到倒排索引
            for word, freq in word_freq.items():
                self.keyword_index[word].append((i, freq))
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        關鍵詞搜索
        
        Args:
            query: 查詢文本
            top_k: 返回前k個結果
            
        Returns:
            (文檔索引, 分數) 列表
        """
        query_words = query.lower().split()
        doc_scores = {}
        
        for word in query_words:
            if word in self.keyword_index:
                for doc_idx, freq in self.keyword_index[word]:
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + freq
        
        # 排序並返回top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 3, alpha: float = 0.5) -> Tuple[List[str], List[float]]:
        """
        混合檢索（結合向量和關鍵詞檢索）
        
        Args:
            query: 查詢文本
            top_k: 返回前k個結果
            alpha: 向量檢索權重（1-alpha 為關鍵詞檢索權重）
            
        Returns:
            (文檔列表, 分數列表)
        """
        # 向量檢索
        dense_docs, dense_scores, _ = self.retrieve_relevant_docs(query, top_k * 2)
        
        # 關鍵詞檢索
        if not self.keyword_index:
            self.build_keyword_index()
        sparse_results = self.keyword_search(query, top_k * 2)
        
        # 合併結果
        combined_scores = {}
        
        # 添加向量檢索結果
        for doc, score in zip(dense_docs, dense_scores):
            doc_idx = self.documents.index(doc)
            combined_scores[doc_idx] = alpha * score
        
        # 添加關鍵詞檢索結果
        max_keyword_score = max([s for _, s in sparse_results]) if sparse_results else 1
        for doc_idx, score in sparse_results:
            normalized_score = score / max_keyword_score
            if doc_idx in combined_scores:
                combined_scores[doc_idx] += (1 - alpha) * normalized_score
            else:
                combined_scores[doc_idx] = (1 - alpha) * normalized_score
        
        # 排序並返回top-k
        sorted_indices = sorted(combined_scores.keys(), 
                              key=lambda x: combined_scores[x], 
                              reverse=True)[:top_k]
        
        final_docs = [self.documents[i] for i in sorted_indices]
        final_scores = [combined_scores[i] for i in sorted_indices]
        
        return final_docs, final_scores


def expand_query(original_query: str, llm=None) -> List[str]:
    """
    使用LLM擴展查詢
    
    Args:
        original_query: 原始查詢
        llm: 語言模型（可選）
        
    Returns:
        擴展的查詢列表
    """
    # 這裡是簡單的模擬，實際使用時調用LLM
    expanded = [original_query]
    
    # 簡單的同義詞擴展
    if "AI" in original_query:
        expanded.append(original_query.replace("AI", "人工智慧"))
    if "機器學習" in original_query:
        expanded.append(original_query.replace("機器學習", "ML"))
    if "深度學習" in original_query:
        expanded.append(original_query.replace("深度學習", "神經網絡"))
    
    return expanded


def main():
    """主函數 - 使用範例"""
    
    print("=== RAG 系統範例 ===\n")
    
    # 準備範例文檔
    documents = [
        "人工智慧（AI）是電腦科學的一個分支，致力於創建能夠執行通常需要人類智慧的任務的系統。",
        "機器學習是人工智慧的子領域，使電腦能夠從資料中學習而無需明確編程。",
        "深度學習是機器學習的一種，使用多層神經網絡來學習資料的複雜模式。",
        "自然語言處理（NLP）是AI的一個分支，專注於使電腦能夠理解、解釋和生成人類語言。",
        "電腦視覺使機器能夠從圖像和影片中獲取資訊並理解視覺世界。",
        "強化學習是一種機器學習方法，代理通過與環境互動來學習如何做出決策。"
    ]
    
    metadata = [
        {"source": "AI基礎教程", "chapter": 1},
        {"source": "ML入門", "chapter": 2},
        {"source": "深度學習指南", "chapter": 3},
        {"source": "NLP手冊", "chapter": 1},
        {"source": "電腦視覺教程", "chapter": 1},
        {"source": "RL理論", "chapter": 1}
    ]
    
    # 初始化RAG系統（使用模擬的LLM）
    rag = RAGSystem(llm_model=None)
    
    # 索引文檔
    rag.index_documents(documents, metadata)
    
    # 測試查詢
    questions = [
        "什麼是機器學習？",
        "深度學習和神經網絡的關係是什麼？",
        "AI可以處理圖像嗎？"
    ]
    
    for question in questions:
        print(f"\n問題：{question}")
        result = rag.query(question, top_k=2)
        
        print(f"答案：{result['answer']}")
        print(f"信心度：{result['confidence']:.2f}")
        
        if result.get('sources'):
            print("參考來源：")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. [{source['score']:.2f}] {source['metadata'].get('source', 'unknown')}")
    
    # 測試混合檢索
    print("\n=== 混合檢索範例 ===\n")
    
    hybrid_rag = HybridRAG(llm_model=None)
    hybrid_rag.index_documents(documents, metadata)
    hybrid_rag.build_keyword_index()
    
    query = "深度學習和AI"
    docs, scores = hybrid_rag.hybrid_search(query, top_k=3, alpha=0.6)
    
    print(f"混合檢索結果（查詢：{query}）：")
    for doc, score in zip(docs, scores):
        print(f"  [{score:.2f}] {doc[:50]}...")
    
    # 測試查詢擴展
    print("\n=== 查詢擴展範例 ===\n")
    
    original = "AI和機器學習的應用"
    expanded = expand_query(original)
    
    print(f"原始查詢：{original}")
    print(f"擴展查詢：")
    for q in expanded:
        print(f"  - {q}")


if __name__ == "__main__":
    main()