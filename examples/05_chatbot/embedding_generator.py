"""
向量嵌入生成器
用於將文本轉換為向量表示
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingGenerator:
    """向量嵌入生成器"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        初始化嵌入生成器
        
        Args:
            model_name: Sentence Transformer 模型名稱
        """
        print(f"載入嵌入模型：{model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"嵌入維度：{self.embedding_dim}")
    
    def encode_texts(self, 
                    texts: Union[str, List[str]], 
                    batch_size: int = 32,
                    show_progress: bool = True) -> np.ndarray:
        """
        將文本列表轉換為向量
        
        Args:
            texts: 文本或文本列表
            batch_size: 批次大小
            show_progress: 是否顯示進度條
            
        Returns:
            嵌入向量陣列
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        
        return embeddings
    
    def compute_similarity(self, 
                         query_embedding: np.ndarray, 
                         doc_embeddings: np.ndarray) -> np.ndarray:
        """
        計算查詢與文檔的相似度
        
        Args:
            query_embedding: 查詢向量
            doc_embeddings: 文檔向量陣列
            
        Returns:
            相似度分數陣列
        """
        # 確保輸入形狀正確
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # 計算餘弦相似度
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        return similarities
    
    def find_similar_texts(self, 
                          query: str, 
                          texts: List[str], 
                          top_k: int = 5) -> List[tuple]:
        """
        找出最相似的文本
        
        Args:
            query: 查詢文本
            texts: 文本列表
            top_k: 返回前k個結果
            
        Returns:
            (文本, 相似度分數) 的列表
        """
        # 編碼查詢
        query_embedding = self.encode_texts(query, show_progress=False)
        
        # 編碼文檔
        doc_embeddings = self.encode_texts(texts)
        
        # 計算相似度
        similarities = self.compute_similarity(query_embedding, doc_embeddings)
        
        # 獲取top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((texts[idx], float(similarities[idx])))
        
        return results
    
    def create_index(self, embeddings: np.ndarray) -> 'VectorIndex':
        """
        創建向量索引
        
        Args:
            embeddings: 向量陣列
            
        Returns:
            向量索引物件
        """
        return VectorIndex(embeddings)
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """
        保存嵌入向量
        
        Args:
            embeddings: 向量陣列
            filepath: 保存路徑
        """
        np.save(filepath, embeddings)
        print(f"嵌入向量已保存到：{filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """
        載入嵌入向量
        
        Args:
            filepath: 檔案路徑
            
        Returns:
            向量陣列
        """
        embeddings = np.load(filepath)
        print(f"已載入嵌入向量，形狀：{embeddings.shape}")
        return embeddings


class VectorIndex:
    """向量索引類"""
    
    def __init__(self, embeddings: np.ndarray):
        """
        初始化向量索引
        
        Args:
            embeddings: 向量陣列
        """
        self.embeddings = embeddings
        self.size = len(embeddings)
        
    def search(self, 
              query_embedding: np.ndarray, 
              top_k: int = 5) -> List[tuple]:
        """
        搜索最相似的向量
        
        Args:
            query_embedding: 查詢向量
            top_k: 返回前k個結果
            
        Returns:
            (索引, 相似度分數) 的列表
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # 計算相似度
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 獲取top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((int(idx), float(similarities[idx])))
        
        return results
    
    def add_embeddings(self, new_embeddings: np.ndarray):
        """
        添加新的嵌入向量
        
        Args:
            new_embeddings: 新的向量陣列
        """
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.size = len(self.embeddings)
    
    def remove_embedding(self, index: int):
        """
        移除指定索引的嵌入向量
        
        Args:
            index: 要移除的索引
        """
        if 0 <= index < self.size:
            self.embeddings = np.delete(self.embeddings, index, axis=0)
            self.size = len(self.embeddings)
        else:
            raise IndexError(f"索引 {index} 超出範圍")
    
    def get_statistics(self) -> dict:
        """
        獲取索引統計資訊
        
        Returns:
            統計資訊字典
        """
        return {
            "total_vectors": self.size,
            "dimension": self.embeddings.shape[1] if self.size > 0 else 0,
            "memory_size_mb": self.embeddings.nbytes / (1024 * 1024) if self.size > 0 else 0,
            "mean_norm": np.mean(np.linalg.norm(self.embeddings, axis=1)) if self.size > 0 else 0,
            "std_norm": np.std(np.linalg.norm(self.embeddings, axis=1)) if self.size > 0 else 0
        }


class MultilingualEmbeddingGenerator(EmbeddingGenerator):
    """多語言嵌入生成器"""
    
    def __init__(self):
        """初始化多語言嵌入生成器"""
        # 使用支援多語言的模型
        super().__init__('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.supported_languages = [
            'zh', 'en', 'ja', 'ko', 'es', 'fr', 'de', 'ru', 'ar', 'hi'
        ]
    
    def encode_multilingual(self, texts: List[str], languages: List[str] = None) -> np.ndarray:
        """
        編碼多語言文本
        
        Args:
            texts: 文本列表
            languages: 語言代碼列表（可選）
            
        Returns:
            嵌入向量陣列
        """
        # 如果提供了語言資訊，可以進行特殊處理
        # 這裡簡單地使用基礎編碼
        return self.encode_texts(texts)
    
    def cross_lingual_search(self, 
                            query: str, 
                            texts: List[str], 
                            query_lang: str = 'zh',
                            text_langs: List[str] = None,
                            top_k: int = 5) -> List[tuple]:
        """
        跨語言搜索
        
        Args:
            query: 查詢文本
            texts: 文本列表
            query_lang: 查詢語言
            text_langs: 文本語言列表
            top_k: 返回前k個結果
            
        Returns:
            搜索結果
        """
        print(f"執行跨語言搜索（查詢語言：{query_lang}）")
        return self.find_similar_texts(query, texts, top_k)


def main():
    """主函數 - 使用範例"""
    
    print("=== 向量嵌入生成器範例 ===\n")
    
    # 創建嵌入生成器
    embedder = EmbeddingGenerator()
    
    # 範例文本
    texts = [
        "人工智慧正在改變世界",
        "機器學習是人工智慧的一個分支",
        "深度學習使用神經網絡",
        "自然語言處理幫助電腦理解人類語言",
        "今天天氣真好"
    ]
    
    # 生成嵌入
    print("生成文本嵌入...")
    embeddings = embedder.encode_texts(texts, show_progress=False)
    print(f"嵌入形狀：{embeddings.shape}\n")
    
    # 查詢相似文本
    query = "AI和機器學習的關係"
    print(f"查詢：{query}")
    results = embedder.find_similar_texts(query, texts, top_k=3)
    
    print("最相似的文本：")
    for i, (text, score) in enumerate(results, 1):
        print(f"  {i}. {text} (相似度: {score:.4f})")
    
    # 創建向量索引
    print("\n=== 向量索引範例 ===\n")
    index = embedder.create_index(embeddings)
    stats = index.get_statistics()
    
    print("索引統計：")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # 搜索索引
    query_embedding = embedder.encode_texts(query, show_progress=False)
    search_results = index.search(query_embedding, top_k=3)
    
    print(f"\n索引搜索結果：")
    for idx, score in search_results:
        print(f"  文檔 {idx}: {texts[idx][:30]}... (相似度: {score:.4f})")
    
    # 多語言範例
    print("\n=== 多語言嵌入範例 ===\n")
    ml_embedder = MultilingualEmbeddingGenerator()
    
    multilingual_texts = [
        "Artificial Intelligence",  # 英文
        "人工知能",  # 日文
        "人工智慧",  # 中文
        "Intelligence Artificielle",  # 法文
        "Künstliche Intelligenz"  # 德文
    ]
    
    query_zh = "AI技術"
    print(f"跨語言查詢：{query_zh}")
    ml_results = ml_embedder.cross_lingual_search(
        query_zh, 
        multilingual_texts, 
        query_lang='zh',
        top_k=3
    )
    
    print("跨語言搜索結果：")
    for text, score in ml_results:
        print(f"  {text} (相似度: {score:.4f})")


if __name__ == "__main__":
    # 需要安裝：pip install sentence-transformers scikit-learn
    main()