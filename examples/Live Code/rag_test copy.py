import re
from typing import Any, Dict, List, Tuple
import pickle

import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


class Document:
    """文件類別，用於儲存文字內容和相關資訊"""

    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}


class PDFProcessor:
    """處理 PDF 檔案的類別"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化設定

        參數解釋：
        - chunk_size: 每個文字塊的大小（字數）
          想像成：每張筆記卡片可以寫 500 個字

        - chunk_overlap: 相鄰塊的重疊字數
          想像成：為了保持連貫，下一張卡片會重複前一張的最後 50 個字
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, pdf_path: str) -> str:
        """
        讀取 PDF 檔案

        流程：
        1. 開啟 PDF 檔案
        2. 逐頁讀取文字
        3. 合併所有頁面的文字
        """
        text = ""
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)

            # 逐頁處理
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # 加入頁碼標記，方便追蹤來源
                    text += f"\n[Page {page_num + 1}]\n{page_text}"

        return text

    def chunk_text(self, text: str, source: str) -> List[Document]:
        """
        將長文字切成小塊

        步驟詳解：
        1. 清理文字（移除多餘空白）
        2. 按照字數切割
        3. 保留重疊部分
        4. 記錄每塊的來源資訊
        """
        # 步驟1：清理文字
        text = re.sub(r"\s+", " ", text)  # 多個空白變一個
        text = text.strip()  # 移除頭尾空白

        # 步驟2：分割成字詞
        words = text.split()

        chunks = []
        # 步驟3：建立文字塊
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            # 取出 chunk_size 個字
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            # 只保留有意義的塊（至少 50 個字元）
            if len(chunk_text) > 50:
                doc = Document(
                    content=chunk_text,
                    metadata={
                        "source": source,  # 來源檔案
                        "chunk_id": len(chunks),  # 第幾塊
                        "start_index": i,  # 在原文的位置
                    },
                )
                chunks.append(doc)

        return chunks


class EmbeddingModel:
    """將文字轉換成向量的類別"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        載入嵌入模型

        all-MiniLM-L6-v2 模型介紹：
        - sentence-transformers: 專門處理句子嵌入的框架
        - MiniLM: 輕量化的語言模型（Microsoft 開發）
        - L6: 6層 Transformer（較少層數，更快速）
        - v2: 第二版，改進的版本

        優點：
        - 檔案小（只有 22MB）
        - 速度快（比 BGE-large 快 5 倍）
        - 不需要 GPU，CPU 就能流暢運行
        - 準確度仍然很好

        輸出維度：384 維（384個數字表示一段文字）
        """
        print(f"載入嵌入模型: {model_name}")
        # 強制使用 CPU，避免 CUDA 相容性問題
        self.model = SentenceTransformer(model_name, device='cpu')
        self.dimension = 384  # all-MiniLM-L6-v2 的向量維度

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        將文字列表轉換成向量

        參數說明：
        - texts: 要轉換的文字列表
        - batch_size: 批次處理大小（一次處理幾個）

        處理流程：
        1. 將文字分批（避免記憶體不足）
        2. 每批轉換成向量
        3. 正規化向量（讓長度為1，方便計算相似度）
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,  # 顯示進度條
            convert_to_numpy=True,  # 轉成 NumPy 陣列
            normalize_embeddings=True,  # 正規化（重要！）
        )
        return embeddings


from pathlib import Path


class FAISSVectorStore:
    """FAISS 向量資料庫類別"""

    def __init__(self, dimension: int = 384):
        """
        初始化 FAISS 索引

        參數：
        - dimension: 向量維度（預設為 all-MiniLM-L6-v2 的 384 維）
        """
        self.dimension = dimension
        # 使用 L2 距離的索引（也可以改用內積：faiss.IndexFlatIP）
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []  # 儲存對應的文件
        self.id_to_doc = {}  # ID 對應到文件的映射

    def add(self, embeddings: np.ndarray, documents: List[Document]):
        """
        添加向量和對應的文件到索引

        參數：
        - embeddings: 向量陣列 (n_samples, dimension)
        - documents: 對應的文件列表
        """
        # 確保 embeddings 是 float32 類型（FAISS 要求）
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # 獲取當前索引大小
        start_id = self.index.ntotal

        # 添加向量到索引
        self.index.add(embeddings)

        # 儲存文件並建立 ID 映射
        for i, doc in enumerate(documents):
            doc_id = start_id + i
            self.documents.append(doc)
            self.id_to_doc[doc_id] = doc

        print(f"已添加 {len(documents)} 個向量，索引總數: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """
        搜尋最相似的向量

        參數：
        - query_embedding: 查詢向量
        - k: 返回的結果數量

        返回：
        - 文件和距離的列表
        """
        # 確保查詢向量是正確的形狀和類型
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # 搜尋
        distances, indices = self.index.search(query_embedding, k)

        # 整理結果
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(dist)))

        return results

    def save(self, index_path: str, metadata_path: str):
        """
        儲存 FAISS 索引和元資料

        參數：
        - index_path: FAISS 索引檔案路徑
        - metadata_path: 元資料檔案路徑
        """
        # 儲存 FAISS 索引
        faiss.write_index(self.index, index_path)
        print(f"FAISS 索引已儲存到: {index_path}")

        # 儲存文件元資料
        metadata = {
            "documents": self.documents,
            "id_to_doc": self.id_to_doc,
            "dimension": self.dimension,
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        print(f"元資料已儲存到: {metadata_path}")

    def load(self, index_path: str, metadata_path: str):
        """
        載入 FAISS 索引和元資料

        參數：
        - index_path: FAISS 索引檔案路徑
        - metadata_path: 元資料檔案路徑
        """
        # 載入 FAISS 索引
        self.index = faiss.read_index(index_path)
        print(f"FAISS 索引已載入，共 {self.index.ntotal} 個向量")

        # 載入文件元資料
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        self.documents = metadata["documents"]
        self.id_to_doc = metadata["id_to_doc"]
        self.dimension = metadata["dimension"]
        print(f"元資料已載入，共 {len(self.documents)} 個文件")


def process_all_pdfs(data_folder: str = "data"):
    """
    處理資料夾中的所有 PDF 檔案並生成 embeddings

    參數：
    - data_folder: PDF 檔案所在的資料夾

    返回：
    - all_results: 包含所有檔案的 chunks 和 embeddings 的字典
    """
    # 初始化
    pdf_processor = PDFProcessor()
    embedding_model = EmbeddingModel()
    all_results = {}

    # 獲取所有 PDF 檔案
    pdf_files = list(Path(data_folder).glob("*.pdf"))
    print(f"找到 {len(pdf_files)} 個 PDF 檔案")

    # 處理每個 PDF
    for pdf_path in pdf_files:
        print(f"\n處理檔案: {pdf_path.name}")

        try:
            # 1. 讀取 PDF
            text = pdf_processor.load_pdf(str(pdf_path))
            print(f"  - 已讀取文字，長度: {len(text)} 字元")

            # 2. 切割文字
            chunks = pdf_processor.chunk_text(text, str(pdf_path))
            print(f"  - 切割成 {len(chunks)} 個文字塊")

            # 3. 生成向量
            if chunks:
                texts = [chunk.content for chunk in chunks]
                embeddings = embedding_model.encode(texts)
                print(f"  - 已生成 {len(embeddings)} 個向量，維度: {embeddings.shape}")

                # 儲存結果
                all_results[pdf_path.name] = {
                    "chunks": chunks,
                    "embeddings": embeddings
                }

        except Exception as e:
            print(f"  - 錯誤: {e}")
            continue

    # 顯示統計資訊
    print(f"\n=== 處理完成 ===")
    print(f"總共處理: {len(pdf_files)} 個 PDF 檔案")

    total_chunks = sum(len(result["chunks"]) for result in all_results.values())
    print(f"總文件塊: {total_chunks} 個")

    return all_results


def build_faiss_index(results: Dict[str, Any], save_path: str = None):
    """
    從處理結果建立 FAISS 索引

    參數：
    - results: process_all_pdfs 的返回結果
    - save_path: 儲存路徑的前綴（可選）

    返回：
    - vector_store: FAISS 向量儲存物件
    """
    # 初始化 FAISS 向量儲存
    vector_store = FAISSVectorStore(dimension=384)  # all-MiniLM-L6-v2 的維度

    # 將所有結果添加到 FAISS
    for filename, data in results.items():
        print(f"\n添加 {filename} 到 FAISS 索引...")
        vector_store.add(data["embeddings"], data["chunks"])

    # 儲存索引（如果提供了路徑）
    if save_path:
        vector_store.save(f"{save_path}.index", f"{save_path}.metadata")

    return vector_store


def demo_search(vector_store: FAISSVectorStore, query: str, embedding_model: EmbeddingModel):
    """
    示範搜尋功能

    參數：
    - vector_store: FAISS 向量儲存
    - query: 查詢文字
    - embedding_model: 嵌入模型
    """
    print(f"\n查詢: {query}")

    # 生成查詢向量
    query_embedding = embedding_model.encode([query])

    # 搜尋
    results = vector_store.search(query_embedding[0], k=3)

    # 顯示結果
    print("\n搜尋結果:")
    for i, (doc, distance) in enumerate(results, 1):
        print(f"\n--- 結果 {i} (距離: {distance:.4f}) ---")
        print(f"來源: {doc.metadata['source']}")
        print(f"塊 ID: {doc.metadata['chunk_id']}")
        print(f"內容預覽: {doc.content[:200]}...")


if __name__ == "__main__":
    # 1. 處理所有 PDF 並生成 embeddings
    print("=== 步驟 1: 處理 PDF 檔案 ===")
    results = process_all_pdfs()

    # 顯示每個檔案的處理結果
    print("\n各檔案處理結果：")
    for filename, data in results.items():
        print(f"- {filename}: {len(data['chunks'])} 個文字塊, "
              f"embeddings 形狀: {data['embeddings'].shape}")

    # 2. 建立 FAISS 索引
    print("\n=== 步驟 2: 建立 FAISS 索引 ===")
    vector_store = build_faiss_index(results, save_path="faiss_index")

    # 3. 示範搜尋（可選）
    print("\n=== 步驟 3: 測試搜尋功能 ===")
    embedding_model = EmbeddingModel()
    demo_search(vector_store, "multi agent debate", embedding_model)
