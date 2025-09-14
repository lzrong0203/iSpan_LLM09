import re
from typing import Any, Dict, List, Tuple, Optional
import pickle
import os

import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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


class RAGRetriever:
    """RAG 檢索器類別，整合檢索和生成功能"""

    def __init__(self, vector_store: FAISSVectorStore, embedding_model: EmbeddingModel):
        """
        初始化 RAG 檢索器

        參數：
        - vector_store: FAISS 向量儲存
        - embedding_model: 嵌入模型
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query: str, k: int = 5, rerank: bool = True) -> List[Document]:
        """
        檢索相關文件

        參數：
        - query: 查詢文字
        - k: 要檢索的文件數量
        - rerank: 是否進行重新排序

        返回：
        - 相關文件列表
        """
        # 生成查詢向量
        print(f"正在為查詢生成向量: '{query}'")
        query_embedding = self.embedding_model.encode([query])[0]

        # 從向量資料庫搜尋
        results = self.vector_store.search(query_embedding, k=k*2 if rerank else k)

        # 重新排序（可選）
        if rerank:
            results = self._rerank_results(query, results, top_k=k)
        else:
            results = results[:k]

        # 只返回文件
        documents = [doc for doc, _ in results]
        return documents

    def _rerank_results(self, query: str, results: List[Tuple[Document, float]],
                        top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        重新排序檢索結果

        使用更精細的相似度計算進行重新排序
        """
        # 這裡可以實作更複雜的重新排序邏輯
        # 例如：考慮關鍵字匹配、文件長度等

        reranked = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for doc, distance in results:
            # 計算額外的相關性分數
            content_lower = doc.content.lower()
            content_words = set(content_lower.split())

            # 關鍵字重疊分數
            word_overlap = len(query_words & content_words) / len(query_words) if query_words else 0

            # 精確匹配加分
            exact_match_bonus = 1.0 if query_lower in content_lower else 0.0

            # 綜合分數（距離越小越好，所以用負號）
            combined_score = -distance + word_overlap * 0.5 + exact_match_bonus * 0.3

            reranked.append((doc, distance, combined_score))

        # 按綜合分數排序
        reranked.sort(key=lambda x: x[2], reverse=True)

        # 返回前 k 個結果
        return [(doc, dist) for doc, dist, _ in reranked[:top_k]]

    def retrieve_with_context(self, query: str, k: int = 5,
                            context_window: int = 1) -> List[Document]:
        """
        檢索文件並包含上下文

        參數：
        - query: 查詢文字
        - k: 要檢索的文件數量
        - context_window: 上下文窗口大小（前後各取幾個塊）

        返回：
        - 包含上下文的文件列表
        """
        # 先進行基本檢索
        retrieved_docs = self.retrieve(query, k=k)

        # 收集需要的塊 ID
        expanded_docs = []
        seen_ids = set()

        for doc in retrieved_docs:
            chunk_id = doc.metadata.get('chunk_id', 0)
            source = doc.metadata.get('source', '')

            # 添加上下文塊
            for offset in range(-context_window, context_window + 1):
                target_id = chunk_id + offset

                # 避免重複
                doc_key = (source, target_id)
                if doc_key in seen_ids or target_id < 0:
                    continue

                # 尋找對應的文件
                for candidate in self.vector_store.documents:
                    if (candidate.metadata.get('source') == source and
                        candidate.metadata.get('chunk_id') == target_id):
                        expanded_docs.append(candidate)
                        seen_ids.add(doc_key)
                        break

        # 按來源和塊 ID 排序
        expanded_docs.sort(key=lambda x: (x.metadata.get('source', ''),
                                         x.metadata.get('chunk_id', 0)))

        return expanded_docs


class RAGPipeline:
    """完整的 RAG 管道"""

    def __init__(self, retriever: RAGRetriever, openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-4o"):
        """
        初始化 RAG 管道

        參數：
        - retriever: RAG 檢索器
        - openai_api_key: OpenAI API 金鑰（可選，也可從環境變數讀取）
        - model_name: 使用的模型名稱（預設為 gpt-4o）
        """
        self.retriever = retriever
        self.model_name = model_name

        # 初始化 OpenAI 客戶端
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            self.use_llm = True
            print(f"✅ OpenAI API 已初始化，使用模型: {model_name}")
        else:
            self.openai_client = None
            self.use_llm = False
            print("⚠️ 未提供 OpenAI API 金鑰，將使用簡化版答案生成")

    def format_context(self, documents: List[Document]) -> str:
        """
        格式化檢索到的文件作為上下文

        參數：
        - documents: 文件列表

        返回：
        - 格式化的上下文字串
        """
        context_parts = []
        current_source = None

        for doc in documents:
            source = Path(doc.metadata.get('source', 'Unknown')).name
            chunk_id = doc.metadata.get('chunk_id', 0)

            # 如果來源改變，添加分隔符
            if source != current_source:
                if current_source is not None:
                    context_parts.append("\n" + "="*50 + "\n")
                context_parts.append(f"📄 來源: {source}\n")
                current_source = source

            # 添加文件內容
            context_parts.append(f"[區塊 {chunk_id}]")
            context_parts.append(doc.content)
            context_parts.append("\n")

        return "\n".join(context_parts)

    def generate_answer(self, query: str, context: str, max_tokens: int = 1000,
                       temperature: float = 0.7) -> str:
        """
        基於上下文生成答案

        參數：
        - query: 使用者查詢
        - context: 檢索到的上下文
        - max_tokens: 最大生成 token 數
        - temperature: 生成溫度（0-2，越高越有創意）

        返回：
        - 生成的答案
        """
        if self.use_llm and self.openai_client:
            try:
                # 構建系統提示詞
                system_prompt = """你是一個專業的知識助手。請基於提供的上下文資料，準確且詳細地回答使用者的問題。

重要規則：
1. 只使用提供的上下文資料來回答問題
2. 如果上下文中沒有相關資訊，請明確說明
3. 回答要結構清晰、邏輯嚴謹
4. 使用繁體中文回答
5. 如果資訊來自多個來源，請整合並提供完整答案"""

                # 構建使用者提示詞
                user_prompt = f"""基於以下上下文資料，請回答問題。

【上下文資料】
{context}

【問題】
{query}

請提供詳細且準確的答案："""

                # 調用 OpenAI API
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    frequency_penalty=0.2,
                    presence_penalty=0.2
                )

                # 提取答案
                answer = response.choices[0].message.content.strip()

                # 添加模型資訊
                answer += f"\n\n---\n🤖 使用模型: {self.model_name}"
                if hasattr(response.usage, 'total_tokens'):
                    answer += f" | 使用 tokens: {response.usage.total_tokens}"

                return answer

            except Exception as e:
                print(f"❌ OpenAI API 調用失敗: {e}")
                return self._generate_fallback_answer(query, context)
        else:
            return self._generate_fallback_answer(query, context)

    def _generate_fallback_answer(self, query: str, context: str) -> str:
        """
        備用答案生成（當 OpenAI API 不可用時）

        參數：
        - query: 使用者查詢
        - context: 檢索到的上下文

        返回：
        - 簡化版答案
        """
        answer = f"""
基於檢索到的資料，關於您的問題「{query}」：

📚 相關內容摘要：
{context[:800]}...

⚠️ 注意：這是簡化版答案。要獲得更好的回答，請設定 OPENAI_API_KEY 環境變數或在初始化時提供 API 金鑰。
        """
        return answer

    def query(self, question: str, k: int = 5, use_context_window: bool = True,
             verbose: bool = True) -> Dict[str, Any]:
        """
        執行完整的 RAG 查詢

        參數：
        - question: 使用者問題
        - k: 檢索的文件數量
        - use_context_window: 是否使用上下文窗口
        - verbose: 是否顯示詳細資訊

        返回：
        - 包含答案和相關資訊的字典
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 處理查詢: {question}")
            print(f"{'='*60}")

        # 步驟 1: 檢索相關文件
        if verbose:
            print("\n📥 步驟 1: 檢索相關文件...")

        if use_context_window:
            documents = self.retriever.retrieve_with_context(question, k=k)
        else:
            documents = self.retriever.retrieve(question, k=k)

        if verbose:
            print(f"✅ 檢索到 {len(documents)} 個相關文件塊")

        # 步驟 2: 格式化上下文
        if verbose:
            print("\n📝 步驟 2: 格式化上下文...")

        context = self.format_context(documents)

        if verbose:
            print(f"✅ 上下文長度: {len(context)} 字元")

        # 步驟 3: 生成答案
        if verbose:
            print("\n🤖 步驟 3: 生成答案...")

        answer = self.generate_answer(question, context)

        if verbose:
            print("✅ 答案已生成")

        # 整理結果
        result = {
            "question": question,
            "answer": answer,
            "context": context,
            "documents": documents,
            "num_documents": len(documents),
            "sources": list(set(Path(doc.metadata.get('source', 'Unknown')).name
                              for doc in documents))
        }

        return result

    def batch_query(self, questions: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """
        批次處理多個查詢

        參數：
        - questions: 問題列表
        - k: 每個查詢檢索的文件數量

        返回：
        - 結果列表
        """
        results = []

        for i, question in enumerate(questions, 1):
            print(f"\n處理第 {i}/{len(questions)} 個問題...")
            result = self.query(question, k=k, verbose=False)
            results.append(result)

        return results


def create_rag_system(index_path: str = "faiss_index", openai_api_key: Optional[str] = None):
    """
    建立並載入 RAG 系統（用於已有索引的情況）

    參數：
    - index_path: FAISS 索引檔案路徑前綴
    - openai_api_key: OpenAI API 金鑰

    返回：
    - rag_pipeline: 配置好的 RAG 管道
    """
    # 載入向量儲存
    vector_store = FAISSVectorStore()
    vector_store.load(f"{index_path}.index", f"{index_path}.metadata")

    # 初始化嵌入模型
    embedding_model = EmbeddingModel()

    # 建立檢索器和管道
    retriever = RAGRetriever(vector_store, embedding_model)
    rag_pipeline = RAGPipeline(retriever, openai_api_key=openai_api_key)

    return rag_pipeline


def interactive_rag_query(rag_pipeline: RAGPipeline):
    """
    互動式 RAG 查詢介面

    參數：
    - rag_pipeline: RAG 管道
    """
    print("\n" + "="*80)
    print("💬 互動式 RAG 查詢系統")
    print("="*80)
    print("輸入您的問題（輸入 'quit' 或 'exit' 結束）")

    while True:
        print("\n" + "-"*60)
        query = input("❓ 請輸入問題: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 感謝使用，再見！")
            break

        if not query:
            print("⚠️ 請輸入有效的問題")
            continue

        # 執行查詢
        result = rag_pipeline.query(query, k=5, verbose=False)

        # 顯示答案
        print("\n📝 答案：")
        print(result['answer'])

        # 顯示來源
        print(f"\n📚 資料來源: {', '.join(result['sources'])}")
        print(f"📊 使用了 {result['num_documents']} 個文件塊")


def test_retrieval_system(vector_store: FAISSVectorStore, embedding_model: EmbeddingModel,
                         openai_api_key: Optional[str] = None):
    """
    測試完整的檢索系統

    參數：
    - vector_store: FAISS 向量儲存
    - embedding_model: 嵌入模型
    - openai_api_key: OpenAI API 金鑰（可選）
    """
    print("\n" + "="*80)
    print("🚀 測試 RAG 檢索系統")
    print("="*80)

    # 初始化檢索器和管道
    retriever = RAGRetriever(vector_store, embedding_model)
    rag_pipeline = RAGPipeline(retriever, openai_api_key=openai_api_key)

    # 測試查詢列表
    test_queries = [
        "multi agent debate",
        "system performance",
        "data processing",
    ]

    # 測試單一查詢（詳細模式）
    print("\n📊 單一查詢測試（詳細模式）")
    result = rag_pipeline.query(test_queries[0], k=3, verbose=True)

    print("\n" + "="*60)
    print("📋 查詢結果摘要：")
    print(f"問題: {result['question']}")
    print(f"檢索到的文件數: {result['num_documents']}")
    print(f"資料來源: {', '.join(result['sources'])}")
    print(f"\n答案預覽:")
    print(result['answer'][:500] + "...")

    # 測試批次查詢
    print("\n" + "="*60)
    print("📊 批次查詢測試")
    print("="*60)

    batch_results = rag_pipeline.batch_query(test_queries[1:], k=2)

    for i, result in enumerate(batch_results, 1):
        print(f"\n查詢 {i}: {result['question']}")
        print(f"  - 檢索文件數: {result['num_documents']}")
        print(f"  - 資料來源: {', '.join(result['sources'])}")

    print("\n✅ 檢索系統測試完成！")


def simple_rag_query(query: str, vector_store: FAISSVectorStore,
                     embedding_model: EmbeddingModel):
    """
    簡單的 RAG 查詢函數

    參數：
    - query: 使用者查詢
    - vector_store: FAISS 向量儲存
    - embedding_model: 嵌入模型
    - openai_api_key: OpenAI API 金鑰
    """
    print(f"\n🔍 查詢: {query}")
    print("="*60)

    # 1. 生成查詢向量
    query_embedding = embedding_model.encode([query])[0]

    # 2. 檢索相關文件
    results = vector_store.search(query_embedding, k=5)

    # 3. 組合上下文
    context = "\n\n".join([doc.content for doc, _ in results])

    print(f"✅ 找到 {len(results)} 個相關文件")

    # 4. 生成答案
    
    try:
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一個專業的知識助手。請基於提供的上下文資料，準確地回答問題。使用繁體中文回答。"},
                {"role": "user", "content": f"上下文：\n{context}\n\n問題：{query}\n\n請回答："}
            ],
            max_tokens=800,
            temperature=0
        )

        answer = response.choices[0].message.content
        print(f"\n📝 GPT-4o 答案：\n{answer}")
        print(f"📄 相關內容：\n{context[:500]}...")

    except Exception as e:
        print(f"❌ OpenAI API 錯誤: {e}")
        print(f"\n📄 相關內容：\n{context[:500]}...")
    


if __name__ == "__main__":
    # 檢查是否已有索引
    if Path("faiss_index.index").exists() and Path("faiss_index.metadata").exists():
        print("✅ 載入現有索引...")
        vector_store = FAISSVectorStore()
        vector_store.load("faiss_index.index", "faiss_index.metadata")
        embedding_model = EmbeddingModel()
    else:
        print("📚 建立新索引...")
        # 處理 PDF
        results = process_all_pdfs()
        # 建立索引
        vector_store = build_faiss_index(results, save_path="faiss_index")
        embedding_model = EmbeddingModel()

    # 執行查詢
    queries = [
        "multi agent debate is useful or not",
    ]

    for query in queries:
        simple_rag_query(query, vector_store, embedding_model)

    
