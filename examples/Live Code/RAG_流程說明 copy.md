# RAG (Retrieval-Augmented Generation) 系統完整流程說明

## 專案概覽
本專案實作了一個基於 FAISS 的 RAG 系統，能夠將 PDF 文件轉換為向量並建立可搜尋的向量資料庫。

## 系統架構

```
PDF 文件 → 文字擷取 → 文字分塊 → 向量化 → FAISS 索引 → 相似度搜尋
```

## 主要元件

### 1. 使用的模型與套件

- **Embedding 模型**: `sentence-transformers/all-MiniLM-L6-v2`
  - 輕量級模型（22MB）
  - 輸出維度：384 維
  - CPU 友好，不需要 GPU
  - 速度比大型模型快 5 倍

- **向量資料庫**: FAISS (Facebook AI Similarity Search)
  - 使用 `faiss-cpu` 版本
  - 支援大規模向量搜尋
  - 高效的相似度計算

- **PDF 處理**: PyPDF2
  - 用於讀取和解析 PDF 文件

### 2. 資料處理流程

#### 步驟 1: PDF 文件處理

```python
PDFProcessor 類別：
- load_pdf(): 讀取 PDF 檔案，逐頁擷取文字
- chunk_text(): 將長文字切割成小塊
  - chunk_size: 500 個字（預設）
  - chunk_overlap: 50 個字的重疊
```

#### 步驟 2: 文字向量化

```python
EmbeddingModel 類別：
- 載入 all-MiniLM-L6-v2 模型
- encode(): 將文字轉換為 384 維向量
- 批次處理支援（batch_size=32）
- 自動正規化向量
```

#### 步驟 3: FAISS 索引建立

```python
FAISSVectorStore 類別：
- add(): 將向量加入索引
- search(): 執行相似度搜尋
- save(): 儲存索引到硬碟
- load(): 從硬碟載入索引
```

## 執行流程

### 1. 環境準備

```bash
# 安裝必要套件
pip install sentence-transformers
pip install faiss-cpu
pip install PyPDF2
pip install numpy
```

### 2. 資料準備

將 PDF 檔案放在 `data/` 資料夾中：
```
data/
├── 2305.14325v1.pdf
├── 2502.14767v2.pdf
├── 2506.08292v1.pdf
└── 2509.05396v1.pdf
```

### 3. 執行主程式

```python
python rag_test.py
```

程式會執行以下步驟：

1. **處理所有 PDF**
   - 讀取每個 PDF 檔案
   - 擷取文字內容
   - 切割成適當大小的文字塊
   - 顯示處理進度

2. **生成 Embeddings**
   - 使用 all-MiniLM-L6-v2 模型
   - 批次處理文字塊
   - 生成 384 維向量

3. **建立 FAISS 索引**
   - 將所有向量加入索引
   - 儲存索引到 `faiss_index.index`
   - 儲存元資料到 `faiss_index.metadata`

4. **測試搜尋功能**
   - 輸入查詢文字
   - 生成查詢向量
   - 搜尋最相似的文字塊
   - 顯示搜尋結果

## 輸出檔案

執行後會產生：
- `faiss_index.index`: FAISS 索引檔案
- `faiss_index.metadata`: 文件元資料（包含原始文字塊）

## 程式碼結構

```
rag_test.py
├── Document 類別：儲存文字塊和元資料
├── PDFProcessor 類別：處理 PDF 檔案
├── EmbeddingModel 類別：文字向量化
├── FAISSVectorStore 類別：向量資料庫操作
├── process_all_pdfs()：批次處理 PDF
├── build_faiss_index()：建立 FAISS 索引
└── demo_search()：示範搜尋功能
```

## 使用範例

### 載入已存在的索引

```python
# 載入索引
vector_store = FAISSVectorStore()
vector_store.load("faiss_index.index", "faiss_index.metadata")

# 執行搜尋
embedding_model = EmbeddingModel()
query = "machine learning algorithms"
query_embedding = embedding_model.encode([query])
results = vector_store.search(query_embedding[0], k=5)

# 顯示結果
for doc, distance in results:
    print(f"來源: {doc.metadata['source']}")
    print(f"內容: {doc.content[:200]}...")
    print(f"距離: {distance}")
```

### 自訂參數

```python
# 自訂文字分塊大小
pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=100)

# 使用不同的 embedding 模型
embedding_model = EmbeddingModel(model_name="sentence-transformers/all-mpnet-base-v2")
```

## 效能考量

1. **記憶體使用**
   - all-MiniLM-L6-v2: 約 100MB
   - FAISS 索引: 取決於向量數量
   - 每個向量: 384 * 4 bytes = 1.5KB

2. **處理速度**
   - PDF 讀取: 取決於檔案大小
   - Embedding 生成: 約 100-200 個文字塊/秒（CPU）
   - FAISS 搜尋: 毫秒級

3. **準確度**
   - all-MiniLM-L6-v2 在大多數任務上表現良好
   - 如需更高準確度，可改用 all-mpnet-base-v2

## 注意事項

1. **CUDA 相容性**
   - 程式已設定強制使用 CPU (`device='cpu'`)
   - 避免 GPU 相容性問題

2. **文字編碼**
   - 確保 PDF 檔案包含可擷取的文字
   - 掃描的 PDF 需要先進行 OCR

3. **索引類型**
   - 目前使用 `IndexFlatL2`（精確搜尋）
   - 大規模資料可考慮使用 IVF 索引

## 後續優化建議

1. **加入重排序（Reranking）**
   - 使用交叉編碼器提升搜尋準確度

2. **實作 RAG 生成**
   - 整合 LLM 生成回答

3. **支援更多檔案格式**
   - 加入 Word、TXT、Markdown 等格式

4. **優化索引結構**
   - 使用 HNSW 或 IVF 加速大規模搜尋

5. **加入中文支援**
   - 使用支援中文的 embedding 模型

## 總結

本系統提供了完整的 PDF 文件向量化和搜尋功能，使用輕量級但高效的模型，適合在一般硬體上運行。FAISS 索引確保了快速的相似度搜尋，為後續的 RAG 應用奠定基礎。