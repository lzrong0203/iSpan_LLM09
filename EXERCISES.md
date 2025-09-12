# 📚 LLM 課程練習題庫

## 使用說明
- 練習題分為初級、中級、進階三個等級
- 每個練習都有詳細的提示和參考解答
- 建議按順序完成，循序漸進
- 遇到困難時可以查看提示，但建議先自己嘗試

---

## 🌱 初級練習（第一天上午）

### 練習 1-1：環境測試
**目標**：確認環境設置正確

**任務**：
1. 執行 Python，輸出版本號
2. 導入 torch，檢查是否有 GPU
3. 導入 transformers，確認版本

**提示**：
```python
import sys
import torch
import transformers

# 你的程式碼...
```

**預期輸出**：
```
Python 版本: 3.9.x
PyTorch 版本: 2.x.x
Transformers 版本: 4.x.x
GPU 可用: True/False
```

---

### 練習 1-2：第一個 LLM 程式
**目標**：使用預訓練模型生成文本

**任務**：
創建一個簡單的文本生成程式，輸入一句話，讓模型續寫。

**步驟提示**：
1. 載入一個小型預訓練模型（如 gpt2）
2. 輸入提示詞："今天天氣很好，"
3. 生成接下來的 20 個字

**程式碼框架**：
```python
from transformers import pipeline

# 創建文本生成器
generator = pipeline('text-generation', model='gpt2')

# 你的程式碼...
prompt = "今天天氣很好，"

# 生成文本
result = # 完成這裡

print(result)
```

---

### 練習 1-3：理解 Token
**目標**：了解文本如何被分割成 token

**任務**：
將以下句子進行 tokenization，並觀察結果：
- "Hello World"
- "你好世界"
- "LLM 很有趣！"

**程式碼框架**：
```python
from transformers import AutoTokenizer

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

sentences = ["Hello World", "你好世界", "LLM 很有趣！"]

for sentence in sentences:
    # 你的程式碼：進行 tokenization
    tokens = # ?
    token_ids = # ?
    
    print(f"原句: {sentence}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print("-" * 40)
```

---

## 🚀 中級練習（第一天下午）

### 練習 2-1：建立簡單問答系統
**目標**：創建一個可以回答問題的簡單系統

**需求**：
1. 系統能記住對話歷史
2. 能回答至少 3 種類型的問題
3. 有基本的錯誤處理

**程式碼框架**：
```python
class SimpleQA:
    def __init__(self):
        self.history = []
        self.knowledge = {
            "天氣": "今天天氣晴朗，氣溫 25 度",
            "時間": "現在是下午 2 點",
            "課程": "我們正在學習 LLM"
        }
    
    def answer(self, question):
        """根據問題返回答案"""
        # 你的程式碼
        pass
    
    def chat(self):
        """對話迴圈"""
        print("歡迎使用問答系統！（輸入 'quit' 結束）")
        while True:
            question = input("你的問題: ")
            if question.lower() == 'quit':
                break
            
            answer = self.answer(question)
            print(f"回答: {answer}\n")
            
            # 記錄對話
            self.history.append({"Q": question, "A": answer})

# 測試
qa = SimpleQA()
qa.chat()
```

---

### 練習 2-2：文本情感分析
**目標**：判斷文本的情感傾向

**任務**：
使用預訓練模型分析以下文本的情感：
1. "這部電影太棒了！"
2. "服務態度很差"
3. "還可以，普普通通"

**提示**：
- 使用 pipeline('sentiment-analysis')
- 處理中文可能需要指定模型

**評分標準**：
- 正確載入模型 (30%)
- 成功分析情感 (40%)
- 輸出格式清晰 (30%)

---

### 練習 2-3：實作文本摘要
**目標**：將長文本自動摘要

**輸入文本**：
```
人工智慧（AI）是電腦科學的一個分支，致力於創建能夠執行
通常需要人類智慧的任務的系統。這些任務包括學習、推理、
問題解決、感知和語言理解。近年來，隨著深度學習技術的
突破，AI 在許多領域取得了顯著進展，包括圖像識別、
自然語言處理和遊戲對弈等。
```

**要求**：
1. 將文本摘要成 1-2 句話
2. 保留關鍵信息
3. 使用中文輸出

---

## 🎯 進階練習（第二天）

### 練習 3-1：實作 RAG 系統
**目標**：建立檢索增強生成系統

**需求**：
1. 能夠索引至少 5 個文檔
2. 根據查詢檢索相關文檔
3. 基於檢索結果生成答案

**文檔範例**：
```python
documents = [
    "Python 是一種高階程式語言，由 Guido van Rossum 於 1991 年創建。",
    "機器學習是人工智慧的一個分支，讓電腦能夠從數據中學習。",
    "深度學習使用多層神經網絡來處理複雜的模式識別任務。",
    "Transformer 是一種基於注意力機制的神經網絡架構。",
    "GPT 是 Generative Pre-trained Transformer 的縮寫。"
]
```

**程式碼框架**：
```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleRAG:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
    
    def index_documents(self, docs):
        """索引文檔"""
        # 你的程式碼
        pass
    
    def search(self, query, top_k=2):
        """搜尋相關文檔"""
        # 你的程式碼
        pass
    
    def generate_answer(self, query):
        """生成答案"""
        # 1. 搜尋相關文檔
        # 2. 組合上下文
        # 3. 生成回答
        pass

# 測試
rag = SimpleRAG()
rag.index_documents(documents)
answer = rag.generate_answer("什麼是 Python？")
print(answer)
```

---

### 練習 3-2：Prompt 優化挑戰
**目標**：優化 prompt 以獲得更好的輸出

**任務場景**：
你需要讓 LLM 扮演一個專業的程式碼審查員

**初始 Prompt**：
```
"檢查這段程式碼"
```

**要求**：
1. 改進 prompt，使其更具體
2. 加入角色設定
3. 指定輸出格式
4. 提供評判標準

**評分標準**：
- Prompt 清晰度 (25%)
- 角色定義明確 (25%)
- 輸出格式規範 (25%)
- 實際效果提升 (25%)

---

### 練習 3-3：建立對話機器人
**目標**：創建一個有個性的對話機器人

**需求**：
1. 機器人有明確的個性設定
2. 能記住對話上下文
3. 回答符合其個性
4. 支援至少 5 輪對話

**個性選項**：
- 友善的客服
- 嚴謹的老師
- 幽默的朋友

**加分項**：
- 實作情緒識別
- 根據用戶情緒調整回應
- 保存對話歷史

---

## 💡 綜合專案

### 專案：智慧文檔助手
**時間**：2-3 小時

**目標**：
建立一個可以回答文檔相關問題的智慧助手

**功能需求**：
1. **文檔上傳**：支援 txt 文件
2. **智慧問答**：基於文檔內容回答問題
3. **多輪對話**：記住對話歷史
4. **來源標註**：標明答案來自哪個文檔

**技術要求**：
- 使用 RAG 技術
- 實作向量檢索
- 整合 LLM 生成

**評分標準**：
- 基本功能完成度 (40%)
- 程式碼品質 (20%)
- 回答準確性 (20%)
- 使用者體驗 (20%)

**提交內容**：
1. 完整程式碼
2. 使用說明文檔
3. 測試範例
4. 心得報告（選填）

---

## 📝 練習提示

### 給初學者的建議
1. **不要害怕錯誤**：錯誤訊息是最好的老師
2. **逐步調試**：一次只改一個地方
3. **查看文檔**：Hugging Face 有豐富的文檔
4. **使用 print**：多用 print 了解程式執行狀況

### 常見錯誤與解決
1. **ImportError**：檢查套件是否安裝
2. **CUDA Error**：改用 CPU 或檢查 GPU 設置
3. **Memory Error**：減少批次大小或使用更小的模型
4. **Timeout**：使用更短的文本或設置 max_length

### 學習資源
- [Hugging Face 教程](https://huggingface.co/docs)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [課程範例程式碼](./examples/)

---

## 🏆 挑戰任務（選做）

### 挑戰 1：多語言翻譯器
建立支援中、英、日三語互譯的系統

### 挑戰 2：程式碼生成器
輸入需求描述，生成對應的 Python 程式碼

### 挑戰 3：新聞摘要系統
自動抓取新聞並生成摘要

### 挑戰 4：個人知識庫
建立可以持續學習的個人知識管理系統

---

## 完成證明

完成所有必做練習後，您將能夠：
- ✅ 理解 LLM 的基本原理
- ✅ 使用預訓練模型
- ✅ 實作簡單的 AI 應用
- ✅ 優化模型輸出
- ✅ 建立實用的專案

祝學習愉快！💪