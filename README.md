# 人工智慧大型語言模型實作應用班
## Large Language Model Practical Application

**課程時間：2025年9月13-14日**  
**總時數：12小時**

---

> ⚠️ **重要提醒 / Important Notice**
> 
> 本教材內容為 AI (Claude Opus 4.1) 自動生成的初步版本，僅供參考使用。
> 
> 實際課程內容將根據授課需求持續調整與優化，請以最終版本為準。
> 
> *This material is an AI-generated preliminary version for reference only.*
> 
> *The actual course content will be continuously adjusted and optimized based on teaching requirements.*

---

## 📎 課程資料

- **投影片：** [2025-0913 LLM.pdf](./2025-0913%20LLM.pdf)
- **更新時間：** 2025年9月9日

---

# 課程大綱總覽

## 第一天 (6小時)
- **09:00-10:30** - LLM基本概念與架構
- **10:45-12:00** - 環境設置與工具準備
- **13:00-14:30** - 不使用框架開發Chatbot (Part 1)
- **14:45-16:00** - 不使用框架開發Chatbot (Part 2)

## 第二天 (6小時)
- **09:00-10:30** - 微調Llama模型
- **10:45-12:00** - LoRA技術與Prompt Engineering
- **13:00-14:30** - RAG技術深入應用
- **14:45-16:00** - LLM Agent實作與部署

---

# 模組一：大型語言模型基本概念

## 什麼是大型語言模型？

**定義：** 基於深度學習的自然語言處理模型，能夠理解和生成人類語言

**特點：** 參數量龐大（數十億到數千億）、預訓練-微調架構、強大的泛化能力

## LLM的核心架構

```
輸入文本 → Tokenization → Transformer → 輸出預測
```

### 關鍵組件說明：
- **Tokenization：** 將文本轉換為模型可理解的數字序列
- **Embedding Layer：** 將token映射到高維向量空間
- **Transformer Blocks：** 自注意力機制 + 前饋網絡
- **Output Layer：** 生成下一個token的概率分布

---

# Transformer架構深入剖析

## Self-Attention機制

```python
# Self-Attention 計算流程
def attention(Q, K, V):
    """
    Q (Query): 查詢向量
    K (Key): 鍵向量  
    V (Value): 值向量
    d_k: 鍵向量的維度
    """
    d_k = K.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output
```

## Multi-Head Attention
- 平行執行多個attention操作
- 捕捉不同層次的語義關係
- 增強模型的表達能力

## Position Encoding

**為什麼需要位置編碼？**
Transformer本身沒有順序概念，需要通過位置編碼來注入序列信息

---

# 主流LLM模型比較

| 模型系列 | 開發者 | 參數量 | 特點 | 應用場景 |
|---------|--------|--------|------|----------|
| **GPT-4** | OpenAI | 約1.8T | 多模態、強大推理能力 | 通用對話、程式碼生成 |
| **Llama 3** | Meta | 8B-70B | 開源、可本地部署 | 企業私有化部署 |
| **Claude** | Anthropic | 未公開 | 安全性高、長文本處理 | 文檔分析、安全應用 |
| **Gemini** | Google | 多版本 | 原生多模態 | 多媒體理解與生成 |

## 選擇考量因素
- **成本：** API費用 vs 本地部署成本
- **隱私：** 資料安全性需求
- **效能：** 推理速度與準確度平衡
- **客製化：** 是否需要微調

---

# 模組二：環境設置與工具準備

## 1. Python環境配置

```bash
# 建議使用Python 3.9+
# 創建虛擬環境
python -m venv llm_env
source llm_env/bin/activate  # Linux/Mac
# 或
llm_env\Scripts\activate  # Windows

# 安裝必要套件
pip install torch torchvision transformers
pip install numpy pandas matplotlib
pip install sentencepiece protobuf
pip install accelerate bitsandbytes
```

## 2. GPU環境檢查

```python
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU數量: {torch.cuda.device_count()}")
    print(f"GPU名稱: {torch.cuda.get_device_name(0)}")
```

**⚠️ 注意：** 運行大型模型建議至少16GB VRAM，如無GPU可使用Google Colab

---

# OpenAI API設置與使用

## API Key申請流程
1. 訪問 platform.openai.com
2. 註冊/登入帳號
3. 前往 API Keys 頁面
4. 創建新的 API Key
5. 安全保存 Key（只顯示一次）

## 基礎API調用範例

```python
import openai
import os

# 設置API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 基本對話調用
def chat_with_gpt(prompt, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一個專業的AI助手"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content

# 使用範例
result = chat_with_gpt("解釋什麼是機器學習")
print(result)
```

---

# Llama 3 本地部署實戰

## 方法一：使用Hugging Face

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 載入模型和tokenizer
model_name = "meta-llama/Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 生成文本
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 方法二：使用llama.cpp (更省資源)

```python
# 安裝llama-cpp-python
# pip install llama-cpp-python

from llama_cpp import Llama

# 載入量化模型
llm = Llama(
    model_path="./models/llama-3-8b.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8
)

# 生成回應
output = llm(
    "Q: 什麼是人工智慧? A:", 
    max_tokens=256
)
print(output['choices'][0]['text'])
```

---

# 模組三：不使用框架開發Chatbot (Part 1)

## 為什麼不使用LangChain？

### 優勢
- **更深入理解：** 掌握底層運作原理
- **靈活性更高：** 可完全客製化流程
- **效能優化：** 避免不必要的抽象層
- **除錯更容易：** 直接控制每個步驟

## 基礎聊天機器人架構

```python
class SimpleChatbot:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []
        
    def add_message(self, role, content):
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
    def generate_response(self, user_input):
        # 添加用戶輸入到歷史
        self.add_message("user", user_input)
        
        # 構建prompt
        prompt = self._build_prompt()
        
        # 生成回應
        response = self._generate(prompt)
        
        # 添加回應到歷史
        self.add_message("assistant", response)
        
        return response
        
    def _build_prompt(self):
        # 將對話歷史轉換為模型輸入格式
        prompt = ""
        for msg in self.conversation_history:
            if msg["role"] == "user":
                prompt += f"用戶: {msg['content']}\n"
            else:
                prompt += f"助手: {msg['content']}\n"
        prompt += "助手: "
        return prompt
```

---

# 資料處理與向量嵌入

## 文本預處理

```python
import re
from typing import List

class TextProcessor:
    def __init__(self):
        self.chunk_size = 512
        self.overlap = 50
        
    def clean_text(self, text: str) -> str:
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？]', '', text)
        return text.strip()
    
    def split_into_chunks(self, text: str) -> List[str]:
        """將長文本切分為固定大小的chunks"""
        chunks = []
        text = self.clean_text(text)
        
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def load_documents(self, file_paths: List[str]) -> List[str]:
        """載入並處理多個文檔"""
        all_chunks = []
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = self.split_into_chunks(text)
                all_chunks.extend(chunks)
        return all_chunks
```

## 生成向量嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """將文本列表轉換為向量"""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings
    
    def compute_similarity(self, query_embedding, doc_embeddings):
        """計算查詢與文檔的相似度"""
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            doc_embeddings
        )[0]
        return similarities
```

---

# 模組四：RAG系統實作

## 什麼是RAG (Retrieval Augmented Generation)?

### RAG工作流程
```
用戶查詢 → 檢索相關文檔 → 上下文增強 → 生成回答
```

## 完整RAG系統實作

```python
class RAGSystem:
    def __init__(self, llm_model, embedding_model):
        self.llm = llm_model
        self.embedder = EmbeddingGenerator(embedding_model)
        self.documents = []
        self.doc_embeddings = None
        
    def index_documents(self, documents: List[str]):
        """索引文檔並生成嵌入"""
        self.documents = documents
        self.doc_embeddings = self.embedder.encode_texts(documents)
        print(f"已索引 {len(documents)} 個文檔片段")
        
    def retrieve_relevant_docs(self, query: str, top_k: int = 3):
        """檢索最相關的文檔"""
        query_embedding = self.embedder.encode_texts([query])
        similarities = self.embedder.compute_similarity(
            query_embedding,
            self.doc_embeddings
        )
        
        # 獲取top-k最相關的文檔
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_docs = [self.documents[i] for i in top_indices]
        scores = [similarities[i] for i in top_indices]
        
        return relevant_docs, scores
    
    def generate_answer(self, query: str, context_docs: List[str]):
        """基於檢索的文檔生成答案"""
        # 構建增強的prompt
        context = "\n\n".join(context_docs)
        prompt = f"""
        基於以下相關文檔回答問題：
        
        文檔內容：
        {context}
        
        問題：{query}
        
        請根據提供的文檔內容回答，如果文檔中沒有相關信息，請說明。
        
        回答：
        """
        
        # 使用LLM生成回答
        response = self.llm.generate(prompt)
        return response
        
    def query(self, question: str):
        """完整的RAG查詢流程"""
        # 1. 檢索相關文檔
        relevant_docs, scores = self.retrieve_relevant_docs(question)
        
        # 2. 生成答案
        answer = self.generate_answer(question, relevant_docs)
        
        return {
            "question": question,
            "answer": answer,
            "sources": relevant_docs,
            "scores": scores
        }
```

---

# 模組五：使用自有資料微調Llama模型

## 微調的意義與應用場景
- **領域適應：** 讓模型學習特定領域知識
- **任務優化：** 針對特定任務提升表現
- **風格調整：** 調整回應風格符合需求

## 準備訓練資料

```python
import json
import pandas as pd

class DatasetPreparer:
    def __init__(self):
        self.data = []
        
    def create_instruction_dataset(self, qa_pairs):
        """創建指令微調資料集"""
        formatted_data = []
        
        for qa in qa_pairs:
            entry = {
                "instruction": qa["question"],
                "input": "",
                "output": qa["answer"]
            }
            formatted_data.append(entry)
        
        return formatted_data
    
    def save_dataset(self, data, output_path):
        """保存資料集為JSON格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def create_conversation_format(self, conversations):
        """創建對話格式的訓練資料"""
        formatted = []
        for conv in conversations:
            messages = []
            for turn in conv:
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"]
                })
            formatted.append({"messages": messages})
        
        return formatted
```

## 資料格式範例

```json
{
  "instruction": "什麼是深度學習？",
  "input": "",
  "output": "深度學習是機器學習的分支，使用多層神經網絡..."
}
```

---

# 微調訓練實作

## 使用Transformers進行微調

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

class LlamaFineTuner:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """載入預訓練模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 添加padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def prepare_dataset(self, data):
        """準備訓練資料集"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512
            )
        
        dataset = Dataset.from_dict({"text": data})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def setup_training_args(self):
        """設置訓練參數"""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            gradient_accumulation_steps=4,
            fp16=True,
        )
    
    def train(self, train_dataset, eval_dataset=None):
        """執行訓練"""
        training_args = self.setup_training_args()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
        )
        
        # 開始訓練
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
```

---

# 模組六：LoRA技術應用

## 什麼是LoRA？

**LoRA (Low-Rank Adaptation)**
一種參數高效的微調方法，通過在原始模型中插入可訓練的低秩矩陣，大幅減少需要訓練的參數量

## LoRA的優勢
- ⚡ **訓練效率高：** 只需訓練少量參數（通常< 1%）
- 💾 **儲存空間小：** LoRA權重檔案通常只有幾MB
- 🔄 **切換方便：** 可以快速切換不同的LoRA適配器
- 🎯 **任務特定：** 為不同任務訓練不同的LoRA

## LoRA實作程式碼

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

class LoRATrainer:
    def __init__(self, base_model_name):
        self.base_model_name = base_model_name
        
    def setup_lora_model(self):
        """設置LoRA配置並創建PEFT模型"""
        # 載入基礎模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA秩
            lora_alpha=32,  # LoRA縮放參數
            lora_dropout=0.1,  # Dropout率
            target_modules=[  # 要應用LoRA的模組
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj"
            ],
            bias="none"
        )
        
        # 創建PEFT模型
        peft_model = get_peft_model(base_model, lora_config)
        
        # 顯示可訓練參數統計
        peft_model.print_trainable_parameters()
        
        return peft_model
    
    def merge_and_save(self, peft_model, output_path):
        """合併LoRA權重並保存"""
        # 合併權重
        merged_model = peft_model.merge_and_unload()
        
        # 保存合併後的模型
        merged_model.save_pretrained(output_path)
        
        print(f"模型已保存到: {output_path}")
```

---

# 模組七：Prompt Engineering精要

## Prompt設計原則

### 🎯 核心原則
1. **明確性：** 清楚表達你的需求
2. **具體性：** 提供具體的指示和範例
3. **結構化：** 使用清晰的格式和分隔符
4. **角色設定：** 賦予模型明確的角色

## 常用Prompt技巧

| 技巧 | 說明 | 範例 |
|-----|------|------|
| **Few-shot** | 提供範例 | 輸入：蘋果→水果<br>輸入：汽車→? |
| **Chain-of-Thought** | 步驟思考 | 讓我們一步步思考這個問題... |
| **Role Playing** | 角色扮演 | 你是一位專業的數據分析師... |
| **Output Format** | 指定格式 | 請以JSON格式回答... |

## 進階Prompt範例

```python
class PromptTemplates:
    @staticmethod
    def create_system_prompt(role, constraints):
        return f"""
        你是一位{role}。
        
        你的任務規範：
        {constraints}
        
        請始終遵守以上規範回答問題。
        """
    
    @staticmethod
    def create_cot_prompt(question):
        return f"""
        問題：{question}
        
        讓我們一步步分析這個問題：
        1. 首先，理解問題的關鍵要素
        2. 其次，分析可能的解決方案
        3. 然後，評估每個方案的優缺點
        4. 最後，給出最佳建議
        
        分析過程：
        """
    
    @staticmethod
    def create_few_shot_prompt(examples, query):
        prompt = "以下是一些範例：\n\n"
        for ex in examples:
            prompt += f"輸入：{ex['input']}\n"
            prompt += f"輸出：{ex['output']}\n\n"
        prompt += f"現在，請處理以下輸入：\n"
        prompt += f"輸入：{query}\n輸出："
        return prompt
```

---

# 模組八：RAG技術深入應用

## 進階RAG優化策略

### 1. Hybrid Search (混合檢索)

```python
class HybridRAG:
    def __init__(self):
        self.dense_retriever = None  # 向量檢索
        self.sparse_retriever = None  # BM25檢索
        
    def hybrid_search(self, query, alpha=0.5):
        """結合密集和稀疏檢索"""
        # 密集檢索（向量相似度）
        dense_scores = self.dense_retriever.search(query)
        
        # 稀疏檢索（BM25）
        sparse_scores = self.sparse_retriever.search(query)
        
        # 分數融合
        combined_scores = (
            alpha * dense_scores + 
            (1 - alpha) * sparse_scores
        )
        
        return combined_scores
```

### 2. Query Expansion (查詢擴展)

```python
def expand_query(original_query, llm):
    """使用LLM擴展查詢"""
    prompt = f"""
    原始查詢：{original_query}
    
    請生成3個相關的擴展查詢，以獲得更全面的搜索結果：
    1.
    2.
    3.
    """
    
    expanded = llm.generate(prompt)
    return [original_query] + expanded.split('\n')
```

### 3. Re-ranking (重新排序)

**兩階段檢索策略：**
- 第一階段：快速檢索大量候選文檔
- 第二階段：使用更精確的模型重新排序

---

# 模組九：LLM Agent系統設計

## 什麼是LLM Agent？

LLM Agent是能夠自主執行任務、使用工具、與環境互動的智能系統

## Agent核心組件

```
感知(Perception) → 推理(Reasoning) → 行動(Action) → 記憶(Memory)
```

## 基礎Agent實作

```python
class SimpleLLMAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []
        self.max_iterations = 5
        
    def think(self, task, context):
        """推理下一步行動"""
        prompt = f"""
        任務：{task}
        當前上下文：{context}
        可用工具：{list(self.tools.keys())}
        
        請決定下一步行動：
        思考：
        行動：
        工具：
        參數：
        """
        
        response = self.llm.generate(prompt)
        return self.parse_action(response)
    
    def execute_tool(self, tool_name, params):
        """執行工具"""
        if tool_name in self.tools:
            return self.tools[tool_name](params)
        else:
            return "工具不存在"
    
    def run(self, task):
        """執行任務"""
        context = ""
        
        for i in range(self.max_iterations):
            # 思考
            action = self.think(task, context)
            
            # 執行
            if action["type"] == "tool":
                result = self.execute_tool(
                    action["tool"],
                    action["params"]
                )
                context += f"\n工具結果：{result}"
                
            elif action["type"] == "answer":
                return action["content"]
            
            # 記憶
            self.memory.append({
                "iteration": i,
                "action": action,
                "context": context
            })
        
        return "達到最大迭代次數"
```

---

# Agent工具整合

## 常用工具類型
- 🔍 **搜索工具：** Google Search API、Wikipedia
- 📊 **數據分析：** Python執行器、SQL查詢
- 📁 **文件處理：** PDF讀取、Excel操作
- 🌐 **網路請求：** API調用、網頁爬取

## 工具整合範例

```python
import requests
import pandas as pd
from datetime import datetime

class AgentTools:
    @staticmethod
    def web_search(query):
        """網路搜索工具"""
        # 實作搜索API調用
        api_key = "your_api_key"
        url = f"https://api.search.com/search?q={query}"
        response = requests.get(url, headers={"API-Key": api_key})
        return response.json()
    
    @staticmethod
    def calculate(expression):
        """數學計算工具"""
        try:
            result = eval(expression)
            return f"計算結果：{result}"
        except Exception as e:
            return f"計算錯誤：{str(e)}"
    
    @staticmethod
    def read_csv(file_path):
        """CSV讀取工具"""
        try:
            df = pd.read_csv(file_path)
            return df.describe().to_string()
        except Exception as e:
            return f"讀取錯誤：{str(e)}"
    
    @staticmethod
    def get_current_time():
        """獲取當前時間"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 註冊工具
tools = {
    "search": AgentTools.web_search,
    "calculate": AgentTools.calculate,
    "read_csv": AgentTools.read_csv,
    "get_time": AgentTools.get_current_time
}

# 創建Agent
agent = SimpleLLMAgent(llm_model, tools)
result = agent.run("幫我分析sales.csv的數據趨勢")
```

---

# 系統部署與優化

## 部署架構選擇

| 部署方式 | 優點 | 缺點 | 適用場景 |
|---------|------|------|----------|
| **本地部署** | 資料安全、無延遲 | 硬體要求高 | 企業內部應用 |
| **雲端API** | 維護簡單、擴展性好 | 成本較高、資料隱私 | 公開服務 |
| **邊緣部署** | 低延遲、離線可用 | 模型大小受限 | 移動應用 |

## FastAPI部署範例

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    context: list = []

class ChatResponse(BaseModel):
    response: str
    sources: list = []

# 初始化RAG系統
rag_system = RAGSystem(llm_model, embedding_model)
rag_system.index_documents(documents)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # RAG查詢
        result = rag_system.query(request.message)
        
        return ChatResponse(
            response=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

# 性能優化策略

## 模型優化技術

### 1. 量化 (Quantization)

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 2. 批次處理

```python
def batch_inference(texts, batch_size=8):
    """批次推理優化"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = model.generate(**inputs)
        
        batch_results = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        results.extend(batch_results)
    
    return results
```

### 3. 快取策略

**實作建議：**
- 使用Redis快取常見查詢
- 實作向量資料庫(Pinecone, Weaviate)
- 預計算並儲存embeddings

---

# 實戰練習專案

## 專案：建立企業知識問答系統

### 📋 需求說明
建立一個能夠回答公司內部文檔問題的智能助手

### 🎯 功能要求
- 支援PDF、Word、TXT文檔上傳
- 實作RAG檢索增強生成
- 提供來源追溯功能
- 支援中英文混合查詢

### ⚙️ 技術架構
- 後端：FastAPI + Llama 3
- 向量資料庫：FAISS/ChromaDB
- 前端：Streamlit/Gradio

## 實作步驟
1. **文檔處理：** 實作多格式文檔解析器
2. **向量化：** 使用sentence-transformers生成embeddings
3. **索引建立：** 建立向量資料庫索引
4. **查詢處理：** 實作語意搜索與關鍵字搜索
5. **答案生成：** 整合Llama模型生成回答
6. **介面開發：** 建立使用者友善的Web介面

### 評分標準
- 功能完整性 (30%)
- 回答準確度 (30%)
- 系統效能 (20%)
- 程式碼品質 (20%)

---

# 常見問題與解決方案

## 問題1：記憶體不足 (OOM)

**解決方案：**
- 使用量化技術 (4-bit/8-bit)
- 減少batch size
- 使用gradient checkpointing
- 考慮使用更小的模型

## 問題2：推理速度慢

**解決方案：**
- 實作批次處理
- 使用Flash Attention
- 部署模型到GPU
- 使用ONNX優化

## 問題3：生成品質不佳

**解決方案：**
- 優化Prompt設計
- 調整temperature和top_p參數
- 增加相關上下文
- 考慮微調模型

## 問題4：中文處理效果差

**解決方案：**
- 選擇支援中文的模型 (如ChatGLM)
- 使用中文特定的tokenizer
- 準備高品質中文訓練資料
- 調整分詞策略

---

# LLM技術未來發展趨勢

## 🚀 技術趨勢
- **多模態融合：** 文本、圖像、音訊統一處理
- **長文本處理：** 支援百萬token級別輸入
- **實時學習：** 模型能夠即時更新知識
- **自主Agent：** 更智能的任務規劃與執行

## 📈 應用方向
- **垂直領域深化：** 醫療、法律、金融專業模型
- **個人化AI助理：** 完全客製化的智能助手
- **程式碼生成：** 從需求直接生成完整應用
- **科學研究：** 加速科學發現與創新

## ⚡ 效能優化
- **模型壓縮：** 更小但更強大的模型
- **硬體加速：** 專用AI晶片優化
- **分散式推理：** 多設備協同計算
- **邊緣部署：** 手機端運行大模型

### 學習建議
持續關注最新論文、參與開源專案、實踐真實場景應用

---

# 課程總結與展望

## ✅ 我們學到了什麼
- 深入理解LLM的原理與架構
- 掌握不依賴框架的開發方法
- 實作完整的RAG系統
- 學會模型微調與優化技術
- 建構可部署的LLM應用

## 🎯 核心能力

### 技術層面
- 獨立開發LLM應用的能力
- 解決實際問題的工程能力
- 系統優化與調試能力

### 思維層面
- 理解AI系統的設計思維
- 評估技術選型的決策能力
- 持續學習的方法論

## 📚 後續學習資源
- **論文閱讀：** ArXiv、Papers with Code
- **開源專案：** Hugging Face、GitHub
- **社群交流：** Reddit r/MachineLearning、Discord
- **實戰平台：** Kaggle、Google Colab

---

# 🌟 結語

> LLM技術正在改變世界，而你已經掌握了開啟這扇大門的鑰匙。
> 
> 持續探索、勇於創新，成為AI時代的建設者！

---

*Generated with Claude Opus 4.1*