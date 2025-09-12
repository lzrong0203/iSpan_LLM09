# ❓ LLM 課程常見問題解答 (FAQ)

## 📑 目錄
1. [基礎概念問題](#基礎概念問題)
2. [環境設置問題](#環境設置問題)
3. [程式執行問題](#程式執行問題)
4. [模型相關問題](#模型相關問題)
5. [學習建議](#學習建議)

---

## 基礎概念問題

### Q1: 什麼是 LLM？跟一般的 AI 有什麼不同？

**答案：**
LLM (Large Language Model，大型語言模型) 是 AI 的一種，專門處理文字。

**簡單比喻：**
- 一般 AI = 各種智慧功能（像瑞士刀）
- LLM = 專精於理解和生成文字（像專業翻譯官）

**主要差異：**
| 一般 AI | LLM |
|---------|-----|
| 可能做圖像識別、語音辨識等 | 專注於文字處理 |
| 通常針對特定任務 | 可以處理多種文字任務 |
| 模型大小不一 | 參數量極大（數十億以上）|

---

### Q2: GPT、BERT、Llama 這些名字是什麼意思？

**答案：**

🤖 **GPT** (Generative Pre-trained Transformer)
- 由 OpenAI 開發
- 擅長生成文字
- ChatGPT 就是基於此技術

🔍 **BERT** (Bidirectional Encoder Representations from Transformers)
- 由 Google 開發
- 擅長理解文字
- 常用於搜尋引擎

🦙 **Llama** (Large Language Model Meta AI)
- 由 Meta (Facebook) 開發
- 開源且可商用
- 適合本地部署

---

### Q3: Token 是什麼？為什麼重要？

**答案：**

Token 就像是 LLM 理解文字的「基本單位」。

**舉例說明：**
```
句子: "我愛吃蘋果"
分割成 Tokens: ["我", "愛", "吃", "蘋果"]

句子: "I love apples"  
分割成 Tokens: ["I", " love", " app", "les"]
```

**為什麼重要：**
1. **計費依據**：API 通常按 token 數量收費
2. **長度限制**：模型有最大 token 限制
3. **效能影響**：token 越多，處理越慢

---

### Q4: 什麼是 Transformer？一定要懂嗎？

**答案：**

Transformer 是 LLM 的核心架構，像是汽車的引擎。

**需要了解的程度：**
- **初學者**：知道它存在即可，像開車不需要懂引擎原理
- **進階使用**：了解基本概念有助於優化
- **研究開發**：需要深入理解

**核心概念（簡化版）：**
```
輸入文字 → 注意力機制（判斷重要性）→ 理解上下文 → 輸出結果
```

---

## 環境設置問題

### Q5: 一定要有 GPU 嗎？沒有怎麼辦？

**答案：**

不一定需要 GPU，有多種替代方案：

**選項比較：**
| 方案 | 優點 | 缺點 | 適合場景 |
|------|------|------|----------|
| **CPU 執行** | 免費、隨時可用 | 速度慢 | 學習測試 |
| **Google Colab** | 免費 GPU | 有時間限制 | 練習作業 |
| **雲端 API** | 快速方便 | 需付費 | 實際應用 |
| **小模型** | 資源需求低 | 效果較差 | 初期學習 |

**建議：**
```python
# 檢查是否有 GPU
import torch
if torch.cuda.is_available():
    device = "cuda"  # 使用 GPU
else:
    device = "cpu"   # 使用 CPU
    print("使用 CPU 模式，速度會較慢")
```

---

### Q6: pip install 失敗怎麼辦？

**常見錯誤與解決方法：**

**1. 權限錯誤**
```bash
# 錯誤: Permission denied
# 解決：加上 --user
pip install --user transformers
```

**2. 網路逾時**
```bash
# 使用國內鏡像
pip install -i https://pypi.douban.com/simple transformers
```

**3. 版本衝突**
```bash
# 升級 pip
python -m pip install --upgrade pip

# 清理後重裝
pip uninstall transformers
pip install transformers
```

**4. Windows 編譯錯誤**
- 安裝 Visual Studio Build Tools
- 或下載預編譯的 wheel 檔

---

### Q7: 虛擬環境是什麼？為什麼需要？

**答案：**

虛擬環境就像是程式的「獨立房間」，避免不同專案的套件互相干擾。

**類比說明：**
```
電腦系統
├── 專案A的房間 (虛擬環境1)
│   ├── Python 3.8
│   └── 舊版套件
└── 專案B的房間 (虛擬環境2)
    ├── Python 3.9
    └── 新版套件
```

**創建和使用：**
```bash
# 創建
python -m venv myenv

# 啟動 (Windows)
myenv\Scripts\activate

# 啟動 (Mac/Linux)
source myenv/bin/activate

# 看到 (myenv) 表示成功
```

---

## 程式執行問題

### Q8: "CUDA out of memory" 錯誤

**問題原因：**
GPU 記憶體不足

**解決方法（依序嘗試）：**

```python
# 方法1: 減少批次大小
batch_size = 1  # 原本可能是 8 或 16

# 方法2: 使用較小的模型
model = "gpt2"  # 而不是 "gpt2-large"

# 方法3: 清理記憶體
import torch
torch.cuda.empty_cache()

# 方法4: 使用量化
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    load_in_8bit=True  # 8-bit 量化
)

# 方法5: 改用 CPU
device = "cpu"
```

---

### Q9: 模型下載很慢或失敗

**解決方案：**

**1. 使用鏡像站**
```python
# 設置環境變數
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

**2. 手動下載**
- 從 Hugging Face 網站下載
- 放到指定資料夾
- 從本地載入

**3. 使用較小模型**
```python
# 使用 distilled 版本（精簡版）
model = "distilbert-base-uncased"  # 而不是 "bert-base-uncased"
```

---

### Q10: 生成的文字很奇怪或重複

**常見原因與解決：**

```python
# 問題：重複輸出
# 原因：temperature 太低
response = generate(prompt, temperature=0.1)  # 太低

# 解決：調整參數
response = generate(
    prompt,
    temperature=0.7,      # 增加隨機性
    top_p=0.9,           # 限制選擇範圍
    repetition_penalty=1.2,  # 懲罰重複
    max_length=100       # 限制長度
)
```

**參數說明：**
- **temperature**：0.1（保守）→ 1.0（創意）
- **top_p**：只考慮機率最高的詞
- **repetition_penalty**：避免重複

---

## 模型相關問題

### Q11: 如何選擇適合的模型？

**選擇指南：**

| 需求 | 推薦模型 | 原因 |
|------|----------|------|
| **初學練習** | GPT-2, DistilBERT | 資源需求低 |
| **中文處理** | ChatGLM, Qwen | 中文優化 |
| **程式碼生成** | CodeLlama, StarCoder | 程式特化 |
| **本地部署** | Llama 2/3, Mistral | 開源可商用 |
| **高品質輸出** | GPT-4, Claude | 效果最佳 |

**決策流程：**
```
是否需要商用？
├─ 是 → Llama, Mistral (開源)
└─ 否 → 是否有預算？
         ├─ 是 → GPT-4 API
         └─ 否 → 開源小模型
```

---

### Q12: Fine-tuning 和 Prompt Engineering 的差別？

**簡單比較：**

| 方面 | Fine-tuning (微調) | Prompt Engineering |
|------|-------------------|-------------------|
| **比喻** | 訓練專屬助理 | 給助理詳細指示 |
| **成本** | 高（需要 GPU、時間）| 低（立即可用）|
| **效果** | 專業但狹窄 | 通用但可能不精確 |
| **難度** | 需要技術知識 | 相對簡單 |

**使用時機：**
```
先嘗試 Prompt Engineering
    ↓ 效果不佳
考慮 Few-shot Learning
    ↓ 還是不夠
才使用 Fine-tuning
```

---

### Q13: 什麼是 RAG？為什麼需要它？

**RAG = Retrieval Augmented Generation（檢索增強生成）**

**類比說明：**
```
一般 LLM = 閉卷考試（只靠記憶）
RAG = 開卷考試（可以查資料）
```

**優點：**
1. **更準確**：基於實際資料回答
2. **可更新**：不需重新訓練模型
3. **可追溯**：知道答案來源

**簡單實作概念：**
```python
def rag_answer(question):
    # 1. 搜尋相關文件
    relevant_docs = search(question)
    
    # 2. 組合成上下文
    context = "\n".join(relevant_docs)
    
    # 3. 讓 LLM 基於上下文回答
    prompt = f"根據以下資料回答：\n{context}\n\n問題：{question}"
    
    return llm.generate(prompt)
```

---

## 學習建議

### Q14: 完全沒有程式基礎，能學會嗎？

**答案：可以！但需要循序漸進**

**學習路線圖：**
```
第1週：Python 基礎
├── 變數、迴圈、函數
└── 安裝套件、讀寫檔案

第2週：使用現成工具
├── 調用 API
└── 使用 Hugging Face

第3週：簡單應用
├── 文字生成
└── 情感分析

第4週：進階專案
└── 建立自己的應用
```

**推薦資源：**
1. Python 基礎：W3Schools Python Tutorial
2. 視覺化學習：Google Colab
3. 實作範例：本課程的 examples 資料夾

---

### Q15: 如何持續學習和跟上最新發展？

**學習管道：**

📚 **基礎知識**
- Hugging Face 課程（免費）
- Fast.ai 課程
- YouTube 教學影片

📰 **最新資訊**
- Twitter: 追蹤 AI 研究者
- Reddit: r/MachineLearning
- Papers with Code: 最新論文

🛠️ **實作練習**
- Kaggle 競賽
- GitHub 開源專案
- 建立個人專案

**每週學習計畫：**
```
週一三五：閱讀一篇文章/教程（30分鐘）
週二四：動手實作練習（1小時）
週末：整理筆記、做小專案
```

---

### Q16: 遇到錯誤如何有效求助？

**求助模板：**

```markdown
## 問題描述
簡短說明你想做什麼

## 錯誤訊息
```
完整的錯誤訊息貼在這裡
```

## 已嘗試的解決方法
1. 方法一：結果如何
2. 方法二：結果如何

## 環境資訊
- Python 版本：
- 作業系統：
- 相關套件版本：
```

**求助管道：**
1. **Google 錯誤訊息**（80% 可解決）
2. **ChatGPT/Claude**（解釋錯誤）
3. **Stack Overflow**
4. **課程討論區**
5. **GitHub Issues**

---

## 💡 最後的建議

### 給初學者的話：

1. **不要追求完美理解**
   - 先會用，再深入
   - 做中學最有效

2. **從小專案開始**
   - 第一個專案：文字生成器
   - 第二個專案：簡單問答
   - 逐步增加複雜度

3. **養成好習慣**
   - 寫註解
   - 保存可運行的程式碼
   - 記錄學習筆記

4. **保持耐心**
   - LLM 領域變化快
   - 專注於核心概念
   - 工具會變，原理不變

---

## 🆘 緊急求助

如果以上都無法解決你的問題：

1. **課堂時間**：直接舉手發問
2. **課後**：Email 給講師（附上錯誤截圖）
3. **自學**：加入 Discord/Line 學習群組

記住：**沒有愚蠢的問題，只有不問的遺憾！**

祝學習順利！ 🚀