# Day 2 上午：Prompt工程技巧
# 11:30-12:30 進階Prompt技術

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict

# 載入環境變數
load_dotenv()

# === Step 1: 初始化 ===
print("=== Prompt工程技巧 ===")
print()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# === Step 2: Zero-shot CoT ===
print("=== Zero-shot Chain of Thought (CoT) ===")
print("讓AI一步步思考")
print()

def zero_shot_cot(question: str) -> str:
    """Zero-shot CoT推理"""
    prompt = f"""{question}

讓我們一步步思考這個問題。"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"錯誤：{e}"

# 測試Zero-shot CoT
math_problem = "小明有15顆蘋果，給了小華3顆，又買了7顆，請問現在有幾顆？"
print(f"問題：{math_problem}")
print("回答：")
print(zero_shot_cot(math_problem))
print("\n" + "="*50)

# === Step 3: Few-shot Learning ===
print("\n=== Few-shot Learning ===")
print("提供範例讓AI學習")
print()

def few_shot_learning(task: str, examples: List[Dict], query: str) -> str:
    """Few-shot學習"""
    
    # 建立prompt
    prompt = f"任務：{task}\n\n"
    prompt += "範例：\n"
    
    for i, example in enumerate(examples, 1):
        prompt += f"\n範例{i}：\n"
        prompt += f"輸入：{example['input']}\n"
        prompt += f"輸出：{example['output']}\n"
    
    prompt += f"\n現在請處理：\n"
    prompt += f"輸入：{query}\n"
    prompt += f"輸出："
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"錯誤：{e}"

# 測試Few-shot
examples = [
    {"input": "今天天氣真好", "output": "正面"},
    {"input": "這部電影太無聊了", "output": "負面"},
    {"input": "食物還可以", "output": "中性"}
]

test_input = "這個產品超棒的！"
print("情感分析任務")
result = few_shot_learning("判斷句子的情感", examples, test_input)
print(f"輸入：{test_input}")
print(f"情感：{result}")
print("\n" + "="*50)

# === Step 4: 結構化Prompt ===
print("\n=== 結構化Prompt ===")
print("使用結構化格式提高準確性")
print()

def structured_prompt(data: Dict) -> str:
    """結構化Prompt"""
    
    prompt = f"""
### 任務
{data['task']}

### 輸入資料
{data['input']}

### 要求
{chr(10).join('- ' + req for req in data['requirements'])}

### 輸出格式
{data['output_format']}

### 回答
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"錯誤：{e}"

# 測試結構化Prompt
task_data = {
    "task": "摘要文章重點",
    "input": "Python是一種高階程式語言，具有簡潔易讀的語法。它支援多種程式設計範式，包括物件導向、程序式和函數式程式設計。Python有豐富的標準庫和活躍的社群。",
    "requirements": [
        "摘要不超過50字",
        "包含主要特點",
        "使用繁體中文"
    ],
    "output_format": "用1-2句話摘要"
}

result = structured_prompt(task_data)
print("結構化Prompt結果：")
print(result)
print("\n" + "="*50)

# === Step 5: ReAct框架 ===
print("\n=== ReAct (Reasoning + Acting) ===")
print("結合推理和行動")
print()

class ReActAgent:
    """ReAct代理"""
    
    def __init__(self):
        self.thoughts = []
        self.actions = []
        self.observations = []
    
    def think(self, question: str) -> str:
        """思考步驟"""
        thought_prompt = f"""
問題：{question}

Thought: 我需要思考如何解決這個問題。
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一個會思考和行動的AI助手"},
                    {"role": "user", "content": thought_prompt}
                ],
                temperature=0.5
            )
            thought = response.choices[0].message.content
            self.thoughts.append(thought)
            return thought
        except Exception as e:
            return f"思考錯誤：{e}"
    
    def act(self, action: str) -> str:
        """執行動作"""
        # 模擬執行動作
        if "搜尋" in action:
            observation = "找到相關資料：Python是1991年創造的程式語言"
        elif "計算" in action:
            observation = "計算結果：42"
        else:
            observation = "動作已執行"
        
        self.actions.append(action)
        self.observations.append(observation)
        return observation
    
    def solve(self, question: str) -> str:
        """完整的ReAct流程"""
        print(f"Question: {question}")
        
        # Step 1: 思考
        thought = self.think(question)
        print(f"Thought: {thought}")
        
        # Step 2: 行動
        action = "搜尋相關資料"
        print(f"Action: {action}")
        
        # Step 3: 觀察
        observation = self.act(action)
        print(f"Observation: {observation}")
        
        # Step 4: 最終答案
        final_prompt = f"""
基於以下資訊回答問題：
問題：{question}
思考：{thought}
觀察：{observation}

最終答案："""
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"錯誤：{e}"

# 測試ReAct
agent = ReActAgent()
question = "Python是什麼時候被創造的？"
answer = agent.solve(question)
print(f"Answer: {answer}")
print("\n" + "="*50)

# === Step 6: Prompt模板庫 ===
print("\n=== Prompt模板庫 ===")
print()

class PromptTemplates:
    """常用Prompt模板"""
    
    @staticmethod
    def translation(text: str, target_lang: str) -> str:
        return f"請將以下文字翻譯成{target_lang}，保持原意：\n\n{text}"
    
    @staticmethod
    def summarization(text: str, max_words: int) -> str:
        return f"請用不超過{max_words}字摘要以下內容：\n\n{text}"
    
    @staticmethod
    def code_review(code: str) -> str:
        return f"""請檢查以下程式碼，提供改進建議：

```python
{code}
```

請從以下方面分析：
1. 程式碼品質
2. 潛在錯誤
3. 效能優化
4. 最佳實踐"""
    
    @staticmethod
    def creative_writing(topic: str, style: str) -> str:
        return f"請以{style}的風格，寫一段關於{topic}的創意文字（約100字）"

# 測試模板
templates = PromptTemplates()

# 測試翻譯模板
translate_prompt = templates.translation("Hello World", "日文")
print("翻譯模板：")
print(translate_prompt)
print()

# 測試程式碼審查模板
code = """
def add(a, b):
    return a + b
"""
review_prompt = templates.code_review(code)
print("程式碼審查模板：")
print(review_prompt[:100] + "...")

# === Step 7: Prompt優化建議 ===
print("\n" + "="*50)
print("=== Prompt優化建議 ===")
print()

tips = [
    "1. 明確性：清楚說明任務和期望",
    "2. 結構化：使用標題、列表等組織資訊",
    "3. 範例：提供輸入輸出範例",
    "4. 限制：設定字數、格式等限制",
    "5. 角色：指定AI扮演的角色",
    "6. 步驟：分解複雜任務為步驟",
    "7. 格式：指定輸出格式（JSON、表格等）"
]

for tip in tips:
    print(f"✅ {tip}")

print()
print("💡 記住：好的Prompt是獲得好答案的關鍵！")