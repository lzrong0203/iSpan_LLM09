"""
資料集準備工具
用於準備微調所需的訓練資料
"""

import json
import pandas as pd
from typing import List, Dict
import random


class DatasetPreparer:
    """資料集準備類"""
    
    def __init__(self):
        self.data = []
        
    def create_instruction_dataset(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        創建指令微調資料集
        
        Args:
            qa_pairs: 問答對列表
            
        Returns:
            格式化的資料集
        """
        formatted_data = []
        
        for qa in qa_pairs:
            entry = {
                "instruction": qa["question"],
                "input": "",
                "output": qa["answer"]
            }
            formatted_data.append(entry)
        
        return formatted_data
    
    def save_dataset(self, data: List[Dict], output_path: str, format: str = "json"):
        """
        保存資料集
        
        Args:
            data: 資料集
            output_path: 輸出路徑
            format: 保存格式 (json/jsonl)
        """
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"資料集已保存到：{output_path}")
    
    def create_conversation_format(self, conversations: List[List[Dict]]) -> List[Dict]:
        """
        創建對話格式的訓練資料
        
        Args:
            conversations: 對話列表
            
        Returns:
            格式化的對話資料
        """
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
    
    def create_ptt_style_dataset(self, topics: List[str], num_samples: int = 100) -> List[Dict]:
        """
        創建 PTT 風格的資料集
        
        Args:
            topics: 主題列表
            num_samples: 樣本數量
            
        Returns:
            PTT 風格的資料集
        """
        ptt_responses = [
            "認真回，這個問題其實蠻複雜的",
            "先說結論：沒那麼簡單",
            "我朋友就是做這個的，他說",
            "之前看過一篇文章有講到",
            "笑死，這都不知道嗎",
            "推文小心，這個敏感",
            "有掛嗎？求站內信",
            "這個之前爆過了吧",
            "等等，你確定是這樣？",
            "補充一下樓上說的"
        ]
        
        ptt_endings = [
            "就醬",
            "以上",
            "懂？",
            "沒了",
            "End",
            "大概是這樣",
            "自己想想吧",
            "參考看看",
            "純屬個人意見",
            "有錯請指正"
        ]
        
        dataset = []
        
        for i in range(num_samples):
            topic = random.choice(topics)
            response_start = random.choice(ptt_responses)
            response_end = random.choice(ptt_endings)
            
            # 創建 PTT 風格的回答
            answer = f"{response_start}，{topic}的問題啊，{response_end}"
            
            question = f"關於{topic}，你怎麼看？"
            
            dataset.append({
                "instruction": question,
                "output": answer
            })
        
        return dataset
    
    def load_and_format_ptt_data(self, filepath: str) -> List[Dict]:
        """
        載入並格式化 PTT 資料
        
        Args:
            filepath: PTT 資料檔案路徑
            
        Returns:
            格式化的資料集
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        
        formatted = []
        for item in data:
            # 根據實際的資料格式調整
            if 'text' in item:
                formatted.append(item)
            elif 'instruction' in item and 'output' in item:
                formatted.append({
                    "text": f"### 指令：{item['instruction']}\n### 回答：{item['output']}"
                })
            else:
                # 其他格式的處理
                formatted.append(item)
        
        return formatted
    
    def split_dataset(self, data: List[Dict], train_ratio: float = 0.8) -> tuple:
        """
        分割資料集為訓練集和驗證集
        
        Args:
            data: 完整資料集
            train_ratio: 訓練集比例
            
        Returns:
            (訓練集, 驗證集)
        """
        random.shuffle(data)
        split_idx = int(len(data) * train_ratio)
        
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        return train_data, val_data


def create_sample_ptt_dataset():
    """創建範例 PTT 資料集"""
    
    # PTT 八卦版常見話題
    topics = [
        "台灣房價", "外送平台", "加密貨幣", "疫苗", "選舉",
        "颱風", "地震", "股市", "薪資", "物價",
        "交通", "教育", "健保", "能源", "環保"
    ]
    
    # 創建資料集準備器
    preparer = DatasetPreparer()
    
    # 生成 PTT 風格資料集
    dataset = preparer.create_ptt_style_dataset(topics, num_samples=100)
    
    # 轉換為訓練格式
    formatted_dataset = []
    for item in dataset:
        text = f"### 問題：{item['instruction']}\n### 回答：{item['output']}"
        formatted_dataset.append({"text": text})
    
    # 保存資料集
    preparer.save_dataset(formatted_dataset, "ptt_gossiping_dataset_100.jsonl", format="jsonl")
    
    return formatted_dataset


def main():
    """主函數 - 使用範例"""
    
    print("=== 資料集準備工具範例 ===\n")
    
    # 創建資料集準備器
    preparer = DatasetPreparer()
    
    # 範例1：創建指令資料集
    print("1. 創建指令資料集")
    qa_pairs = [
        {"question": "什麼是深度學習？", "answer": "深度學習是機器學習的分支..."},
        {"question": "如何學習程式設計？", "answer": "學習程式設計需要..."}
    ]
    instruction_dataset = preparer.create_instruction_dataset(qa_pairs)
    print(f"   創建了 {len(instruction_dataset)} 個指令樣本\n")
    
    # 範例2：創建對話資料集
    print("2. 創建對話資料集")
    conversations = [
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什麼可以幫助你的嗎？"}
        ]
    ]
    conversation_dataset = preparer.create_conversation_format(conversations)
    print(f"   創建了 {len(conversation_dataset)} 個對話樣本\n")
    
    # 範例3：創建 PTT 風格資料集
    print("3. 創建 PTT 風格資料集")
    ptt_dataset = create_sample_ptt_dataset()
    print(f"   創建了 {len(ptt_dataset)} 個 PTT 風格樣本")
    print(f"   範例：{ptt_dataset[0]['text'][:100]}...\n")
    
    # 範例4：分割資料集
    print("4. 分割資料集")
    train_data, val_data = preparer.split_dataset(ptt_dataset, train_ratio=0.8)
    print(f"   訓練集：{len(train_data)} 個樣本")
    print(f"   驗證集：{len(val_data)} 個樣本")


if __name__ == "__main__":
    main()