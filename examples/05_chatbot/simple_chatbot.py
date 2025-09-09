"""
不使用框架的簡單聊天機器人實作
展示聊天機器人的基礎架構和對話管理
"""

from typing import List, Dict, Optional
import json
from datetime import datetime


class SimpleChatbot:
    """簡單聊天機器人類"""
    
    def __init__(self, model, tokenizer, max_history: int = 10):
        """
        初始化聊天機器人
        
        Args:
            model: 語言模型
            tokenizer: 分詞器
            max_history: 最大保留的對話歷史數
        """
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []
        self.max_history = max_history
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_message(self, role: str, content: str):
        """
        添加訊息到對話歷史
        
        Args:
            role: 角色（user/assistant/system）
            content: 訊息內容
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(message)
        
        # 限制歷史長度
        if len(self.conversation_history) > self.max_history:
            # 保留系統訊息和最近的對話
            system_messages = [m for m in self.conversation_history if m["role"] == "system"]
            other_messages = [m for m in self.conversation_history if m["role"] != "system"]
            self.conversation_history = system_messages + other_messages[-(self.max_history - len(system_messages)):]
    
    def generate_response(self, user_input: str) -> str:
        """
        生成回應
        
        Args:
            user_input: 用戶輸入
            
        Returns:
            AI 回應
        """
        # 添加用戶輸入到歷史
        self.add_message("user", user_input)
        
        # 構建 prompt
        prompt = self._build_prompt()
        
        # 生成回應
        response = self._generate(prompt)
        
        # 添加回應到歷史
        self.add_message("assistant", response)
        
        return response
    
    def _build_prompt(self) -> str:
        """
        將對話歷史轉換為模型輸入格式
        
        Returns:
            格式化的 prompt
        """
        prompt = ""
        for msg in self.conversation_history:
            if msg["role"] == "system":
                prompt += f"系統：{msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"用戶：{msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"助手：{msg['content']}\n"
        
        # 如果最後一條不是助手訊息，添加助手標記
        if self.conversation_history and self.conversation_history[-1]["role"] != "assistant":
            prompt += "助手："
        
        return prompt
    
    def _generate(self, prompt: str, max_length: int = 500) -> str:
        """
        使用模型生成文本（模擬）
        
        Args:
            prompt: 輸入 prompt
            max_length: 最大生成長度
            
        Returns:
            生成的文本
        """
        # 這裡是模擬生成，實際使用時會調用真實的模型
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs, max_length=max_length)
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 模擬回應
        if "你好" in prompt:
            return "你好！我是AI助手，有什麼可以幫助你的嗎？"
        elif "天氣" in prompt:
            return "抱歉，我無法獲取即時天氣資訊，建議查看天氣預報網站。"
        else:
            return "我理解你的問題。這是一個有趣的話題，讓我為你提供一些資訊..."
    
    def set_system_message(self, message: str):
        """
        設置系統訊息
        
        Args:
            message: 系統訊息內容
        """
        # 移除舊的系統訊息
        self.conversation_history = [
            msg for msg in self.conversation_history 
            if msg["role"] != "system"
        ]
        # 添加新的系統訊息
        self.add_message("system", message)
    
    def reset_conversation(self):
        """重置對話歷史"""
        self.conversation_history = []
        print("對話已重置")
    
    def save_conversation(self, filepath: str = None):
        """
        保存對話歷史
        
        Args:
            filepath: 保存路徑
        """
        if filepath is None:
            filepath = f"conversation_{self.session_id}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        
        print(f"對話已保存到：{filepath}")
    
    def load_conversation(self, filepath: str):
        """
        載入對話歷史
        
        Args:
            filepath: 檔案路徑
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.conversation_history = json.load(f)
        
        print(f"已載入對話歷史：{len(self.conversation_history)} 條訊息")
    
    def get_conversation_summary(self) -> Dict:
        """
        獲取對話摘要
        
        Returns:
            對話統計資訊
        """
        user_messages = [m for m in self.conversation_history if m["role"] == "user"]
        assistant_messages = [m for m in self.conversation_history if m["role"] == "assistant"]
        
        return {
            "session_id": self.session_id,
            "total_messages": len(self.conversation_history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "start_time": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
            "last_message_time": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }


class ChatbotWithMemory(SimpleChatbot):
    """帶記憶功能的聊天機器人"""
    
    def __init__(self, model, tokenizer, max_history: int = 10):
        super().__init__(model, tokenizer, max_history)
        self.long_term_memory = {}  # 長期記憶
        self.user_preferences = {}  # 用戶偏好
    
    def remember(self, key: str, value: str):
        """
        記住資訊
        
        Args:
            key: 記憶鍵
            value: 記憶值
        """
        self.long_term_memory[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
    
    def recall(self, key: str) -> Optional[str]:
        """
        回憶資訊
        
        Args:
            key: 記憶鍵
            
        Returns:
            記憶的值，如果不存在返回 None
        """
        memory = self.long_term_memory.get(key)
        return memory["value"] if memory else None
    
    def update_user_preference(self, preference: str, value: str):
        """
        更新用戶偏好
        
        Args:
            preference: 偏好類型
            value: 偏好值
        """
        self.user_preferences[preference] = value
    
    def _build_prompt(self) -> str:
        """
        構建帶記憶的 prompt
        
        Returns:
            格式化的 prompt
        """
        prompt = super()._build_prompt()
        
        # 添加相關記憶
        if self.long_term_memory:
            relevant_memories = self._get_relevant_memories()
            if relevant_memories:
                prompt = f"相關記憶：{relevant_memories}\n\n" + prompt
        
        return prompt
    
    def _get_relevant_memories(self) -> str:
        """
        獲取相關記憶（簡化版）
        
        Returns:
            相關記憶的字符串
        """
        # 這裡可以實作更複雜的記憶檢索邏輯
        memories = []
        for key, value in list(self.long_term_memory.items())[:3]:  # 只取最近3條
            memories.append(f"{key}: {value['value']}")
        
        return "; ".join(memories) if memories else ""


def main():
    """主函數 - 使用範例"""
    
    print("=== 簡單聊天機器人範例 ===\n")
    
    # 模擬初始化（實際使用需要真實的模型和tokenizer）
    model = None  # 實際使用時替換為真實模型
    tokenizer = None  # 實際使用時替換為真實tokenizer
    
    # 創建聊天機器人
    chatbot = SimpleChatbot(model, tokenizer)
    
    # 設置系統訊息
    chatbot.set_system_message("你是一個友善且專業的AI助手")
    
    # 模擬對話
    print("開始對話（輸入 'quit' 結束）\n")
    
    test_inputs = [
        "你好",
        "今天天氣如何？",
        "給我講個笑話"
    ]
    
    for user_input in test_inputs:
        print(f"用戶：{user_input}")
        response = chatbot.generate_response(user_input)
        print(f"助手：{response}\n")
    
    # 顯示對話摘要
    summary = chatbot.get_conversation_summary()
    print("\n對話摘要：")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 保存對話
    chatbot.save_conversation()
    
    print("\n=== 帶記憶功能的聊天機器人 ===\n")
    
    # 創建帶記憶的聊天機器人
    memory_bot = ChatbotWithMemory(model, tokenizer)
    
    # 記住一些資訊
    memory_bot.remember("用戶名", "小明")
    memory_bot.remember("喜好", "喜歡科技話題")
    memory_bot.update_user_preference("語言", "中文")
    
    # 回憶資訊
    username = memory_bot.recall("用戶名")
    print(f"記住的用戶名：{username}")


if __name__ == "__main__":
    main()