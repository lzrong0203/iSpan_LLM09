"""
OpenAI API 基礎聊天範例
展示如何使用 OpenAI API 進行對話
"""

import openai
import os
from typing import List, Dict


class OpenAIChatBot:
    """OpenAI 聊天機器人封裝"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """
        初始化 OpenAI 聊天機器人
        
        Args:
            api_key: OpenAI API Key，如果不提供會從環境變數讀取
            model: 使用的模型名稱
        """
        # 設置API Key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("請設置 OPENAI_API_KEY 環境變數或提供 API Key")
        
        openai.api_key = self.api_key
        self.model = model
        self.conversation_history = []
    
    def chat_with_gpt(self, prompt: str, system_message: str = "你是一個專業的AI助手") -> str:
        """
        基本對話調用
        
        Args:
            prompt: 用戶輸入
            system_message: 系統訊息，定義助手的角色
            
        Returns:
            AI 的回應
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"錯誤：{str(e)}"
    
    def chat_with_context(self, user_input: str) -> str:
        """
        帶上下文的對話
        
        Args:
            user_input: 用戶輸入
            
        Returns:
            AI 的回應
        """
        # 添加用戶輸入到歷史
        self.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # 調用 API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=1000
            )
            
            # 獲取回應
            ai_response = response.choices[0].message.content
            
            # 添加到歷史
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
        except Exception as e:
            return f"錯誤：{str(e)}"
    
    def reset_conversation(self):
        """重置對話歷史"""
        self.conversation_history = []
    
    def set_system_message(self, message: str):
        """設置系統訊息"""
        # 移除舊的系統訊息
        self.conversation_history = [
            msg for msg in self.conversation_history 
            if msg.get("role") != "system"
        ]
        # 添加新的系統訊息到開頭
        self.conversation_history.insert(0, {"role": "system", "content": message})


def main():
    """主函數 - 使用範例"""
    
    # 初始化聊天機器人
    chatbot = OpenAIChatBot(model="gpt-3.5-turbo")  # 使用較便宜的模型做測試
    
    # 範例1：單次對話
    print("=== 單次對話範例 ===")
    result = chatbot.chat_with_gpt("解釋什麼是機器學習")
    print(f"回應：{result}\n")
    
    # 範例2：帶上下文的對話
    print("=== 帶上下文的對話範例 ===")
    chatbot.set_system_message("你是一個Python程式設計專家")
    
    questions = [
        "什麼是列表推導式？",
        "給我一個實際的例子",
        "它和普通的for循環相比有什麼優勢？"
    ]
    
    for question in questions:
        print(f"問：{question}")
        response = chatbot.chat_with_context(question)
        print(f"答：{response}\n")


if __name__ == "__main__":
    # 注意：運行前請確保已設置 OPENAI_API_KEY 環境變數
    # export OPENAI_API_KEY="your-api-key-here"
    main()