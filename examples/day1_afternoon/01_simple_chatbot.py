# Day 1 下午：簡單聊天機器人
# 13:30-14:30 Chatbot基礎開發

import json
import os
from datetime import datetime

from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# === Step 1: 最簡單的Chatbot ===
print("=== 第一個Chatbot ===")
print()


def simple_chatbot():
    """
    規則型的簡單聊天機器人
    """
    print("🤖 簡單聊天機器人")
    print("輸入'quit'離開")
    print("-" * 40)

    # 預設回應
    responses = {
        "你好": "你好！我是AI助手",
        "你是誰": "我是一個簡單的聊天機器人",
        "天氣": "今天天氣不錯喔！",
        "再見": "再見！祝你有美好的一天！",
    }

    while True:
        user_input = input("👤 你：")

        if user_input.lower() == "quit":
            print("🤖 AI：再見！")
            break

        # 尋找匹配的回應
        found = False
        for keyword, response in responses.items():
            if keyword in user_input:
                print(f"🤖 AI：{response}")
                found = True
                break

        if not found:
            print(f"🤖 AI：你說的是'{user_input}'，我還在學習中！")
        print()


# === Step 2: 加入記憶功能 ===
class MemoryChatbot:
    """
    有記憶功能的Chatbot
    """

    def __init__(self):
        self.conversation_history = []
        self.user_name = None
        self.user_preferences = {}

    def remember_user(self, message):
        """記住使用者資訊"""
        if "我叫" in message or "我是" in message:
            # 提取名字
            if "我叫" in message:
                name = message.split("我叫")[-1].strip()
            else:
                name = message.split("我是")[-1].strip()
            self.user_name = name
            return f"很高興認識你，{name}！"

        if "我喜歡" in message:
            # 記住喜好
            like = message.split("我喜歡")[-1].strip()
            self.user_preferences["likes"] = self.user_preferences.get("likes", [])
            self.user_preferences["likes"].append(like)
            return f"我記住了，你喜歡{like}！"

        return None

    def chat(self):
        print("🤖 有記憶的聊天機器人")
        print("我會記住我們的對話！")
        print("-" * 40)

        while True:
            user_input = input("👤 你：")

            if user_input.lower() == "quit":
                self.save_conversation()
                print("🤖 AI：再見！對話已儲存。")
                break

            # 加入歷史
            self.conversation_history.append(
                {"time": datetime.now().strftime("%H:%M:%S"), "user": user_input}
            )

            # 記憶功能
            memory_response = self.remember_user(user_input)
            if memory_response:
                response = memory_response
            elif "你記得" in user_input:
                if self.user_name:
                    response = f"當然記得！你是{self.user_name}"
                    if self.user_preferences.get("likes"):
                        likes = "、".join(self.user_preferences["likes"])
                        response += f"，你喜歡{likes}"
                else:
                    response = "我還不知道你的名字呢！"
            elif "我們說了什麼" in user_input:
                if len(self.conversation_history) > 1:
                    recent = self.conversation_history[-2]["user"]
                    response = f"你剛剛說：{recent}"
                else:
                    response = "我們剛開始聊天呢！"
            else:
                # 預設回應
                if self.user_name:
                    response = f"{self.user_name}，我收到了你的訊息！"
                else:
                    response = "我收到了！可以告訴我你的名字嗎？"

            # 加入AI回應到歷史
            self.conversation_history.append(
                {"time": datetime.now().strftime("%H:%M:%S"), "ai": response}
            )

            print(f"🤖 AI：{response}")
            print()

    def save_conversation(self):
        """儲存對話記錄"""
        if self.conversation_history:
            filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "user_name": self.user_name,
                        "preferences": self.user_preferences,
                        "conversation": self.conversation_history,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"對話已儲存到 {filename}")


# === Step 3: 測試互動 ===
print("選擇Chatbot版本：")
print("1. 簡單版（規則型）")
print("2. 記憶版（會記住對話）")
print()

choice = input("請選擇 (1 or 2)：")

if choice == "1":
    simple_chatbot()
elif choice == "2":
    bot = MemoryChatbot()
    bot.chat()
else:
    print("無效選擇，啟動簡單版")
    simple_chatbot()
