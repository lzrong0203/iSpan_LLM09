from dotenv import load_dotenv
import os

load_dotenv()
from openai import OpenAI

client = OpenAI()
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[{"role": "system", "content": "你是一個專業的程式設計師，並且善於教導學生"}, 
#               {"role": "user", "content": "Hello, what is Python list?"}]
# )

class Agent:

  def __init__(self, system=""):
    """
    Agent 初始化，建立系統的訊息
    """
    self.system = system
    self.messages = []
    if self.system:
      self.messages.append({"role": "system", "content": system})

  def execute(self):
    """
    建立 chatGPT的聊天，並回傳AI的回覆
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=self.messages,
        temperature=0)
    return completion.choices[0].message.content

  def __call__(self, message):
    self.messages.append({"role": "user", "content": message})
    result = self.execute()
    self.messages.append({"role": "assistant", "content": result})
    return result



system_prompt = "你是一個專業的程式設計師，並且善於教導學生"

agent = Agent(system_prompt)
# print(agent.messages)
print("=============")
print(agent("Hello, what is Python list?"))
print("=============")
# print(agent.messages)
print(agent("我剛剛問了你什麼?"))