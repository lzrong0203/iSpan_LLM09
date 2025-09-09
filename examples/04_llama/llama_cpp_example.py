"""
使用 llama.cpp 部署量化模型
更省資源的 Llama 模型部署方式
"""

from llama_cpp import Llama
import json


class LlamaCppModel:
    """llama.cpp 模型封裝"""
    
    def __init__(self, 
                 model_path: str = "./models/llama-3-8b.Q4_K_M.gguf",
                 n_ctx: int = 2048,
                 n_threads: int = 8,
                 n_gpu_layers: int = 0):
        """
        初始化 llama.cpp 模型
        
        Args:
            model_path: GGUF 格式模型檔案路徑
            n_ctx: 上下文長度
            n_threads: CPU 執行緒數
            n_gpu_layers: 要放到 GPU 的層數（0 表示純 CPU）
        """
        print(f"載入量化模型：{model_path}")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        
        print("模型載入完成！")
    
    def generate(self, 
                prompt: str,
                max_tokens: int = 256,
                temperature: float = 0.7,
                top_p: float = 0.9,
                stop: list = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 輸入提示
            max_tokens: 最大生成 token 數
            temperature: 溫度參數
            top_p: nucleus sampling 參數
            stop: 停止序列
            
        Returns:
            生成的文本
        """
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            echo=False
        )
        
        return output['choices'][0]['text']
    
    def chat_completion(self, messages: list, **kwargs) -> str:
        """
        聊天完成介面（類似 OpenAI API）
        
        Args:
            messages: 訊息列表
            **kwargs: 其他生成參數
            
        Returns:
            模型回應
        """
        # 將訊息列表轉換為 prompt
        prompt = self._format_messages(messages)
        
        # 生成回應
        response = self.generate(prompt, **kwargs)
        
        return response
    
    def _format_messages(self, messages: list) -> str:
        """
        格式化訊息列表為 prompt
        
        Args:
            messages: 訊息列表
            
        Returns:
            格式化的 prompt
        """
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        # 添加最後的 Assistant 標記
        if messages and messages[-1]["role"] != "assistant":
            prompt += "Assistant: "
        
        return prompt
    
    def save_conversation(self, messages: list, filepath: str):
        """
        保存對話歷史
        
        Args:
            messages: 訊息列表
            filepath: 保存路徑
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    
    def load_conversation(self, filepath: str) -> list:
        """
        載入對話歷史
        
        Args:
            filepath: 檔案路徑
            
        Returns:
            訊息列表
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


def download_model_example():
    """展示如何下載量化模型"""
    print("""
    下載量化模型的步驟：
    
    1. 訪問 Hugging Face 模型頁面
       https://huggingface.co/TheBloke
    
    2. 選擇合適的量化版本
       - Q4_K_M: 平衡品質和大小
       - Q5_K_M: 更好的品質，稍大
       - Q8_0: 接近原始品質，較大
    
    3. 下載 GGUF 格式檔案
       wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf
    
    4. 放置到 models 目錄
       mkdir -p models
       mv llama-2-7b.Q4_K_M.gguf models/
    """)


def main():
    """主函數 - 使用範例"""
    
    print("=== llama.cpp 使用範例 ===\n")
    
    # 顯示下載說明
    download_model_example()
    
    print("\n範例程式碼（需要先下載模型）：")
    print("-" * 50)
    
    # 模擬程式碼
    print("""
    # 初始化模型
    model = LlamaCppModel(
        model_path="./models/llama-3-8b.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=8
    )
    
    # 簡單生成
    prompt = "Q: 什麼是人工智慧? A:"
    response = model.generate(prompt, max_tokens=256)
    print(response)
    
    # 聊天介面
    messages = [
        {"role": "system", "content": "你是一個有幫助的助手"},
        {"role": "user", "content": "解釋什麼是機器學習"}
    ]
    response = model.chat_completion(messages)
    print(response)
    """)
    
    print("\n優勢：")
    print("- 記憶體使用量大幅減少（4-bit 量化可減少 75%）")
    print("- CPU 也能運行（雖然較慢）")
    print("- 支援各種量化格式")
    print("- 易於部署和分發")


if __name__ == "__main__":
    # 安裝 llama-cpp-python:
    # pip install llama-cpp-python
    
    main()