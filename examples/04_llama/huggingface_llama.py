"""
使用 Hugging Face 部署 Llama 模型
展示如何載入和使用 Llama 模型進行文本生成
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LlamaModel:
    """Llama 模型封裝類"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3-8B"):
        """
        初始化 Llama 模型
        
        Args:
            model_name: Hugging Face 上的模型名稱
        """
        print(f"載入模型：{model_name}")
        
        # 載入 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 載入模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # 使用半精度以節省記憶體
            device_map="auto"  # 自動分配到可用的設備
        )
        
        print("模型載入完成！")
    
    def generate_text(self, 
                     prompt: str, 
                     max_length: int = 100,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     do_sample: bool = True) -> str:
        """
        生成文本
        
        Args:
            prompt: 輸入提示
            max_length: 最大生成長度
            temperature: 溫度參數，控制隨機性
            top_p: nucleus sampling 參數
            do_sample: 是否使用採樣
            
        Returns:
            生成的文本
        """
        # Tokenize 輸入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 移動到正確的設備
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解碼輸出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def chat(self, message: str, system_prompt: str = None) -> str:
        """
        聊天介面
        
        Args:
            message: 用戶訊息
            system_prompt: 系統提示（可選）
            
        Returns:
            模型回應
        """
        # 構建聊天格式的 prompt
        if system_prompt:
            prompt = f"System: {system_prompt}\n\nHuman: {message}\n\nAssistant:"
        else:
            prompt = f"Human: {message}\n\nAssistant:"
        
        # 生成回應
        response = self.generate_text(prompt, max_length=200)
        
        # 提取助手的回應部分
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response


def main():
    """主函數 - 使用範例"""
    
    # 注意：實際運行需要：
    # 1. 足夠的 GPU 記憶體（至少 16GB）
    # 2. Hugging Face 帳號和 Llama 模型的存取權限
    
    print("=== Llama 模型使用範例 ===\n")
    
    # 模擬程式碼（實際運行會需要大量資源）
    print("初始化模型（需要下載和載入，可能需要幾分鐘）...")
    # model = LlamaModel("meta-llama/Llama-3-8B")
    
    print("\n範例1：文本生成")
    prompt = "人工智慧的未來發展將會"
    print(f"提示：{prompt}")
    # response = model.generate_text(prompt)
    # print(f"生成：{response}\n")
    print("生成：[需要實際載入模型才能生成]\n")
    
    print("範例2：對話生成")
    question = "什麼是深度學習？"
    print(f"問題：{question}")
    # answer = model.chat(question)
    # print(f"回答：{answer}")
    print("回答：[需要實際載入模型才能生成]")
    
    print("\n注意：實際運行需要足夠的硬體資源和模型存取權限")


if __name__ == "__main__":
    main()