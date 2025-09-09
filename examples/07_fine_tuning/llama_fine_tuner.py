"""
Llama 模型微調工具
使用 Transformers 和 PEFT 進行高效微調
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
from typing import Dict, List


class LlamaFineTuner:
    """Llama 模型微調器"""
    
    def __init__(self, model_name: str, output_dir: str):
        """
        初始化微調器
        
        Args:
            model_name: 基礎模型名稱
            output_dir: 輸出目錄
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        
    def load_model(self, load_in_8bit: bool = False, load_in_4bit: bool = False):
        """
        載入預訓練模型
        
        Args:
            load_in_8bit: 是否使用 8-bit 量化
            load_in_4bit: 是否使用 4-bit 量化
        """
        print(f"載入模型：{self.model_name}")
        
        # 載入 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 設定量化配置
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        # 載入模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            load_in_8bit=load_in_8bit if not load_in_4bit else False,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 添加 padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        
        print("模型載入完成")
        
    def prepare_dataset(self, data: List[Dict], max_length: int = 512):
        """
        準備訓練資料集
        
        Args:
            data: 訓練資料
            max_length: 最大序列長度
            
        Returns:
            處理後的資料集
        """
        def tokenize_function(examples):
            # 如果資料有 'text' 欄位
            if 'text' in examples:
                texts = examples['text']
            # 如果資料有 'instruction' 和 'output' 欄位
            elif 'instruction' in examples and 'output' in examples:
                texts = [
                    f"### 指令：{inst}\n### 回答：{out}"
                    for inst, out in zip(examples['instruction'], examples['output'])
                ]
            else:
                raise ValueError("資料格式不支援")
            
            return self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
        
        # 創建 Dataset
        dataset = Dataset.from_list(data)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def setup_training_args(self, 
                           num_epochs: int = 3,
                           batch_size: int = 4,
                           learning_rate: float = 2e-4,
                           warmup_steps: int = 100,
                           logging_steps: int = 10,
                           save_steps: int = 500,
                           gradient_accumulation_steps: int = 4):
        """
        設置訓練參數
        
        Args:
            num_epochs: 訓練輪數
            batch_size: 批次大小
            learning_rate: 學習率
            warmup_steps: 預熱步數
            logging_steps: 記錄間隔
            save_steps: 保存間隔
            gradient_accumulation_steps: 梯度累積步數
            
        Returns:
            訓練參數
        """
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps" if save_steps > 0 else "no",
            eval_steps=save_steps,
            save_total_limit=2,
            load_best_model_at_end=True if save_steps > 0 else False,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            learning_rate=learning_rate,
        )
    
    def train(self, train_dataset, eval_dataset=None, training_args=None):
        """
        執行訓練
        
        Args:
            train_dataset: 訓練資料集
            eval_dataset: 驗證資料集
            training_args: 訓練參數
        """
        if training_args is None:
            training_args = self.setup_training_args()
        
        # 創建 Trainer
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
        print("開始訓練...")
        trainer.train()
        
        # 保存模型
        print(f"保存模型到：{self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer
    
    def generate(self, prompt: str, max_length: int = 200, temperature: float = 0.7):
        """
        生成文本
        
        Args:
            prompt: 輸入提示
            max_length: 最大生成長度
            temperature: 溫度參數
            
        Returns:
            生成的文本
        """
        # Tokenize 輸入
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        # 移動到設備
        if self.model.device.type != 'cpu':
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解碼
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def evaluate_model(self, test_prompts: List[str]):
        """
        評估模型
        
        Args:
            test_prompts: 測試提示列表
        """
        print("\n=== 模型評估 ===")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n測試 {i}:")
            print(f"輸入：{prompt}")
            
            response = self.generate(prompt)
            print(f"輸出：{response}")
            print("-" * 50)


class LoRAFineTuner(LlamaFineTuner):
    """使用 LoRA 的微調器"""
    
    def __init__(self, model_name: str, output_dir: str):
        super().__init__(model_name, output_dir)
        self.peft_config = None
    
    def setup_lora(self, 
                   r: int = 16,
                   lora_alpha: int = 32,
                   lora_dropout: float = 0.1,
                   target_modules: List[str] = None):
        """
        設置 LoRA 配置
        
        Args:
            r: LoRA 秩
            lora_alpha: LoRA 縮放參數
            lora_dropout: Dropout 率
            target_modules: 目標模組
        """
        from peft import LoraConfig, get_peft_model, TaskType
        
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        
        # 應用 LoRA
        if self.model is not None:
            self.model = get_peft_model(self.model, self.peft_config)
            self.model.print_trainable_parameters()
    
    def save_lora_adapter(self, adapter_path: str):
        """
        保存 LoRA 適配器
        
        Args:
            adapter_path: 適配器保存路徑
        """
        if self.model is not None:
            self.model.save_pretrained(adapter_path)
            print(f"LoRA 適配器已保存到：{adapter_path}")
    
    def load_with_lora(self, base_model_path: str, adapter_path: str):
        """
        載入帶 LoRA 適配器的模型
        
        Args:
            base_model_path: 基礎模型路徑
            adapter_path: LoRA 適配器路徑
        """
        from peft import PeftModel
        
        # 載入基礎模型
        self.load_model()
        
        # 載入 LoRA 適配器
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
        print(f"已載入 LoRA 適配器：{adapter_path}")


def main():
    """主函數 - 使用範例"""
    
    print("=== 微調工具範例 ===\n")
    
    # 模擬資料
    sample_data = [
        {"text": "### 問題：什麼是機器學習？\n### 回答：機器學習是一種人工智慧技術..."},
        {"text": "### 問題：如何學習Python？\n### 回答：學習Python可以從基礎語法開始..."},
    ]
    
    # 使用範例（需要實際的模型和資料）
    print("微調流程：")
    print("1. 初始化微調器")
    print("   tuner = LlamaFineTuner('model_name', './output')")
    print("\n2. 載入模型")
    print("   tuner.load_model(load_in_4bit=True)")
    print("\n3. 準備資料集")
    print("   dataset = tuner.prepare_dataset(data)")
    print("\n4. 設置訓練參數")
    print("   args = tuner.setup_training_args()")
    print("\n5. 開始訓練")
    print("   tuner.train(dataset)")
    print("\n6. 評估模型")
    print("   tuner.evaluate_model(test_prompts)")
    
    print("\n=== LoRA 微調範例 ===\n")
    
    print("LoRA 微調流程：")
    print("1. 初始化 LoRA 微調器")
    print("   lora_tuner = LoRAFineTuner('model_name', './output')")
    print("\n2. 載入模型並設置 LoRA")
    print("   lora_tuner.load_model(load_in_4bit=True)")
    print("   lora_tuner.setup_lora(r=16, lora_alpha=32)")
    print("\n3. 訓練")
    print("   lora_tuner.train(dataset)")
    print("\n4. 保存 LoRA 適配器")
    print("   lora_tuner.save_lora_adapter('./lora_adapter')")


if __name__ == "__main__":
    main()