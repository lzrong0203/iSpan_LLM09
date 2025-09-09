"""
Lab 4: 使用 LoRA 微調 Llama 模型 - PTT 鄉民風格版本
本程式展示如何使用 LoRA 技術微調 Llama 3 模型，讓它學會 PTT 鄉民的說話風格

作者：LLM 實作應用班
日期：2025-09-13
"""

import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 1. 設定區 - 可調整的參數
# ===========================

# 基礎模型 ID (使用 Llama 3 8B Instruct 版本)
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# 資料集檔案路徑
DATASET_FILE = "ptt_gossiping_dataset_100.json"  # 或 .jsonl

# 輸出路徑
OUTPUT_DIR = "./results_llama3_ptt"
ADAPTER_PATH = "./lora_adapter_llama3_ptt"

# 訓練參數 (可根據你的 GPU 調整)
TRAIN_CONFIG = {
    "num_train_epochs": 3,           # 訓練輪數
    "per_device_train_batch_size": 2,  # 批次大小 (GPU記憶體不夠可改為1)
    "gradient_accumulation_steps": 2,  # 梯度累積 (有效批次大小 = 2*2=4)
    "learning_rate": 2e-4,            # 學習率
    "max_seq_length": 1024,           # 最大序列長度
}

# LoRA 參數
LORA_CONFIG = {
    "r": 64,                # LoRA 秩 (rank)，越高越精確但越耗資源
    "lora_alpha": 16,       # LoRA 縮放因子
    "lora_dropout": 0.1,    # Dropout 防止過擬合
}

# ===========================
# 2. 環境檢查
# ===========================

def check_environment():
    """檢查執行環境"""
    print("=" * 60)
    print("環境檢查")
    print("=" * 60)
    
    # 檢查 CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用")
        print(f"  GPU 名稱: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("✗ CUDA 不可用，將使用 CPU (訓練會很慢)")
    
    # 檢查資料集檔案
    if os.path.exists(DATASET_FILE):
        print(f"✓ 找到資料集檔案: {DATASET_FILE}")
        # 檢查資料集內容
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            if DATASET_FILE.endswith('.jsonl'):
                data = [json.loads(line) for line in f.readlines()[:5]]
            else:
                data = json.load(f)
                if isinstance(data, list):
                    data = data[:5]
        print(f"  資料集樣本數: {len(data)} (顯示前5筆)")
    else:
        print(f"✗ 找不到資料集檔案: {DATASET_FILE}")
        print("  請確認檔案路徑正確")
        return False
    
    print("=" * 60)
    return True

# ===========================
# 3. 資料集處理
# ===========================

def prepare_dataset():
    """載入並準備資料集"""
    print("\n載入資料集...")
    
    # 載入資料集
    if DATASET_FILE.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=DATASET_FILE, split='train')
    else:
        dataset = load_dataset('json', data_files=DATASET_FILE, split='train')
    
    print(f"資料集大小: {len(dataset)} 筆")
    
    # 檢查資料格式
    sample = dataset[0]
    print(f"資料格式範例:")
    for key in sample.keys():
        if isinstance(sample[key], str):
            print(f"  {key}: {sample[key][:50]}...")
        else:
            print(f"  {key}: {sample[key]}")
    
    return dataset

# ===========================
# 4. 模型設定與載入
# ===========================

def setup_model_and_tokenizer():
    """設定並載入模型和 tokenizer"""
    print("\n設定 4-bit 量化配置...")
    
    # 4-bit 量化設定 (大幅減少記憶體使用)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("載入基礎模型...")
    # 載入模型
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    print("載入 Tokenizer...")
    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # 設定 padding token (Llama 3 需要)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 啟用梯度檢查點以節省記憶體
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

# ===========================
# 5. LoRA 設定
# ===========================

def setup_lora(model):
    """設定 LoRA 參數"""
    print("\n設定 LoRA 參數...")
    
    # LoRA 配置
    peft_config = LoraConfig(
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        r=LORA_CONFIG["r"],
        bias="none",
        task_type="CAUSAL_LM",
        # Llama 3 的目標模組
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    
    # 應用 LoRA 到模型
    model = get_peft_model(model, peft_config)
    
    # 顯示可訓練參數統計
    model.print_trainable_parameters()
    
    return model, peft_config

# ===========================
# 6. 訓練設定
# ===========================

def setup_training_args():
    """設定訓練參數"""
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
        optim="paged_adamw_8bit",  # 8-bit AdamW 節省記憶體
        save_steps=50,
        logging_steps=10,
        learning_rate=TRAIN_CONFIG["learning_rate"],
        weight_decay=0.001,
        fp16=False,
        bf16=True,  # 使用 bf16 (如果 GPU 支援)
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to=["tensorboard"],  # 可以用 tensorboard 查看訓練過程
        seed=42,
    )

# ===========================
# 7. 訓練函數
# ===========================

def train_model(model, tokenizer, dataset, peft_config):
    """執行訓練"""
    print("\n開始訓練...")
    print("=" * 60)
    
    # 設定訓練參數
    training_args = setup_training_args()
    
    # 建立 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",  # 資料集中包含文本的欄位
        max_seq_length=TRAIN_CONFIG["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    
    # 開始訓練
    trainer.train()
    
    # 保存 LoRA 適配器
    print(f"\n保存 LoRA 適配器到: {ADAPTER_PATH}")
    trainer.model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    
    print("訓練完成！")
    return trainer

# ===========================
# 8. 推論與比較
# ===========================

def generate_response(model, tokenizer, prompt, system_prompt=None):
    """生成模型回應"""
    
    # 建立對話格式 (Llama 3 格式)
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt}
        ]
    
    # 使用 tokenizer 的 chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解碼
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取助手回應
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    return response

def compare_models():
    """比較原始模型和微調後模型"""
    print("\n" + "=" * 60)
    print("模型比較測試")
    print("=" * 60)
    
    # 測試問題
    test_prompts = [
        "為什麼台灣的夏天這麼熱？",
        "該不該買房？",
        "加密貨幣是不是詐騙？",
    ]
    
    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    
    # 系統提示
    system_prompt = "你是一個資深的 PTT 鄉民，請用八卦版的風格回答問題。"
    
    for prompt in test_prompts:
        print(f"\n問題: {prompt}")
        print("-" * 40)
        
        # 原始模型回答 (模擬，實際需要重新載入)
        print("原始 Llama 3 (標準回答):")
        print("  [需要載入原始模型才能顯示，通常會是正經的百科全書式回答]")
        
        # 微調後模型回答
        print("\nPTT 鄉民版 (微調後):")
        # 這裡需要實際載入微調後的模型
        print("  [需要載入微調後模型，會顯示鄉民風格的回答]")
        print("  例如：笑死，你是不是都不出門？台灣就熱帶海島氣候啊...")

# ===========================
# 9. 主程式
# ===========================

def main():
    """主程式"""
    print("\n" + "=" * 60)
    print("Lab 4: 使用 LoRA 微調 Llama 3 - PTT 鄉民風格")
    print("=" * 60)
    
    # 步驟 1: 環境檢查
    if not check_environment():
        print("\n環境檢查失敗，請修正問題後重試")
        return
    
    try:
        # 步驟 2: 準備資料集
        dataset = prepare_dataset()
        
        # 步驟 3: 載入模型和 tokenizer
        model, tokenizer = setup_model_and_tokenizer()
        
        # 步驟 4: 設定 LoRA
        model, peft_config = setup_lora(model)
        
        # 步驟 5: 訓練模型
        trainer = train_model(model, tokenizer, dataset, peft_config)
        
        # 步驟 6: 測試與比較
        compare_models()
        
        print("\n" + "=" * 60)
        print("訓練完成！")
        print(f"LoRA 適配器已保存至: {ADAPTER_PATH}")
        print(f"訓練紀錄已保存至: {OUTPUT_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n錯誤發生: {str(e)}")
        print("請檢查錯誤訊息並重試")
        import traceback
        traceback.print_exc()

# ===========================
# 10. 單獨測試微調後模型
# ===========================

def test_finetuned_model():
    """單獨測試微調後的模型"""
    print("\n載入微調後的模型進行測試...")
    
    # 載入基礎模型
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # 載入 LoRA 適配器
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    
    # 測試生成
    while True:
        prompt = input("\n輸入問題 (輸入 'quit' 結束): ")
        if prompt.lower() == 'quit':
            break
        
        response = generate_response(
            model, 
            tokenizer, 
            prompt,
            system_prompt="你是一個資深的 PTT 鄉民，請用八卦版的風格回答問題。"
        )
        
        print(f"\n鄉民回答: {response}")

# ===========================
# 執行點
# ===========================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 執行測試模式: python lab4_finetune_lora.py test
        test_finetuned_model()
    else:
        # 執行訓練模式: python lab4_finetune_lora.py
        main()

"""
使用說明：
1. 安裝必要套件：
   pip install transformers torch accelerate bitsandbytes peft trl datasets tensorboard

2. 準備資料集：
   確保 ptt_gossiping_dataset_100.json 在同一目錄下

3. 執行訓練：
   python lab4_finetune_lora.py

4. 測試微調後的模型：
   python lab4_finetune_lora.py test

注意事項：
- 需要至少 8GB VRAM 的 GPU
- 首次執行會下載 Llama 3 模型 (約 16GB)
- 訓練時間依 GPU 而定，約 30-60 分鐘
- 可調整 TRAIN_CONFIG 中的參數來適應你的硬體
"""