#!/usr/bin/env python3
"""
Gemma 微調腳本 - 使用 PTT 八卦版資料
使用 Transformers + PEFT (LoRA) 進行高效微調
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import os
from typing import Dict, List
from google.colab import drive
from huggingface_hub import login




from google.colab import userdata
userdata.get('HUGGING_FACE_TOKEN')

token = userdata.get('HUGGING_FACE_TOKEN')
if token:
    login(token=token)
else:
    print("Hugging Face token not found in environment variables.")


# Mount Google Drive
drive.mount('/content/drive')


# === 配置參數 ===
MODEL_NAME = "google/gemma-3-1b-it"  
# Update the DATASET_PATH to point to the file in Google Drive
DATASET_PATH = "/content/drive/MyDrive/ptt_gossiping_dataset_100.json"
OUTPUT_DIR = "./gemma-ptt-finetuned"

# LoRA 配置參數
LORA_R = 32  # LoRA rank
LORA_ALPHA = 64  # LoRA alpha
LORA_DROPOUT = 0.05  # LoRA dropout

# 訓練參數
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 10
MAX_LENGTH = 512
WARMUP_STEPS = 100

# === 1. 載入和準備資料 ===
def load_ptt_dataset(file_path: str, max_samples: int = None) -> List[Dict]:
    """載入 PTT 資料集"""
    print(f"正在載入資料集: {file_path}")

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if max_samples and i >= max_samples:
                break

            try:
                item = json.loads(line.strip())
                text = item['text']

                # 解析格式
                if "[INST]" in text and "[/INST]" in text:
                    parts = text.split("[/INST]")
                    instruction_part = parts[0].replace("[INST]", "").strip()
                    response = parts[1].replace("</s>", "").replace("<s>", "").strip()

                    # 提取問題
                    if "問題：" in instruction_part:
                        question = instruction_part.split("問題：")[1].strip()
                    else:
                        question = instruction_part

                    data.append({
                        "instruction": "你是一個資深的 PTT 鄉民，請用八卦版的風格簡潔有力地回答以下問題。",
                        "input": question,
                        "output": response
                    })
            except Exception as e:
                print(f"跳過第 {i+1} 行: {e}")
                continue

    print(f"成功載入 {len(data)} 個訓練樣本")
    return data

def format_prompt(example: Dict) -> str:
    """格式化訓練提示"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    # Gemma 格式
    prompt = f"""<start_of_turn>user
{instruction}
{input_text}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>"""

    return prompt

def prepare_dataset(data: List[Dict], tokenizer) -> Dataset:
    """準備訓練資料集"""

    def tokenize_function(examples):
        # 格式化文本
        texts = []
        for i in range(len(examples["instruction"])):
            text = format_prompt({
                "instruction": examples["instruction"][i],
                "input": examples["input"][i],
                "output": examples["output"][i]
            })
            texts.append(text)

        # Tokenize
        model_inputs = tokenizer(
            texts,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 設定 labels (用於計算 loss)
        model_inputs["labels"] = model_inputs["input_ids"].clone()

        return model_inputs

    # 轉換為 HuggingFace Dataset
    dataset = Dataset.from_list(data)

    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset

# === 2. 模型配置 ===
def setup_model_and_tokenizer():
    """設定模型和 tokenizer"""

    print(f"載入模型: {MODEL_NAME}")

    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4-bit 量化配置 (減少記憶體使用)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 載入模型 (使用 eager attention 以避免警告)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # 使用 eager attention for Gemma3
        use_cache=False,  # 關閉 cache 以相容 gradient checkpointing
    )

    # 準備 k-bit 訓練
    model = prepare_model_for_kbit_training(model)

    # 啟用 gradient checkpointing 以節省記憶體
    model.gradient_checkpointing_enable()

    # LoRA 配置
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 應用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

# === 3. 訓練配置 ===
def setup_training_args():
    """設定訓練參數"""

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        do_eval=False,  # 不進行評估
    )

    return training_args

# === 4. 主程式 ===
def main():
    print("=" * 50)
    print("Gemma PTT 微調腳本")
    print("=" * 50)

    # 檢查 CUDA
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: 未偵測到 GPU，訓練可能會很慢")

    # 載入資料
    print("\n📊 載入 PTT 資料集...")
    ptt_data = load_ptt_dataset(DATASET_PATH, max_samples=100)  # 使用前 50 筆

    # 分割訓練/驗證集
    train_size = int(0.9 * len(ptt_data))
    train_data = ptt_data[:train_size]
    val_data = ptt_data[train_size:]

    print(f"訓練集: {len(train_data)} 筆")
    print(f"驗證集: {len(val_data)} 筆")

    # 設定模型和 tokenizer
    print("\n🤖 載入模型...")
    model, tokenizer = setup_model_and_tokenizer()

    # 準備資料集
    print("\n📝 準備訓練資料...")
    train_dataset = prepare_dataset(train_data, tokenizer)

    # 設定訓練參數
    training_args = setup_training_args()

    # 資料整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 建立 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 開始訓練
    print("\n🚀 開始微調...")
    print(f"總訓練步數: {len(train_dataset) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")

    trainer.train()

    # 儲存模型
    print("\n💾 儲存微調後的模型...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✅ 訓練完成！模型已儲存至: {OUTPUT_DIR}")

    # 測試生成
    print("\n🧪 測試生成...")
    test_prompts = [
        "最近股市怎麼樣？",
        "台北房價會跌嗎？",
        "推薦什麼美食？"
    ]

    model.eval()
    for prompt in test_prompts:
        formatted_prompt = f"""<start_of_turn>user
你是一個資深的 PTT 鄉民，請用八卦版的風格簡潔有力地回答以下問題。
{prompt}<end_of_turn>
<start_of_turn>model"""

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<start_of_turn>model")[-1].strip()

        print(f"\n問題: {prompt}")
        print(f"回答: {response}")

# === 5. 安裝依賴提示 ===
def check_dependencies():
    """檢查依賴"""
    required_packages = [
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "torch"
    ]

    print("請確保已安裝以下套件:")
    print("pip install " + " ".join(required_packages))
    print()

if __name__ == "__main__":
    print("=" * 50)
    print("Gemma PTT 微調腳本")
    print("=" * 50)

    # 檢查依賴
    check_dependencies()

    try:
        main()
    except ImportError as e:
        print(f"\n❌ 缺少必要套件: {e}")
        print("\n請執行以下命令安裝:")
        print("pip install transformers peft bitsandbytes accelerate datasets")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()