# Day 2 下午：微調示範
# 14:30-15:30 LoRA微調概念

import json
import os
from typing import Dict, List

# === Step 1: 什麼是微調？ ===
print("=== 微調 (Fine-tuning) 概念 ===")
print()
print("微調 = 讓AI學習你的風格")
print()
print("想像成：")
print("🎨 原始模型 = 會畫畫的藝術家")
print("🖌️ 微調 = 教他你的畫風")
print("🖼️ 結果 = 用你的風格作畫")
print()

# === Step 2: 什麼是LoRA？ ===
print("=== LoRA技術 ===")
print()
print("LoRA = Low-Rank Adaptation")
print()
print("傳統微調 vs LoRA：")
print("❌ 傳統：調整100%參數（很慢、很貴）")
print("✅ LoRA：只調整1-5%參數（快速、便宜）")
print()
print("就像：")
print("傳統 = 重新裝修整間房子")
print("LoRA = 只換窗簾和家具")
print()


# === Step 3: 準備訓練資料 ===
class DatasetPreparer:
    """訓練資料準備器"""

    def __init__(self):
        self.training_data = []

    def add_example(self, instruction: str, input_text: str, output: str):
        """添加訓練範例"""
        self.training_data.append(
            {"instruction": instruction, "input": input_text, "output": output}
        )

    def prepare_ptt_style_data(self):
        """準備PTT風格資料"""
        print("準備PTT鄉民風格資料...")

        # 從 ptt_gossiping_dataset_100.json 載入資料
        dataset_path = "/home/lzrong/iSpan/LLM09/ptt_gossiping_dataset_100.json"

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                # 逐行讀取 JSON 資料
                lines = f.readlines()
                ptt_examples = []

                for line in lines[:10]:  # 使用前10筆資料作為範例
                    data = json.loads(line.strip())
                    text = data["text"]

                    # 解析格式化的文字
                    # 格式：[INST] 指令：... 問題：... [/INST] 回答
                    if "[INST]" in text and "[/INST]" in text:
                        parts = text.split("[/INST]")
                        instruction_part = parts[0].replace("[INST]", "").strip()
                        output = parts[1].replace("</s>", "").replace("<s>", "").strip()

                        # 提取問題
                        if "問題：" in instruction_part:
                            input_text = instruction_part.split("問題：")[1].strip()
                        else:
                            input_text = instruction_part

                        ptt_examples.append({
                            "instruction": "用PTT鄉民的語氣回答",
                            "input": input_text,
                            "output": output
                        })

                # 如果無法載入外部資料，使用預設範例
                if not ptt_examples:
                    raise Exception("無法解析資料")

        except Exception as e:
            print(f"載入外部資料失敗: {e}，使用預設範例")
            # 預設範例作為備用
            ptt_examples = [
                {
                    "instruction": "用PTT鄉民的語氣回答",
                    "input": "最近股市會漲嗎？",
                    "output": "all in就對了啦，輸了再來發文取暖，嘻嘻",
                },
                {
                    "instruction": "用PTT鄉民的語氣回答",
                    "input": "該買房還是租房？",
                    "output": "買不起啦QQ，繼續當社畜存頭期款，唉",
                },
                {
                    "instruction": "用PTT鄉民的語氣回答",
                    "input": "工作壓力好大怎麼辦？",
                    "output": "先推文再說，下班買鹹酥雞壓壓驚，給你拍拍",
                },
                {
                    "instruction": "用PTT鄉民的語氣回答",
                    "input": "要怎麼脫單？",
                    "output": "先減肥啦，然後記得洗澡，認真回",
                },
                {
                    "instruction": "用PTT鄉民的語氣回答",
                    "input": "推薦什麼美食？",
                    "output": "巷口滷肉飯，便宜又大碗，真香",
                },
            ]

        for example in ptt_examples:
            self.add_example(
                example["instruction"], example["input"], example["output"]
            )

        print(f"準備了 {len(ptt_examples)} 個PTT風格範例")

    def prepare_professional_data(self):
        """準備專業風格資料"""
        print("準備專業客服風格資料...")

        professional_examples = [
            {
                "instruction": "用專業客服的語氣回答",
                "input": "產品有問題",
                "output": "非常抱歉給您帶來不便，我們會立即為您處理。請提供訂單編號，我們將優先處理您的案件。",
            },
            {
                "instruction": "用專業客服的語氣回答",
                "input": "要退貨",
                "output": "了解您的需求。根據我們的退貨政策，您可以在收到商品後7天內申請退貨。請問是否方便告知退貨原因？",
            },
            {
                "instruction": "用專業客服的語氣回答",
                "input": "多久到貨？",
                "output": "感謝您的詢問。一般情況下，訂單會在1-3個工作日內送達。您可以透過訂單追蹤系統查看即時狀態。",
            },
        ]

        for example in professional_examples:
            self.add_example(
                example["instruction"], example["input"], example["output"]
            )

        print(f"準備了 {len(professional_examples)} 個專業風格範例")

    def save_dataset(self, filename: str):
        """儲存資料集"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)
        print(f"資料集已儲存到 {filename}")
        print(f"共 {len(self.training_data)} 個訓練範例")

    def load_dataset(self, filename: str):
        """載入資料集"""
        with open(filename, "r", encoding="utf-8") as f:
            self.training_data = json.load(f)
        print(f"從 {filename} 載入了 {len(self.training_data)} 個範例")


# === Step 4: 資料集格式化 ===
def format_for_training(data: List[Dict]) -> List[str]:
    """格式化成訓練格式"""
    formatted = []

    for item in data:
        # Alpaca格式
        prompt = f"""### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Response:
{item['output']}"""
        formatted.append(prompt)

    return formatted


# === Step 5: 模擬微調過程 ===
class LoRATrainer:
    """LoRA訓練器（模擬）"""

    def __init__(self, base_model: str = "llama2"):
        self.base_model = base_model
        self.lora_weights = None
        self.training_history = []

    def train(self, dataset: List[Dict], epochs: int = 3):
        """模擬訓練過程"""
        print(f"\n=== 開始LoRA訓練 ===")
        print(f"基礎模型：{self.base_model}")
        print(f"訓練樣本：{len(dataset)}")
        print(f"訓練輪數：{epochs}")
        print()

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            # 模擬訓練進度
            for i in range(0, 101, 20):
                print(f"  進度：{'█' * (i//5)}{'░' * (20-i//5)} {i}%", end="\r")
                import time

                time.sleep(0.2)

            # 模擬損失下降
            loss = 2.0 / (epoch + 1)
            self.training_history.append({"epoch": epoch + 1, "loss": loss})

            print(f"  進度：{'█' * 20} 100%")
            print(f"  Loss: {loss:.4f}")
            print()

        print("✅ 訓練完成！")
        self.lora_weights = "lora_weights.bin"  # 模擬權重檔案

    def save_adapter(self, path: str):
        """儲存LoRA適配器"""
        print(f"儲存LoRA適配器到 {path}")
        # 實際應用中會儲存權重檔案
        with open(f"{path}/adapter_config.json", "w") as f:
            json.dump(
                {
                    "base_model": self.base_model,
                    "lora_rank": 8,
                    "lora_alpha": 16,
                    "target_modules": ["q_proj", "v_proj"],
                },
                f,
                indent=2,
            )
        print("✅ 適配器已儲存")


# === Step 6: 測試微調效果 ===
class FineTunedModel:
    """微調後的模型（模擬）"""

    def __init__(self, base_model: str, lora_adapter: str = None):
        self.base_model = base_model
        self.lora_adapter = lora_adapter
        self.style = None

        if lora_adapter:
            # 載入LoRA風格
            if "ptt" in lora_adapter.lower():
                self.style = "ptt"
            elif "professional" in lora_adapter.lower():
                self.style = "professional"

    def generate(self, prompt: str) -> str:
        """生成回應"""
        if self.style == "ptt":
            # PTT風格回應
            responses = {
                "股票": "別問，問就是all in台積電",
                "美食": "巷口那家真的讚，吃了會上癮",
                "工作": "慣老闆不意外，拍拍",
                "感情": "先去健身房啦，認真建議",
            }

            for key, response in responses.items():
                if key in prompt:
                    return response
            return "推文先，等等再來認真回"

        elif self.style == "professional":
            # 專業風格回應
            return f"感謝您的詢問。關於「{prompt}」，我們會盡快為您提供協助。"
        else:
            # 原始模型回應
            return f"這是對「{prompt}」的一般回應。"


# === Step 7: 執行示範 ===
print("=== 微調示範 ===")
print()

# 準備資料
preparer = DatasetPreparer()

print("選擇訓練風格：")
print("1. PTT鄉民風格")
print("2. 專業客服風格")
print("3. 兩種都要")
print()

choice = input("選擇 (1-3)：")

if choice == "1":
    preparer.prepare_ptt_style_data()
    dataset_name = "ptt_dataset.json"
elif choice == "2":
    preparer.prepare_professional_data()
    dataset_name = "professional_dataset.json"
else:
    preparer.prepare_ptt_style_data()
    preparer.prepare_professional_data()
    dataset_name = "mixed_dataset.json"

# 儲存資料集
preparer.save_dataset(dataset_name)

# 格式化資料
formatted_data = format_for_training(preparer.training_data)
print(f"\n格式化後的第一個範例：")
print(formatted_data[0])
print()

# 訓練模型
trainer = LoRATrainer()
trainer.train(preparer.training_data, epochs=3)

# 儲存適配器
os.makedirs("lora_adapter", exist_ok=True)
trainer.save_adapter("lora_adapter")

# === Step 8: 測試微調效果 ===
print("\n=== 測試微調效果 ===")
print()

# 原始模型
original_model = FineTunedModel("llama2")

# 微調後模型
if "ptt" in dataset_name:
    finetuned_model = FineTunedModel("llama2", "ptt_lora")
else:
    finetuned_model = FineTunedModel("llama2", "professional_lora")

# 測試問題
test_prompts = ["最近股票怎麼樣？", "推薦好吃的美食", "工作壓力很大"]

for prompt in test_prompts:
    print(f"問題：{prompt}")
    print(f"原始模型：{original_model.generate(prompt)}")
    print(f"微調模型：{finetuned_model.generate(prompt)}")
    print()

# === Step 9: 微調建議 ===
print("=" * 50)
print("=== 微調實務建議 ===")
print()

tips = [
    "📊 資料品質 > 資料數量",
    "🎯 專注特定任務效果更好",
    "⚡ LoRA只需要消費級GPU",
    "💾 LoRA檔案通常只有幾十MB",
    "🔄 可以切換不同的LoRA風格",
    "🧪 建議從小資料集開始實驗",
]

for tip in tips:
    print(tip)

print()
print("💡 微調讓AI變成你的專屬助手！")
