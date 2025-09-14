# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText
import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

token = os.environ.get("HUGGING_FACE_TOKEN")
if token:
    login(token=token)
else:
    print("Hugging Face token not found in environment variables.")

processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-4b-it", device="cpu")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))