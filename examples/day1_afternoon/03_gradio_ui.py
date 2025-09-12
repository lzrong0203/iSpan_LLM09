# Day 1 下午：Gradio UI介面
# 15:30-16:30 打造漂亮的網頁介面

import os
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# 載入環境變數
load_dotenv()

# === Step 1: 初始化 ===
print("=== Gradio Chatbot UI ===")
print()

# OpenAI客戶端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# === Step 2: 簡單的Gradio介面 ===
def simple_gradio_demo():
    """最簡單的Gradio範例"""
    
    def greet(name):
        return f"你好，{name}！歡迎使用Gradio！"
    
    # 建立介面
    demo = gr.Interface(
        fn=greet,
        inputs="text",
        outputs="text",
        title="簡單問候",
        description="輸入你的名字"
    )
    
    return demo

# === Step 3: AI聊天介面 ===
def create_chat_interface():
    """建立AI聊天介面"""
    
    def chat_with_ai(message, history):
        """處理聊天"""
        try:
            # 建立訊息列表
            messages = [
                {"role": "system", "content": "你是一個友善的AI助手"}
            ]
            
            # 加入歷史對話
            for h in history:
                messages.append({"role": "user", "content": h[0]})
                messages.append({"role": "assistant", "content": h[1]})
            
            # 加入新訊息
            messages.append({"role": "user", "content": message})
            
            # 呼叫API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"錯誤：{e}"
    
    # 建立聊天介面
    demo = gr.ChatInterface(
        fn=chat_with_ai,
        title="AI聊天機器人",
        description="與AI對話",
        examples=["你好", "什麼是Python?", "說個笑話"],
        retry_btn="重試",
        undo_btn="撤銷",
        clear_btn="清除"
    )
    
    return demo

# === Step 4: 多功能介面 ===
def create_multi_function_interface():
    """建立多功能介面"""
    
    # 功能1：聊天
    def chat_function(message, temperature, max_tokens):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": message}
                ],
                temperature=temperature,
                max_tokens=int(max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"錯誤：{e}"
    
    # 功能2：翻譯
    def translate_function(text, target_lang):
        prompt = f"請將以下文字翻譯成{target_lang}：\n{text}"
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"錯誤：{e}"
    
    # 功能3：摘要
    def summarize_function(text, length):
        prompt = f"請用{length}字以內摘要以下內容：\n{text}"
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"錯誤：{e}"
    
    # 建立多頁面介面
    with gr.Blocks(title="AI工具箱") as demo:
        gr.Markdown("# 🤖 AI多功能工具箱")
        gr.Markdown("選擇不同的功能來使用AI")
        
        with gr.Tab("💬 聊天"):
            with gr.Row():
                with gr.Column():
                    chat_input = gr.Textbox(
                        label="輸入訊息",
                        placeholder="問我任何問題...",
                        lines=3
                    )
                    temp_slider = gr.Slider(
                        minimum=0, maximum=1, value=0.7,
                        label="創意度 (Temperature)"
                    )
                    token_slider = gr.Slider(
                        minimum=50, maximum=500, value=150,
                        label="最大長度 (Max Tokens)"
                    )
                    chat_btn = gr.Button("發送", variant="primary")
                
                with gr.Column():
                    chat_output = gr.Textbox(
                        label="AI回應",
                        lines=8
                    )
            
            chat_btn.click(
                fn=chat_function,
                inputs=[chat_input, temp_slider, token_slider],
                outputs=chat_output
            )
        
        with gr.Tab("🌍 翻譯"):
            with gr.Row():
                with gr.Column():
                    trans_input = gr.Textbox(
                        label="原文",
                        placeholder="輸入要翻譯的文字...",
                        lines=5
                    )
                    lang_select = gr.Dropdown(
                        choices=["英文", "日文", "韓文", "法文", "西班牙文"],
                        label="目標語言",
                        value="英文"
                    )
                    trans_btn = gr.Button("翻譯", variant="primary")
                
                with gr.Column():
                    trans_output = gr.Textbox(
                        label="翻譯結果",
                        lines=5
                    )
            
            trans_btn.click(
                fn=translate_function,
                inputs=[trans_input, lang_select],
                outputs=trans_output
            )
        
        with gr.Tab("📝 摘要"):
            with gr.Row():
                with gr.Column():
                    summary_input = gr.Textbox(
                        label="原文",
                        placeholder="貼上要摘要的長文...",
                        lines=10
                    )
                    length_select = gr.Radio(
                        choices=["50", "100", "200"],
                        label="摘要長度（字）",
                        value="100"
                    )
                    summary_btn = gr.Button("生成摘要", variant="primary")
                
                with gr.Column():
                    summary_output = gr.Textbox(
                        label="摘要結果",
                        lines=5
                    )
            
            summary_btn.click(
                fn=summarize_function,
                inputs=[summary_input, length_select],
                outputs=summary_output
            )
        
        # 加入範例
        gr.Examples(
            examples=[
                ["什麼是機器學習？", "英文", "100"],
                ["Python是一種程式語言", "日文", "50"],
            ],
            inputs=[chat_input, lang_select, length_select]
        )
    
    return demo

# === Step 5: 自訂主題介面 ===
def create_custom_theme_interface():
    """建立自訂主題的介面"""
    
    def process_text(text, style):
        """處理文字的函數"""
        styles = {
            "大寫": text.upper(),
            "小寫": text.lower(),
            "反轉": text[::-1],
            "標題": text.title()
        }
        return styles.get(style, text)
    
    # 自訂CSS
    custom_css = """
    .gradio-container {
        font-family: 'Microsoft JhengHei', sans-serif;
    }
    .gr-button-primary {
        background-color: #2196F3;
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 文字處理工具
            ### 簡單的文字轉換功能
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="輸入文字",
                    placeholder="輸入你要處理的文字...",
                    lines=3
                )
                style_radio = gr.Radio(
                    choices=["大寫", "小寫", "反轉", "標題"],
                    label="選擇樣式",
                    value="大寫"
                )
                process_btn = gr.Button(
                    "處理文字",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="處理結果",
                    lines=3,
                    interactive=False
                )
                
                # 加入統計資訊
                gr.Markdown("### 📊 統計")
                char_count = gr.Number(label="字元數", value=0)
                word_count = gr.Number(label="單詞數", value=0)
        
        def process_and_count(text, style):
            result = process_text(text, style)
            chars = len(text)
            words = len(text.split())
            return result, chars, words
        
        process_btn.click(
            fn=process_and_count,
            inputs=[input_text, style_radio],
            outputs=[output_text, char_count, word_count]
        )
        
        # 加入快速範例
        gr.Examples(
            examples=[
                ["Hello World", "大寫"],
                ["PYTHON Programming", "小寫"],
                ["Gradio很好用", "反轉"]
            ],
            inputs=[input_text, style_radio]
        )
    
    return demo

# === Step 6: 執行介面 ===
print("選擇要啟動的介面：")
print("1. 簡單問候")
print("2. AI聊天")
print("3. 多功能工具箱")
print("4. 自訂主題")
print()

choice = input("選擇 (1-4)：")

# 根據選擇啟動不同介面
if choice == "1":
    demo = simple_gradio_demo()
elif choice == "2":
    demo = create_chat_interface()
elif choice == "3":
    demo = create_multi_function_interface()
elif choice == "4":
    demo = create_custom_theme_interface()
else:
    print("預設啟動AI聊天介面")
    demo = create_chat_interface()

# 啟動伺服器
print("\n正在啟動Gradio伺服器...")
print("在瀏覽器開啟 http://127.0.0.1:7860")
print("按 Ctrl+C 停止伺服器\n")

demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False  # 設為True可取得公開連結
)