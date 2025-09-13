from test import Agent

system_prompt = """你在 Thought、Action、PAUSE、Observation 的循環中運行。
在循環結束時，你輸出 Answer。
使用 Thought 描述你對被問問題的想法。
使用 Action 執行你可以使用的行動之一，然後返回 PAUSE。
Observation 將是執行這些行動的結果。

你可以使用的行動有：


fetch_ticker:
分析一段文字，讓你判斷其中的金融商品標的
例如：fetch_ticker: "今天台積電股價不太行"
返回原文給你分析

fetch_stock_data:
使用股票代碼查詢近期股價變化
例如 fetch_stock_data: 2330.TW
重要股票代碼：
- 台積電: 2330.TW
- 鴻海: 2317.TW
- 聯發科: 2454.TW
注意：必須使用完整股票代碼（如 2330.TW）

analyze_sentiment:
例如 analyze_sentiment: 台積電
以"正面"、"負面"、"中性"的三種結果分析一段關於金融市場的情緒
例如：analyze_sentiment: 一段文字"台積電今天不太行" 是"負面"的
Runs a analyze_sentiment and returns results

範例對話：

Question: 今天台積電股價不太行
Thought: 我需要先分析這句話的金融標的
Action: fetch_ticker: 今天台積電股價不太行
PAUSE

Observation: 今天台積電股價不太行

Thought: 這句話提到的是"台積電"，我知道台積電的股票代碼是 2330.TW。現在查詢股價資料。
Action: fetch_stock_data: 2330.TW
PAUSE

Observation: 最近五天股價變化為：-0.025

Thought: 現在分析這句話的情緒
Action: analyze_sentiment: 今天台積電股價不太行
PAUSE

Observation: 負面

Answer: 標的：台積電 (2330.TW)，情緒：負面，股價變化：-2.5%
"""

import re


def extract_stock_code(text):
    # 定義股票代碼的正則表達式模式（支援 .tw 或 .TW）
    pattern = r"\b\d{4}\.[Tt][Ww]\b"

    # 使用正則表達式搜索文本中的股票代碼
    match = re.search(pattern, text)

    if match:
        return match.group(0).upper()  # 統一轉成大寫
    else:
        return None


def fetch_stock_data(text):
    # 如果直接傳入的就是股票代碼格式，直接使用
    if re.match(r"^\d{4}\.[Tt][Ww]$", text.strip()):
        ticker = text.strip().upper()
    else:
        ticker = extract_stock_code(text)

    print("=======", ticker)

    if ticker is None:
        return "無法識別股票代碼"

    import yfinance as yf

    # 使用 yfinance 下載指定股票代碼的數據
    stock = yf.Ticker(ticker)

    # 獲取最新的市場數據
    data = stock.history(period="5d")

    # 提取最新收盤價
    # print(data)
    change = data.Close.diff(4).iloc[-1]
    # print(change)
    ratio = change / data.Close.iloc[-1]
    return "最近五天股價變化為：" + str(round(ratio, 3))


action_re = re.compile("^Action: (\w+): (.*)$")


def fetch_ticker(text):
    # 直接返回輸入文字，讓 LLM 自己判斷
    return text


def analyze_sentiment(text):
    return f"Observation: {text}"


known_actions = {
    "fetch_ticker": fetch_ticker,
    "fetch_stock_data": fetch_stock_data,
    "analyze_sentiment": analyze_sentiment,
}


def query(question, max_turns=5):
    i = 0
    bot = Agent(system_prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split("\n") if action_re.match(a)]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            if "fetch_stock_data" in action:
                action_input = result
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return


if __name__ == "__main__":
    query("今天台積電股價不太行阿")
