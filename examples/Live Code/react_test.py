from test import Agent

system_prompt = """你在 Thought、Action、PAUSE、Observation 的循環中運行。
在循環結束時，你輸出 Answer。
使用 Thought 描述你對被問問題的想法。
使用 Action 執行你可以使用的行動之一，然後返回 PAUSE。
Observation 將是執行這些行動的結果。

你可以使用的行動有：


fetch_ticker:
找出一段文字中所描述的金融商品、標的或是整個市場
例如：fetch_ticker： 一段文字"今天 CPI 低於預期" 標的為"市場"
     fetch_ticker: 一段文字"台積電今天不太行" 標的為"台積電"

fetch_stock_data:
例如 fetch_stock_data: 台積電
台積電在yfinance的代號為 2330.tw
查詢近期股價變化

analyze_sentiment:
例如 analyze_sentiment: 台積電
以"正面"、"負面"、"中性"的三種結果分析一段關於金融市場的情緒
例如：analyze_sentiment: 一段文字"台積電今天不太行" 是"負面"的
Runs a analyze_sentiment and returns results

範例對話：

Question: 台積電將調高資本資出
Thought: 這句話的金融標的為何
Action: 分析標的: 台積電將調高資本資出
PAUSE

這時會返回：

Observation: 這句話的標的為"台積電"

接下來你會執行：

Action: fetch_stock_data: 台積電
台積電在 yfinance 的代號為 2330.tw
PAUSE

Observation: 台積電最近五天股價變化（例如：-20, -10, 0, 20）

接下來你會執行：

Action: analyze_sentiment: 最近五天股價變化為（例如：-20, -10, 0, 20），"台積電將調高資本資出"的情緒為?
PAUSE

最後你輸出：

Answer: 標的：台積電，情緒：正面，股價變化：例如：-20, -10, 0, 20）
"""

import re

def extract_stock_code(text):
    # 定義股票代碼的正則表達式模式（以 2454.tw 為例）
    pattern = r'\b\d{4}\.tw\b'

    # 使用正則表達式搜索文本中的股票代碼
    match = re.search(pattern, text)

    if match:
        return match.group(0)
    else:
        return None

def fetch_stock_data(text):
    ticker = extract_stock_code(text)
    print("=======", ticker)
    import yfinance as yf
    # 使用 yfinance 下載指定股票代碼的數據
    stock = yf.Ticker(ticker)

    # 獲取最新的市場數據
    data = stock.history(period="5d")

    # 提取最新收盤價
    # print(data)
    change = data.Close.diff(4).iloc[-1]
    # print(change)
    ratio = change / data.Close[-1]
    return "最近五天股價變化為：" + str(round(ratio, 3))


action_re = re.compile(r'^Action: (\w+): (.*)$')

def fetch_ticker(text):
  return f"Observation: {text}"

def analyze_sentiment(text):
  return f"Observation: {text}"

known_actions = {
    "fetch_ticker": fetch_ticker,
    "fetch_stock_data": fetch_stock_data,
    "analyze_sentiment": analyze_sentiment
}
def query(question, max_turns=5):
    i = 0
    bot = Agent(system_prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a)
            for a in result.split('\n')
            if action_re.match(a)
        ]
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
