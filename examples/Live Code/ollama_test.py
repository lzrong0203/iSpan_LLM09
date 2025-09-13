import ollama

response = ollama.chat(
    model="gemma3:270m",
    messages=[{"role": "user", "content": "Who are you?"}]
)

print(response)