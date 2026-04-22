from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

response = client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[{"role": "user", "content": "Hello"}]
)

print(response.choices[0].message.content)