from openai import OpenAI
import os

client = OpenAI(
    api_key="flp_pmrHz69WHNNJmtXdParHDm89EcsT2PHuNvSaVZlZDyu74",
    base_url="https://api.friendli.ai/serverless/v1",
)

completion = client.chat.completions.create(
    model="meta-llama-3.1-8b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a funny joke."},
    ],
)

print(completion.choices[0].message.content)