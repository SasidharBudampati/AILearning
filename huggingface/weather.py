# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
##I want the agent to find the weather of specific location and report in a specific format

import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv(override=True)

HF_TOKEN=os.getenv("HF_TOKEN")
print(HF_TOKEN)

client = InferenceClient(model="meta-llama/Llama-4-Scout-17B-16E-Instruct", token=HF_TOKEN)


SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {{"location": {{"type": "string"}}}}
example use :
```
{
{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}
}

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one or more actions to take :
Action:
```
$JSON_BLOB
```
You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question should be rich text with beautiful images.
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in Fremont, CA and Sunnyvale, CA?"},
]

messages

# %%
output = client.chat.completions.create(
    messages=messages,
    stream=False,
    max_tokens=1000,
)
print(output.choices[0].message.content)

# %%
