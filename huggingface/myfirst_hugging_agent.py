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
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv(override=True)

## I expect the inference client to understand the context
## and respond to the prompt
## Place the environment variable HF_TOKEN in .env. Token shall be obtained from hf.co/../tokens
HF_TOKEN=os.getenv('HF_TOKEN')
print(HF_TOKEN)
client = InferenceClient(model="meta-llama/Llama-4-Scout-17B-16E-Instruct", token=HF_TOKEN)


# %%
output = client.chat.completions.create(
    [
        {"role:", "user", "content:", "The capital of India is "},

    ],
    stream=False,
    max_tokens=100
)

print("Output : ", output.message[0].message.content)

# %%
