import torch
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

from huggingface_hub import snapshot_download
snapshot_download(repo_id="bert-base-uncased")  # ✅ Correct

load_dotenv(override=True)

print(os.getenv('HF_HUB_ENABLE_HF_TRANSFER'))

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", device_map="auto", torch_dtype=torch.bfloat16)

messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a george clooney",},
    {"role": "user", "content": "How many houses can a man eat for lunch?"},
]

tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")

print(tokenized_chat[0])

print(tokenizer.decode(tokenized_chat[0]))

outputs = model.generate(tokenized_chat, max_new_tokens=128) 
print(tokenizer.decode(outputs[0]))

formatted_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, continue_final_message=True)

print('hi... \n',  formatted_chat)

print('hi... \n' ,  formatted_chat[0], 'size : ', formatted_chat['input_ids'])
##model.generate(formatted_chat)