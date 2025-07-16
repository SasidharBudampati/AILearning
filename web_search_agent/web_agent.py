# web_search.py

import os
from smolagents import InferenceClientModel
from huggingface_hub import InferenceClient

from dotenv import load_dotenv
from web_search import visit_webpage

load_dotenv(override=True)

HF_TOKEN=os.getenv("HF_TOKEN")

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    WebSearchTool,
    LiteLLMModel,
)
model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
model = InferenceClientModel(model=model_id)

web_agent = ToolCallingAgent(
    tools=[WebSearchTool(), visit_webpage],
    model=model,
    max_steps=10,
    name="web_search_agent",
    description="Runs web searches for you.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)

answer = manager_agent.run("""
Do search in the web page https://weather.com/ (url) using the web search tool I have provided.
                           Provide me the weather in FH in Fremont, CA and Vijayawada, AP, India in simple JSON format
                           {{
                                city: {city},
                                temperature: {temperature}, 
                                wind : {wind speed}                           
                           }}
                           if you can't find the details in the website. Just say "Sorry, details not found"
                           but DONOT hallucinate.
""")