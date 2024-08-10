import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from transformers.agents import Tool, HfEngine, ReactJsonAgent
from transformers.agents.llm_engine import MessageRole, get_clean_message_list
from typing import List, Dict
from huggingface_hub import InferenceClient
import json
import openai

from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

class Item(BaseModel):
    content: str

@app.post("/ask-question/")
async def example(item: Item):
    return {"content": item.content}
