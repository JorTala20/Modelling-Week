from zero_shot import  get_documents_by_class
from prompt import generate_prompt
from rag import generate_text
from transformers import pipeline
import time
import requests

HF_TOKEN = "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"  # Hugging Face token
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1" # zero-shot model
