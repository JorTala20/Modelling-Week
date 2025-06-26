from zero_shot import  get_documents_by_class
from prompt import generate_prompt
from rag import generate_text
from transformers import pipeline
import time
import requests

#May need to update the tokens if the license has expired
HF_TOKEN = "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"  # Hugging Face token
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1" # zero-shot model
