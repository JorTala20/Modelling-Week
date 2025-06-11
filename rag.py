from utils import *


def generate_text(prompt: str, model_id: str, hf_token: str, max_new_tokens: int = 2000) -> str:
    """
    Sends a prompt to a Hugging Face model using the Inference API and returns the generated text.

    Args:
        prompt (str): The input prompt to send to the model.
        model_id (str): The Hugging Face model identifier (e.g., "HuggingFaceH4/zephyr-7b-beta").
        hf_token (str): Hugging Face access token with read permissions.
        max_new_tokens (int): Maximum number of tokens to generate in the response.

    Returns:
        str: The generated text from the model.
    """
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {hf_token}"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"].strip()
    else:
        raise Exception(f"API Error {response.status_code}: {response.text}")


# Example usage (comment out or remove before importing in server.py)
if __name__ == "__main__":
    HF_TOKEN = "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"  # Replace with your Hugging Face token
    MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"            # Example model

    prompt = """
You are a medical assistant. Based on the patient's history, generate a therapeutic plan.

Patient: 45-year-old male, diagnosed with type 1 diabetes and hypertension.
Current medication: insulin and metformin.
"""

    print("Sending prompt to Hugging Face...")
    result = generate_text(prompt, MODEL_ID, HF_TOKEN)
    print("\nGenerated Output:\n")
    print(result)
