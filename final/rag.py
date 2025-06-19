from utils import *
import requests


def generate_text(prompt: str, model_id: str, hf_token: str, max_new_tokens: int = 2000) -> str:
    """
    Sends a prompt to a Hugging Face model using the Inference API and returns the generated text.

    Args:
        prompt (str): The input prompt to send to the model.
        model_id (str): The Hugging Face model identifier.
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


# Example usage 
if __name__ == "__main__":
    HF_TOKEN = "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"  # Hugging Face token
    MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

            # Example model

    prompt = "You will receive several inputs divided by section, each indicating its class.\n- Sources (class: list[str]): relevant papers, guidelines, and clinical trials.\n- Patient information (class: dict): current demographic and clinical data.\n\nUse all sections to provide precise and up-to-date recommendations on the most suitable treatment. Cite relevant sources in your response.\n\nGuidelines (class: list[str]):\n- METHODS: All groups are visited at home for therapy sessions 5 times per week for the first 3 weeks and 3 times per week\n- METHODS: A 15-course FES was performed for 15 min and 3 times per week .\n- METHODS: The treatment was given 5 times weekly in the first two weeks and 3 times weekly in the later 6 weeks .\n- Environment and Lung Cancer\n- METHODS: Eccentric training participants exercised their dominant limb with a dynamometer in eccentric mode at 60/s , 3\n\nPapers (class: list[str]):\n\n\nClinical Trials (class: list[str]):\n\n\nPatient information (class: str):\n35-year-old male with lung cancer and ibuprofen 3 times per week\n\nTASK:\nYou are assisting a clinical team at a referral hospital that receives patients with rare diseases. Given the EHR and demographic data of a new patient, please:\n1. Summarize relevant medical information (diagnosis, history, comorbidities, current medications) in clear language for healthcare personnel.\n2. Present current clinical trials that match the patient's profile between the one you're receiving.\n3. Propose a personalized therapeutic plan.\n4. Present all findings in a structured report.\nCite relevant sources (guidelines, papers, trials) in your recommendations.\n\nProvide a detailed and well-motivated treatment recommendation, based on the information and sources listed above."
    """
    You are a medical assistant. Based on the patient's history, generate a therapeutic plan.
    
    Patient: 45-year-old male, diagnosed with type 1 diabetes and hypertension.
    Current medication: insulin and metformin.
    """

    print("Sending prompt to Hugging Face...")
    result = generate_text(prompt, MODEL_ID, HF_TOKEN)
    print("\nGenerated Output:\n")
    print(result)
