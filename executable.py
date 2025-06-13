from hs_nuevo import *
from utils import *

if __name__ == "__main__":
    query = "35-year-old male with lung cancer and ibuprofen 3 times per week"
    documents = get_documents_hybrid_search(query)
    classified_docs = get_documents_by_class(documents)
    new_prompt = generate_prompt(classified_docs, query)
    response = generate_text(new_prompt, MODEL_ID, HF_TOKEN)
    print(response)
