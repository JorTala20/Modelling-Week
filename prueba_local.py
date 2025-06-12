from utils import *

if __name__ == "__main__":
    user_input = input("Mensaje (o 'salir' para terminar): ")
    cui_prompt = normalizar_prompt_a_cui(user_prompt)
    documents = cli(cui_prompt)
    classified_docs = get_documents_by_class(documents)
    prompt = generate_prompt(classified_docs, user_prompt)
    response = generate_text(prompt, MODEL_ID, HF_TOKEN)
    print(f"La respuesta generada es: {response}")