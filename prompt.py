def generate_prompt(docs_by_class=None, patient_info=None):
    """
    Generates a detailed prompt for the RAG, including sources, patient information,
    and clinical history, specifying the class of each input section.

    :param documents: list of strings (papers, guidelines, or trials)
    :param docs_by_class: optional dict with keys 'guidelines', 'papers', 'trials' and list[str] values
    :param patient_info: optional dictionary with relevant patient information
    :return: prompt string
    """

    def _format_dict(info_dict):
        return "\n".join([f"{k}: {v}" for k, v in info_dict.items()])

    intro = (
        "You will receive several inputs divided by section, each indicating its class.\n"
        "- Sources (class: list[str]): relevant papers, guidelines, and clinical trials.\n"
        "- Patient information (class: dict): current demographic and clinical data.\n\n"
        "Use all sections to provide precise and up-to-date recommendations on the most suitable treatment. "
        "Cite relevant sources in your response.\n"
    )

    # Prepare sources by class if provided
    if docs_by_class:
        guidelines = "\n".join([f"- {doc.strip()}" for doc in docs_by_class.get('guidelines', []) if doc.strip()])
        papers = "\n".join([f"- {doc.strip()}" for doc in docs_by_class.get('papers', []) if doc.strip()])
        trials = "\n".join([f"- {doc.strip()}" for doc in docs_by_class.get('trials', []) if doc.strip()])
        sources_section = (
            f"Guidelines (class: list[str]):\n{guidelines}\n\n"
            f"Papers (class: list[str]):\n{papers}\n\n"
            f"Clinical Trials (class: list[str]):\n{trials}"
        )
    else:
        sources = "\n".join([f"- {doc.strip()}" for doc in documents if doc.strip()])
        sources_section = f"Sources (class: list[str]):\n{sources}"

    # Patient information
    patient_section = ""
    if patient_info:
        formatted_patient = _format_dict(patient_info)
        patient_section = f"\n\nPatient information (class: dict):\n{formatted_patient}"

    task = (
        "\n\nTASK:\n"
        "You are assisting a clinical team at a referral hospital that receives patients with rare diseases. "
        "Given the EHR and demographic data of a new patient, please:\n"
        "1. Summarize relevant medical information (diagnosis, history, comorbidities, current medications) in clear language for healthcare personnel.\n"
        "2. Identify current clinical trials that match the patient's profile.\n"
        "3. Propose a personalized therapeutic plan.\n"
        "4. Present all findings in a structured report.\n"
        "Cite relevant sources (guidelines, papers, trials) in your recommendations."
    )

    prompt = (
        f"{intro}\n"
        f"{sources_section}"
        f"{patient_section}"
        f"{task}\n\n"
        "Provide a detailed and well-motivated treatment recommendation, "
        "based on the information and sources listed above."
    )
    return prompt

# Example usage:
if __name__ == "__main__":
    docs_by_class = {
        "guidelines": [
            "2023 EULAR guidelines for the management of rare autoimmune diseases."
        ],
        "papers": [
            "Smith et al., 2022: Efficacy of Drug X in rare disease Y."
        ],
        "trials": [
            "NCT01234567: Ongoing trial of Drug Z for rare disease Y."
        ]
    }
    patient_info = {
        "Age": 42,
        "Sex": "Female",
        "Diagnosis": "Rare disease Y",
        "Current medications": "Drug A, Drug B"
    }
    print(generate_prompt(docs_by_class=docs_by_class, patient_info=patient_info))
