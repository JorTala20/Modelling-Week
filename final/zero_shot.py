from transformers import pipeline
import torch

def get_documents_by_class(
    documents: list[str],
    possible_classes: list[str] | None = None,
    threshold: float = 0.5,
    hypothesis_template: str = "This text is about {}."
) -> dict[str, list[str]]:
    """
    Zero-shot classifies documents into candidate classes (english prompts).
    Filters out low-confidence predictions with a threshold.
    Args:
        documents (List[str]): documents to classify
        possible_classes (List[str]): possible classifications for the model
        threshold (float)
        hypotesis_template (str)

    Returns:
        dict[str, List[str]]: list of documents divided per class
    """
    if possible_classes is None:
        possible_classes = ["clinical trial", "practice guideline", "research article"]

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1,
    )

    docs_by_class = {cls: [] for cls in possible_classes}
    for doc in documents:
        res = classifier(
            doc,
            candidate_labels=possible_classes,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )
        top_label, top_score = res["labels"][0], res["scores"][0]
        if top_score >= threshold:
            docs_by_class[top_label].append(doc)
    return docs_by_class

# Example usage
if __name__ == "__main__":
    # 10 example documents
    base_documents = [
        "A new smartphone model was released with advanced camera features.",
        "Researchers discovered a potential link between sleep and memory retention.",
        "The city council approved new regulations for electric scooters.",
        "A clinical trial is underway to test a novel cancer therapy.",
        "Guidelines for sustainable urban development were updated this year.",
        "A recent article explores the impact of social media on mental health.",
        "Practice guidelines for asthma management have been revised.",
        "The study analyzes voting patterns in recent elections.",
        "A randomized controlled trial evaluated a new diet for weight loss.",
        "Experts published recommendations for cybersecurity best practices."
    ]
    sample_documents = base_documents

    classes = None
    result = get_documents_by_class(sample_documents, classes)
    for cls, docs in result.items():
        print(f"\nClass: {cls}")
        for doc in docs:
            print(f" - {doc}")
