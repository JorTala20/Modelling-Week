from utils import *

def get_documents_by_class(documents, possible_classes=None):
    """
    Classify a list of documents into one of the candidate labels using zero-shot learning.
    Returns a dict: Keys are class names, values are lists of document texts assigned to that class.
    The model used is facebook/bart-large-mnli by META.
    """
    if possible_classes is None:
        possible_classes = ["trial", "guidelines", "paper"]

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    results = [classifier(doc, possible_classes) for doc in documents]

    docs_by_class = {cls: [] for cls in possible_classes}
    for result in results:
        top_class = result['labels'][0]
        docs_by_class[top_class].append(result['sequence'])
    return docs_by_class

# Example usage: classify a list of sample documents into predefined categories using zero-shot classification.
if __name__ == "__main__":
    # 10 example documents (repeating the base list for demonstration)
    base_documents = [
        "This clinical trial investigates the effects of a new drug.",
        "The latest guidelines for COVID-19 prevention have been published.",
        "A recent paper discusses advances in machine learning.",
        "Guidelines for hypertension management were updated.",
        "This paper presents a novel approach to data analysis.",
        "A randomized trial was conducted to test the vaccine.",
        "Clinical guidelines recommend regular exercise for heart health.",
        "The trial results show significant improvement in patients.",
        "This paper reviews the literature on deep learning.",
        "Guidelines for diabetes care have changed recently."
    ]
    sample_documents = base_documents

    classes = ["trial", "guidelines", "paper"]
    result = get_documents_by_class(sample_documents, classes)
    for cls, docs in result.items():
        print(f"\nClass: {cls}")
        for doc in docs:
            print(f" - {doc}")
