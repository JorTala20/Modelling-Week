from transformers import pipeline

def get_documents_by_class(
    documents: list[str],
    possible_classes: list[str] | None = None,
    threshold: float = 0.5,                     # ← nuevo
    hypothesis_template: str = "This text is about {}."
) -> dict[str, list[str]]:
    """
    Zero-shot classify documents into candidate classes (english prompts).
    Filters out low-confidence predictions with a tunable threshold.
    """
    if possible_classes is None:
        possible_classes = ["clinical trial", "practice guideline", "research article"]

    # 1) use longer, less-ambiguous labels
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
        if top_score >= threshold:                  # 2) aplica umbral
            docs_by_class[top_label].append(doc)
    return docs_by_class
