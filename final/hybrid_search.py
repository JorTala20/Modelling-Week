# =============================================================================
#  HYBRID SEARCH
# =============================================================================
#  • Ingestion           Synthea (synthetic EHR, CSV)
#                        PubMed-RCT (Kaggle, CSV)
#                        ClinicalTrials.gov (REST v2, on-the-fly)
#  • Embeddings          SBERT (biomed-roberta-base-msmarco)
#  • Vector index        FAISS (CPU/GPU)  ─or─  HNSWlib (fallback)
#  • Lexical index       BM25 (rank-bm25)
#  • Fusion              Reciprocal Rank Fusion (RRF)
# =============================================================================

"""
hybrid_search.py

A compact hybrid retrieval engine for biomedical text.  It
integrates sparse BM25 ranking with dense SBERT embeddings and fuses the two
streams using Reciprocal Rank Fusion.  Three heterogeneous corpora are
ingested:

1. **Synthea** electronic-health-record CSVs  → one fragment per clinical fact;
2. **PubMed-RCT** dataset (labelled sentences of abstracts);
3. **ClinicalTrials.gov** titles fetched at query time via REST.
"""

from __future__ import annotations

###############################################################################
# Standard library
###############################################################################
import argparse, sys
from pathlib import Path
from typing import List, Dict, Tuple
from functools import lru_cache
from itertools import chain
from collections import defaultdict
import warnings
from urllib.parse import quote

###############################################################################
# Third‑party
###############################################################################
import numpy as np
import pandas as pd
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import spacy, scispacy, scispacy.umls_linking
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ───────────────────────────── Optional ANN engines ──────────────────────────
try:
    import faiss;  _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
try:
    import hnswlib; _HAS_HNSW = True
except ImportError:
    _HAS_HNSW = False

###############################################################################
# Hyper‑parameters and file paths (all relative to CWD)
###############################################################################
DOC_IDS_NPY   = "doc_ids.npy"         # numpy array[str] – one per fragment
DOC_CUIS_NPY  = "doc_cuis.npy"        # numpy array[str] – tokenised by CUIs
VEC_FAISS     = "vectors.index"       # binary FAISS index (flat IP)
VEC_HNSW      = "hnsw_index.bin"       # binary HNSW index
CORPUS_CSV    = "corpus.csv"          # id,text – master corpus cache

DIM           = 384                    # SBERT embedding dimension
MODEL_NAME    = "sentence-transformers/biomed-roberta-base-msmarco"
MODEL_PATH    = "transformer_model"    # on‑disk SBERT cache

###############################################################################
# UMLS / spaCy helpers
###############################################################################
def load_umls_pipeline():
    """
    Build and return a *spaCy* pipeline augmented with the **scispaCy UMLS linker**.

    Returns
    -------
    tuple(nlp, linker)
        * **nlp** – spaCy `Language` object.
        * **linker** – the `EntityLinker` pipe already installed in `nlp`.
    """
    
    print("Loading spaCy + UMLS linker …")
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe(
        "scispacy_linker",
        last=True,
        config={"linker_name": "umls", "resolve_abbreviations": True},
    )
    linker = nlp.get_pipe("scispacy_linker")
    print("Pipeline ready!\n")
    return nlp, linker


def extract_cuis(text: str, nlp) -> List[str]:
    """
    Return *all* distinct CUIs found in *text*.
    Only the first (highest‑scoring) candidate per entity is kept.

    Parameters
    ----------
    text : str
        Raw input string.
    nlp
        spaCy pipeline (obtained from: load_umls_pipeline).

    Returns
    -------
    List[str]
        List of CUIs in textual order (no duplicates).
    """
    
    return [ent._.kb_ents[0][0]   # first candidate CUI
            for ent in nlp(text).ents
            if getattr(ent._, "kb_ents", None)]


def prompt_to_cuis(text: str, nlp) -> str:
    """
    Replace every recognised medical entity in *text* with its CUI.
    Non‑medical tokens are preserved so the BM25 context is intact.

    Parameters
    ----------
    text : str
        Input sentence or paragraph.
    nlp : spacy.Language
        A spaCy pipeline containing the scispaCy linker.

    Returns
    -------
    str
        The transformed string (entities → CUI codes).
    """
    
    doc = nlp(text)
    out = text
    #  Walk backwards so that character offsets remain valid while editing
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        if ent._.kb_ents:
            cui = ent._.kb_ents[0][0]
            out = out[:ent.start_char] + cui + out[ent.end_char:]
    return out

###############################################################################
# CSV Ingestion (Synthea + PubMed‑RCT)
###############################################################################

def ingest_synthea_csv(csv_dir: str) -> Tuple[List[str], List[str]]:
    """
    Convert an *entire* Synthea export (CSV directory) into plain‑text EHR fragments.
    Each row of every clinical table becomes a short human‑readable sentence.

    Parameters
    ----------
    csv_dir : str
        Path pointing to the directory containing the Synthea CSV files

    Returns
    -------
    tuple(list[str], list[str])
        * **texts** – list of textual fragments.
        * **ids**   – list of unique fragment identifiers (same length).
    """
    
    p = Path(csv_dir)
    
    # Load every Synthea table as *string* so we do no implicit numeric casts
    patients   = pd.read_csv(p / "patients.csv", dtype=str)
    conditions = pd.read_csv(p / "conditions.csv", dtype=str)
    allergies  = pd.read_csv(p / "allergies.csv", dtype=str)
    careplans  = pd.read_csv(p / "careplans.csv", dtype=str)
    devices    = pd.read_csv(p / "devices.csv", dtype=str)
    encounters = pd.read_csv(p / "encounters.csv", dtype=str)
    imaging    = pd.read_csv(p / "imaging_studies.csv", dtype=str)
    immun      = pd.read_csv(p / "immunizations.csv", dtype=str)
    meds       = pd.read_csv(p / "medications.csv", dtype=str)
    obs        = pd.read_csv(p / "observations.csv", dtype=str)
    procs      = pd.read_csv(p / "procedures.csv", dtype=str)

    def safe(val, sl=None):
        """Return *val* safely truncated to *sl* characters (or 'unknown')."""
        return str(val)[:sl] if pd.notna(val) else "unknown"

    texts, ids = [], []
    
    for _, pat in patients.iterrows():
        pid = pat["Id"]
        # ── demographics
        texts.append(f"Patient {safe(pat['GENDER'])} born {safe(pat['BIRTHDATE'],10)} "
                     f"race {safe(pat['RACE'])}")
        ids.append(f"ehr_{pid}_demo")

        # ── iterate over each related table by foreign key PATIENT = pid
        for i, row in conditions[conditions["PATIENT"] == pid].iterrows():
            texts.append(f"Condition {safe(row['DESCRIPTION'])} started {safe(row['START'],10)}")
            ids.append(f"ehr_{pid}_cond_{i}")

        for i, row in encounters[encounters["PATIENT"] == pid].iterrows():
            texts.append(f"Encounter {safe(row['ENCOUNTERCLASS'])} for {safe(row['DESCRIPTION'])} "
                         f"from {safe(row['START'],10)} to {safe(row['STOP'],10)}")
            ids.append(f"ehr_{pid}_enc_{i}")

        for i, row in allergies[allergies["PATIENT"] == pid].iterrows():
            texts.append(f"Allergy to {safe(row['DESCRIPTION'])} recorded {safe(row['START'],10)}")
            ids.append(f"ehr_{pid}_alg_{i}")

        for i, row in careplans[careplans["PATIENT"] == pid].iterrows():
            texts.append(f"Care plan {safe(row['DESCRIPTION'])} "
                         f"{safe(row['START'],10)}–{safe(row['STOP'],10)}")
            ids.append(f"ehr_{pid}_care_{i}")

        for i, row in devices[devices["PATIENT"] == pid].iterrows():
            texts.append(f"Device {safe(row['DESCRIPTION'])} implanted {safe(row['START'],10)}")
            ids.append(f"ehr_{pid}_dev_{i}")

        for i, row in imaging[imaging["PATIENT"] == pid].iterrows():
            body = safe(row.get("BODYSITE_DESCRIPTION") or row.get("BODYSITE"))
            texts.append(f"Imaging study {body} on {safe(row['DATE'],10)}")
            ids.append(f"ehr_{pid}_img_{i}")

        for i, row in immun[immun["PATIENT"] == pid].iterrows():
            texts.append(f"Immunisation {safe(row['DESCRIPTION'])} on {safe(row['DATE'],10)}")
            ids.append(f"ehr_{pid}_imm_{i}")

        for i, row in meds[meds["PATIENT"] == pid].iterrows():
            texts.append(f"Medication {safe(row['DESCRIPTION'])} from {safe(row['START'],10)} "
                         f"to {safe(row['STOP'],10)}")
            ids.append(f"ehr_{pid}_med_{i}")

        for i, row in obs[obs["PATIENT"] == pid].iterrows():
            texts.append(f"Observation {safe(row['DESCRIPTION'])}={safe(row['VALUE'])} "
                         f"{safe(row['DATE'],10)}")
            ids.append(f"ehr_{pid}_obs_{i}")

        for i, row in procs[procs["PATIENT"] == pid].iterrows():
            texts.append(f"Procedure {safe(row['DESCRIPTION'])} on {safe(row['DATE'],10)}")
            ids.append(f"ehr_{pid}_proc_{i}")

    print(f"-> Synthea fragments: {len(texts)}")
    return texts, ids


def ingest_pubmed_csv(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Convert a PubMed-RCT CSV into one fragment per sentence,
    prefixing each sentence with its rhetorical label.

    Args
    ----
    csv_path : str
        Path to the PubMed-RCT CSV (needs columns `abstract_text`,`target`).

    Returns
    -------
    Tuple[List[str], List[str]]
        * texts – sentence fragments
        * ids   – pm_<index> for each sentence
    """
    
    df = pd.read_csv(csv_path, dtype=str)
    if {"abstract_text", "target"} - set(df.columns):
        raise ValueError("CSV must have columns 'abstract_text' and 'target'")
    df = df[df["abstract_text"].notna() & df["target"].notna()]
    
    texts = [f"{row['target']}: {row['abstract_text']}" for _, row in df.iterrows()]
    ids   = [f"pm_{i}" for i in range(len(texts))]
    
    print(f"-> PubMed-RCT fragments: {len(texts)}")
    return texts, ids


def pre_ingest(synthea_path: str = "synthea/csv", pubmed_path: str = "pubmed_rct/train.csv") -> Tuple[List[str], List[str]]:
    """
    Helper: run Synthea and PubMed ingestion sequentially.
    Returns combined list of texts + IDs, ready to embed.

    Returns
    -------
    Tuple[List[str], List[str]]
    """
    
    texts, ids = [], []

    # Ingest from Synthea
    synthea_texts, synthea_ids = ingest_synthea_csv(synthea_path)
    texts += synthea_texts
    ids += synthea_ids

    # Ingest from PubMed-RCT
    pubmed_texts, pubmed_ids = ingest_pubmed_csv(pubmed_path)
    texts += pubmed_texts
    ids += pubmed_ids

    return texts, ids


###############################################################################
#  ClinicalTrials.gov  (hierarchical search)
###############################################################################

@lru_cache(maxsize=256)
def ct_query(term: str, size: int) -> List[str]:
    """
    Cached JSON query against ClinicalTrials.gov v2.

    Args
    ----
    term : str
        Query string.
    size : int
        Maximum number of trials to fetch.

    Returns
    -------
    List[str]
        Study titles containing the term.
    """
    
    url = f"https://clinicaltrials.gov/api/v2/studies?query.term={quote(term)}&pageSize={size}"
    try:
        js = requests.get(url, timeout=30).json()
        return [
            e["protocolSection"]["identificationModule"]["briefTitle"].strip()
            for e in js.get("studies", [])
            if e.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle")
        ]
    except Exception:
        return []
        

def synonim_cui(cui: str, linker) -> List[str]:
    """
    Convenience wrapper (canonical + two aliases) for ClinicalTrials search.

    Args
    ----
    cui : str
        UMLS concept identifier.

    Returns
    -------
    List[str]
        Unique synonyms.
    """
    
    entity = linker.kb.cui_to_entity.get(cui)
    if not entity:
        return [cui]

    sinonimos = set(entity.aliases)
    sinonimos.add(entity.canonical_name)
    return list(sinonimos)


def ingest_clinical_trials(linker, terms: List[str], limit: int = 200) -> List[Tuple[str, List[str]]]:
    """
    Hierarchical ClinicalTrials.gov search.

    Parameters
    ----------
    linker : scispaCy linker
    terms  : List[str]
        CUIs sorted by clinical salience (first element is the main concept).
    limit  : int
        Max trials per API request.

    Returns
    -------
    List[(query_label, List[str])]
        Each tuple contains the compounded search label and its matching titles.
    """
    
    def consultation(cui):
        qs = sorted(synonim_cui(cui, linker))
        numero = int(len(qs)/3)+1
        result = []
        for q in qs[:numero]:
            print(q)
            url = f"https://clinicaltrials.gov/api/v2/studies?query.term={quote(q)}&pageSize={limit}"
            try:
                js = requests.get(url, timeout=30).json()
                result += [
                    e.get("protocolSection", {})
                     .get("identificationModule", {})
                     .get("briefTitle", "").strip()
                    for e in js.get("studies", [])
                ]
            except Exception:
                return result
        return result

    if not terms:
        return []

    # Most important term in first place
    primary = consulta(terms[0])

    # Filtering by remaining terms
    filtrados = []

     remaining_tokens = list(
            map(str.lower, chain.from_iterable(synonim_cui(t, linker)[:2] for t in terms[1:]))
     )

    filtered = [t for t in primary if any(tok in t.lower() for tok in remaining_tokens)]
    final = filtered if filtered else primary
    return [(f"{' & '.join(terms)}", final[:limit])]

###############################################################################
#  Priority table (semantic type → clinical importance)
###############################################################################

TUI_PRIORITY: Dict[str, int] = {
    "T047": 1,  # Disease or Syndrome
    "T191": 1,  # Neoplastic Process
    "T121": 2,  # Pharmacologic Substance
    "T200": 2,  # Clinical Drug
    "T061": 3,  # Therapeutic / Preventive Procedure
    "T060": 3,  # Diagnostic Procedure
    "T123": 4,  # Biologically Active Substance
    "T109": 5,  # Organic Chemical
    "T103": 6,  # Chemical
    "T074": 7,  # Medical Device
    "T170": 8,  # Intellectual Product
    "T078": 9,  # Idea or Concept
}


def cui_priority(cui: str, linker) -> int:
    """
    Map CUI → integer priority based on its UMLS semantic type.
    Smaller = clinically more salient; fall back = 999.
    """
    
    ent = linker.kb.cui_to_entity.get(cui)
    if not ent or not ent.types:
        return 999
    return min(TUI_PRIORITY.get(t, 999) for t in ent.types)

###############################################################################
#  Trial-specific ingestion wrapper
###############################################################################

def ingest_trials(nlp, linker, query_text: str, limit: int = 100) -> Tuple[List[str], List[str]]:
    """
    Run UMLS extraction + ClinicalTrials hierarchical search for *query_text*.

    Args
    ----
    nlp        : spaCy Language
    linker     : scispaCy UMLS linker
    query_text : str
    limit      : int
        Max trials per API request.

    Returns
    -------
    Tuple[List[str], List[str]]
        Titles and corresponding IDs (ct_*).
    """
    
    cuis = sorted(extract_cuis(query_text, nlp),
                  key=lambda c: cui_priority(c, linker))
    trials = ingest_clinical_trials(linker, cuis, limit)

    t_texts, t_ids = [], []
    for block, (label, titles) in enumerate(trials):
        for i, title in enumerate(titles):
            t_texts.append(title)
            t_ids.append(f"ct_{block}_{i}")
    return t_texts, t_ids

###############################################################################
#  Vector index helpers
###############################################################################

def build_vector_index(texts: List[str], model) -> SentenceTransformer:
    """
    Encode *texts* and create either a FAISS or HNSW index on disk.

    Side effect
    -----------
    Writes VEC_FAISS or VEC_HNSW and prints confirmation.
    """
    
    vecs  = model.encode(texts, normalize_embeddings=True)

    if _HAS_FAISS:
        idx = faiss.IndexFlatIP(DIM)
        idx.add(vecs.astype(np.float32))
        faiss.write_index(idx, VEC_FAISS)
        print("OK --> FAISS index saved")
        
    elif _HAS_HNSW:
        idx = hnswlib.Index(space="cosine", dim=DIM)
        idx.init_index(len(vecs), ef_construction=200, M=16)
        idx.add_items(vecs, np.arange(len(vecs)))
        idx.save_index(VEC_HNSW)
        print("OK --> HNSW index saved")
    else:
        raise ImportError("Install faiss-cpu or hnswlib")

    return model

###############################################################################
#  BM25 helpers
###############################################################################

def compute_cui_docs(texts: List[str], nlp) -> List[str]:
    """
    Convert each document to a sequence of CUIs for BM25.

    Args
    ----
    texts : List[str]

    Returns
    -------
    List[str]
        Each element is 'CUI1 CUI2 …'.
    """
    return [prompt_to_cuis(t.lower(), nlp) for t in texts]


def bm25_scores(docs_as_cuis: List[str], query: str, nlp, min_overlap: int = 2) -> np.ndarray:
    """
    BM25 with optional filter: require at least *min_overlap* CUIs in common.

    Returns
    -------
    np.ndarray
        Score for every document (same ordering as docs_as_cuis).
    """
    
    docs_tokenised = [d.split() for d in docs_as_cuis]
    query_tokens   = prompt_to_cuis(query.lower(), nlp).split()

    # mask keeps docs with enough overlapping CUIs
    mask = [sum(tok in d for tok in query_tokens) >= min_overlap
            for d in docs_tokenised]
    docs_filtered = [d for d, keep in zip(docs_tokenised, mask) if keep]
    if not docs_filtered:
        docs_filtered, mask = docs_tokenised, [True] * len(docs_tokenised)

    bm25 = BM25Okapi(docs_filtered)
    scores_part = bm25.get_scores(query_tokens)

    full = np.zeros(len(docs_as_cuis))
    ptr  = 0
    for i, keep in enumerate(mask):
        if keep:
            full[i] = scores_part[ptr]
            ptr += 1
    return full

###############################################################################
#  Dense vector search
###############################################################################

def dense_scores(query: str, model: SentenceTransformer, top_n=150) -> Dict[str, float]:
    """
    ANN search (FAISS or HNSW) → similarity scores.

    Returns
    -------
    Dict[str, float]
        Mapping doc_id → cosine similarity.
    """
    
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    ids   = np.load(DOC_IDS_NPY)

    if Path(VEC_FAISS).exists() and _HAS_FAISS:
        idx = faiss.read_index(VEC_FAISS)
        D, I = idx.search(q_emb, top_n)
        sims = 1 - D[0]
        return {str(ids[i]): float(sims[j])
                for j, i in enumerate(I[0]) if i < len(ids)}
    idx = hnswlib.Index(space="cosine", dim=DIM); idx.load_index(VEC_HNSW)
    lbl, dist = idx.knn_query(q_emb, k=top_n)
    sims = 1 - dist[0]
    return {str(ids[i]): float(sims[j])
            for j, i in enumerate(lbl[0]) if i < len(ids)}

###############################################################################
#  Reciprocal Rank Fusion
###############################################################################

def rrf(
    bm25_vec: np.ndarray,
    dense_dict: Dict[str, float],
    doc_ids: List[str],
    k: int = 20,
    rrf_k: int = 60,
    allow_dense_only: int = 0,
    intersection_bonus: float = 0.1,
) -> List[Tuple[str, float]]:
    """
    Intersection-aware Reciprocal Rank Fusion.

    • BM25 contributes 1 / (rrf_k + rank_bm25).  
    • Dense contributes 1 / (rrf_k + rank_dense) if the document has BM25 > 0
      or is within the first *allow_dense_only* purely-dense hits.  
    • If a document appears in **both** rankings, a fixed bonus
      *intersection_bonus* is added to the final score.

    Parameters
    ----------
    bm25_vec : np.ndarray
        BM25 scores (raw or normalised); only their relative order for scores > 0 matters.
    dense_dict : Dict[str, float]
        Mapping doc_id → dense cosine similarity.
    doc_ids : List[str]
        IDs in the exact order of *bm25_vec*.
    k : int, default 20
        Number of final results to return.
    rrf_k : int, default 60
        RRF constant (large values reduce the weight of low-rank items).
    allow_dense_only : int, default 0
        How many “dense-only” documents to keep (0 = dense must also appear in BM25).
    intersection_bonus : float, default 0.1
        Extra score added when a document is present in both rankings.

    Returns
    -------
    List[Tuple[str, float]]
        Top-k list of (doc_id, fused_score) sorted by descending score.
    """
    scores: Dict[str, float] = defaultdict(float)

    # 1) Build BM25 ranking: indices with score > 0 and map doc → rank
    nz_idx = np.flatnonzero(bm25_vec)               # indices where BM25 > 0
    if nz_idx.size > 0:
        # sort by descending BM25
        sorted_bm25_idx = nz_idx[np.argsort(-bm25_vec[nz_idx])]
    else:
        sorted_bm25_idx = np.array([], dtype=int)
    bm25_rank = {doc_ids[i]: rank + 1 for rank, i in enumerate(sorted_bm25_idx)}

    # 2) Build dense ranking: sorted list and map doc → rank
    dense_sorted = sorted(dense_dict.items(), key=lambda x: -x[1])
    dense_rank = {did: rank + 1 for rank, (did, _) in enumerate(dense_sorted)}

    # 3) Sets and counters
    bm25_seen = set(bm25_rank)
    added_dense_only = 0

    # 4) First iterate BM25 (guarantees inclusion of all BM25 > 0 docs)
    for did, rank in bm25_rank.items():
        scores[did] += 1.0 / (rrf_k + rank)

    # 5) Iterate dense ranking with filter + intersection bonus
    #    Break early once we have enough candidates (speed-up)
    for rank, (did, _) in enumerate(dense_sorted, start=1):
        in_bm25 = did in bm25_seen
        if not in_bm25:
            if added_dense_only >= allow_dense_only:
                continue
            added_dense_only += 1

        scores[did] += 1.0 / (rrf_k + rank)

        # fixed bonus for intersection
        if in_bm25:
            scores[did] += intersection_bonus

        # heuristic early-stop: keep at most 3 × k candidates
        if len(scores) >= k * 3:
            break

    # 6) Sort and return top-k
    return sorted(scores.items(), key=lambda x: -x[1])[:k]

###############################################################################
#  CLI
###############################################################################

def cli(nlp, linker):
    """
    Interactive CLI:
      ingest → build corpus and indices
      search → query hybrid engine
    """
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("--synthea", default="synthea/csv")
    p_ing.add_argument("--pubmed",  default="pubmed_rct/train.csv")

    p_s = sub.add_parser("search")
    p_s.add_argument("prompt")
    p_s.add_argument("--trial_limit", type=int, default=200)
    p_s.add_argument("--k", type=int, default=20)

    args = parser.parse_args()

    # ----------------------- INGEST MODE -----------------------
    if args.cmd == "ingest":
        txt, ids = pre_ingest(args.synthea, args.pubmed)
        cui_docs = compute_cui_docs(txt, nlp)
        np.save(DOC_CUIS_NPY, np.array(cui_docs))
        np.save(DOC_IDS_NPY, np.array(ids))
        model = SentenceTransformer(MODEL_NAME)
        build_vector_index(txt, model)
        model.save(MODEL_PATH)
        pd.DataFrame({"id": ids, "text": txt}).to_csv(CORPUS_CSV, index=False)
        print("Ingestion finished.")
        return

    # ----------------------- SEARCH MODE -----------------------
    if not Path(CORPUS_CSV).exists():
        print("No corpus found – running default ingest first …")
        txt, ids = pre_ingest()
        cui_docs = compute_cui_docs(txt, nlp)
        np.save(DOC_CUIS_NPY, np.array(cui_docs))
        np.save(DOC_IDS_NPY, np.array(ids))
        model = SentenceTransformer(MODEL_NAME)
        build_vector_index(txt, model)
        model.save(MODEL_PATH)
        pd.DataFrame({"id": ids, "text": txt}).to_csv(CORPUS_CSV, index=False)

    # load persistent artefacts
    df   = pd.read_csv(CORPUS_CSV).fillna("")
    txts = df["text"].astype(str).tolist()
    ids  = df["id"].tolist()
    cuis = np.load(DOC_CUIS_NPY)
    model = SentenceTransformer(MODEL_PATH)

    # Live ingestion of relevant clinical trials
    ct_txt, ct_ids = ingest_trials(nlp, linker, args.prompt, args.trial_limit)
    ct_cuis        = compute_cui_docs(ct_txt, nlp)
    build_vector_index(ct_txt + txts, model)

    bm = bm25_scores(ct_cuis + list(cuis), args.prompt, nlp)
    dense = dense_scores(args.prompt, model)
    fused = rrf(bm, dense, ct_ids + ids, k=args.k)

    # pretty print
    lookup = {**dict(zip(ids, txts)), **dict(zip(ct_ids, ct_txt))}
    print("\nTop results")
    results = []
    for did, score in fused:
        snippet = lookup[did][:120].replace("\n", " ")
        results.append(snippet)
        print(f"{did:20}  RRF={score:.3f}  → {snippet}…")
    return results

###############################################################################
#  ENTRY-POINT
###############################################################################

def get_documents_hybrid_search(query: str):
    """
    Convenience wrapper for external calls (e.g. via Flask API).

    Args
    ----
    query : str
        Natural-language question.

    Returns
    -------
    List[str]
        Snippets returned by hybrid search.
    """
    
    nlp, linker = load_umls_pipeline()
    if len(sys.argv) == 1:
        sys.argv.extend(["search", query])
    return cli(nlp, linker)
