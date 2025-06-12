from __future__ import annotations
"""
data_pipeline_hybrid.py · Versión local · sin Elasticsearch
-----------------------------------------------------------
• Ingesta unificada:  Synthea (+patients/+conditions),
                      PubMed-RCT (Kaggle TSV),
                      ClinicalTrials.gov (API v2).
• Vectorización con sentence-transformers/MiniLM-L6-v2.
• Índice vectorial FAISS  (o HNSW si FAISS no está disponible).
• BM25 local (rank_bm25) + Vector search  +  RRF.
"""

# ───────────────────────── Imports ─────────────────────────────────────────
import argparse, csv, json, os, sys
from pathlib import Path
from typing import List, Dict

import requests
from urllib.parse import quote
from itertools import combinations
from typing import List, Tuple
from itertools import combinations, product
from urllib.parse import quote
import requests
from typing import List, Tuple
from itertools import chain

import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from urllib.parse import quote
import requests

try:
    import faiss;  _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
try:
    import hnswlib; _HAS_HNSW = True
except ImportError:
    _HAS_HNSW = False

# ─────────────────── Parámetros / ficheros ────────────────────────────────
DOC_IDS_NPY, VEC_FAISS, VEC_HNSW = "doc_ids.npy", "vectors.index", "hnsw_index.bin"
DOC_CUIS_NPY = "doc_cuis.npy"
DIM, MODEL_NAME = 384, "sentence-transformers/all-mpnet-base-v2"
CORPUS_CSV = "corpus.csv"

import spacy
import scispacy
import scispacy.umls_linking
from urllib.parse import quote
import requests
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import random

# Ignorar advertencias de versión de sklearn que a veces surgen con scispacy
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# --- Funciones proporcionadas por el usuario (sin cambios) ---

def load_umls_pipeline():
    """Create spaCy pipeline + scispaCy UMLS linker."""
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

def extract_cuis(text: str, nlp, linker) -> List[str]:
    """Return CUIs found in *text* (first candidate of each entity)."""
    return [ent._.kb_ents[0][0]
            for ent in nlp(text).ents
            if getattr(ent._, "kb_ents", None)]

# --- 1. Función de Normalización del Prompt (Nueva) ---

def prompt_to_cuis(text: str, nlp) -> str:
    """
    Replace each recognised medical entity in *text* with its CUI.
    Leaving non-medical tokens intact preserves context for BM25.
    """
    doc = nlp(text)
    out = text
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        if ent._.kb_ents:
            cui = ent._.kb_ents[0][0]
            out = out[:ent.start_char] + cui + out[ent.end_char:]
    return out

# ──────────────────── Ingesta Synthea CSV ─────────────────────────────────
def ingest_synthea_csv(csv_dir: str) -> Tuple[List[str], List[str]]:
    
    def safe(val, slice_len=None):
        if pd.notna(val):
            val_str = str(val)
            return val_str[:slice_len] if slice_len else val_str
        return "unknown"

    """Flatten multiple Synthea CSVs into plain text fragments."""
    p = Path(csv_dir)
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
    
    texts, ids = [], []
    for _, pat in patients.iterrows():
        pid = pat["Id"]
        texts.append(f"Patient {safe(pat['GENDER'])} born {safe(pat['BIRTHDATE'],10)} "
                     f"race {safe(pat['RACE'])}")
        ids.append(f"ehr_{pid}_demo")

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

    print(f"✔️  Synthea fragments: {len(texts)}")
    return texts, ids

# ──────────────────── Ingesta PubMed-RCT CSV ──────────────────────────────
def ingest_pubmed_csv(csv_path: str) -> Tuple[List[str], List[str]]:
    """PubMed-RCT file → one text fragment per sentence + target label."""
    df = pd.read_csv(csv_path, dtype=str)
    if {"abstract_text", "target"} - set(df.columns):
        raise ValueError("CSV must have columns 'abstract_text' and 'target'")
    df = df[df["abstract_text"].notna() & df["target"].notna()]
    
    # Asegura que todo es texto plano
    df["abstract_text"] = df["abstract_text"].astype(str)
    df["target"] = df["target"].astype(str)
    
    texts = [f"{row['target']}: {row['abstract_text']}" for _, row in df.iterrows()]
    ids   = [f"pm_{i}" for i in range(len(texts))]
    print(f"✔️  PubMed-RCT fragments: {len(texts)}")
    return texts, ids

def pre_ingest(synthea_path: str = "synthea/csv", pubmed_path: str = "pubmed_rct/train.csv"):
    """
    Loads and combines the Synthea and PubMed-RCT datasets.

    Args:
        synthea_path (str): Path to the Synthea CSV directory.
        pubmed_path (str): Path to the PubMed-RCT CSV file.

    Returns:
        Tuple[List[str], List[str]]: Combined list of texts and their corresponding IDs.
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

# ──────────────────── Ingesta ClinicalTrials.gov ───────────────────────────
def cui_synonyms(cui: str, linker) -> List[str]:
    """Canonical name + aliases (deduplicated)."""
    ent = linker.kb.cui_to_entity.get(cui)
    if not ent:
        return [cui]
    return list(set(ent.aliases) | {ent.canonical_name})


def ingest_clinical_trials(nlp, linker, terms: List[str], limit: int = 200):
    """
    terms = lista de descripciones en lenguaje natural, ej. ['lung cancer','ibuprofen']
    Implementa búsqueda jerárquica: primero term[0]; luego filtra con term[1:], etc.
    """
    def consulta(cui, linker):
        qs = cui_synonyms(cui, linker)
        numero = int(len(qs)/4)+1
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

    # 1) Primera capa: term principal
    primarios = consulta(terms[0], linker)

    # 2) Filtra por los términos restantes
    filtrados = []

    resto = list(
        map(lambda x: x.lower(),
            chain.from_iterable(cui_synonyms(t, linker)[:2] for t in terms[1:]))
    )
    for t in primarios:
        if any(r in t for r in resto):
            filtrados.append(t)

    finales = filtrados if filtrados else primarios
    return [(f"{' & '.join(terms)}", finales[:limit])]



# ──────────────────── Ingesta unificada ────────────────────────────────────
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
    ent = linker.kb.cui_to_entity.get(cui)
    if not ent or not ent.types:
        return 999
    return min(TUI_PRIORITY.get(t, 999) for t in ent.types)


def ingest_trials(
    nlp,
    linker,
    query_text: str,
    limit: int = 100,
) -> Tuple[List[str], List[str]]:
    cuis = sorted(extract_cuis(query_text, nlp, linker),
                  key=lambda c: cui_priority(c, linker))
    trials = ingest_clinical_trials(nlp, linker, cuis, limit)

    t_texts, t_ids = [], []
    for block, (label, titles) in enumerate(trials):
        for i, title in enumerate(titles):
            t_texts.append(title)
            t_ids.append(f"ct_{block}_{i}")
    return t_texts, t_ids

# ──────────────────── Vectorización / Índice ───────────────────────────────
def build_vector_index(texts: List[str], ids: List[str]) -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME)
    vecs  = model.encode(texts, normalize_embeddings=True)

    if _HAS_FAISS:
        idx = faiss.IndexFlatIP(DIM)
        idx.add(vecs.astype(np.float32))
        faiss.write_index(idx, VEC_FAISS)
        print("✅ FAISS index saved")
    elif _HAS_HNSW:
        idx = hnswlib.Index(space="cosine", dim=DIM)
        idx.init_index(len(vecs), ef_construction=200, M=16)
        idx.add_items(vecs, np.arange(len(vecs)))
        idx.save_index(VEC_HNSW)
        print("✅ HNSW index saved")
    else:
        raise ImportError("Install faiss-cpu or hnswlib")

    np.save(DOC_IDS_NPY, np.array(ids))
    return model

# ──────────────────── BM25 local ───────────────────────────────────────────
def compute_cui_docs(texts: List[str], nlp) -> List[str]:
    """Return each document as space-separated CUIs."""
    return [prompt_to_cuis(t.lower(), nlp) for t in texts]

def bm25_scores(
    docs_as_cuis: List[str],
    query: str,
    nlp,
    min_overlap: int = 1,
) -> np.ndarray:
    docs_tokenised = [d.split() for d in docs_as_cuis]
    query_tokens   = prompt_to_cuis(query.lower(), nlp).split()

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

# ──────────────────── Vector search local ──────────────────────────────────
def dense_scores(query: str, model: SentenceTransformer, top_n=150) -> Dict[str, float]:
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

# =============================================================================
#  Reciprocal Rank Fusion
# =============================================================================
def rrf(
    bm25_vec: np.ndarray,
    dense_dict: Dict[str, float],
    doc_ids: List[str],
    k: int = 20,
    rrf_k: int = 10,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    # BM25 contribution
    nz = [i for i, s in enumerate(bm25_vec) if s > 0]
    for rank, idx in enumerate(sorted(nz, key=lambda i: bm25_vec[i], reverse=True), 1):
        did = doc_ids[idx]
        scores[did] = scores.get(did, 0) + 1 / (rrf_k + rank)

    # Dense contribution (only where BM25 > 0)
    rank = 1
    for did, _ in sorted(dense_dict.items(), key=lambda x: x[1], reverse=True):
        try:
            idx = doc_ids.index(did)
        except ValueError:
            continue
        
        scores[did] = scores.get(did, 0) + 1 / (rrf_k + rank)
        rank += 1

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

# ──────────────────── CLI ──────────────────────────────────────────────────
def cli(nlp, linker):
    argp = argparse.ArgumentParser()
    sub  = argp.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("--synthea", default="synthea/csv")
    p_ing.add_argument("--pubmed",  default="pubmed_rct/train.csv")
    p_ing.add_argument("--trial",   default="cancer")
    p_ing.add_argument("--trial_limit", type=int, default=100)

    p_s = sub.add_parser("search")
    p_s.add_argument("prompt")
    p_s.add_argument("--k", type=int, default=20)

    args = argp.parse_args()


    if not Path(CORPUS_CSV).exists():
        print("🔄 No corpus.csv. Ejecutando ingesta rápida...")
        txt, ids = pre_ingest("synthea/csv", "pubmed_rct/train.csv")
        print("ingestado")
        cui_docs = compute_cui_docs(txt, nlp)
        print("cuidado")
        
        np.save(DOC_CUIS_NPY, np.array(cui_docs))
        np.save(DOC_IDS_NPY, np.array(ids))
        pd.DataFrame({"id": ids, "text": txt}).to_csv(CORPUS_CSV, index=False)
        print("💾 corpus.csv generado")
        # txt, ids = ingest_all(nlp, linker, "synthea/csv", "pubmed_rct/train.csv", "35-year-old male with lung cancer and ibuprofen 3 times per week", 200)
        # build_vector_index(txt, ids)
    df = pd.read_csv(CORPUS_CSV).fillna("")        # ← elimina NaN
    txt = df["text"].astype(str).tolist()        # ← forzamos str
    ids = df["id"].tolist()
    cui_docs = np.load(DOC_CUIS_NPY)
    
    query = "35-year-old male with lung cancer and ibuprofen 3 times per week"
    ct_txt, ct_ids = ingest_trials(nlp, linker, query, limit=100)
    print(f"✔️  ClinicalTrials.gov fragments: {len(ct_txt)}")
    
    ct_cuis = compute_cui_docs(ct_txt, nlp)
    build_vector_index(ct_txt + txt, ct_ids + ids)
     
    model = SentenceTransformer(MODEL_NAME)
    print("modelado")
    bm = bm25_scores(list(ct_cuis) + list(cui_docs), args.prompt, nlp)
    print("BMeado")
    vec   = dense_scores(args.prompt, model)
    print("scoreado")
    res   = rrf(bm, vec, ct_ids + ids, k=args.k)
    print("rrfuseado")
    text_lookup = {**dict(zip(ids, txt)), **dict(zip(ct_ids, ct_txt))}
    
    print("\n📄 Top resultados:")
    snippets = []
    for did, sc in res:
        snippet = text_lookup.get(did, "")[:120].replace("\n", " ")
        snippets.append(snippet)
        print(f"{did:15}  score={sc:.3f}  → {snippet}...")

    return [r + (s,) for r, s in zip(res, snippets)]

# ──────────────────── Auto-ejecución ───────────────────────────────────────
if __name__ != "__main__":
    #nlp, linker = load_umls_pipeline()
    if len(sys.argv) == 1:
        sys.argv.extend(["search", "35-year-old male with lung cancer and ibuprofen 3 times per week"])
    cli(nlp, linker)
