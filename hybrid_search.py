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
DIM, MODEL_NAME = 384, "sentence-transformers/all-MiniLM-L6-v2"
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

def crear_pipeline_umls():
    """Carga el modelo de spacy y añade el linker de UMLS."""
    print("Cargando modelo de spaCy y linker UMLS...")
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe(
        "scispacy_linker",
        last=True,
        config={
            "linker_name": "umls",
            "resolve_abbreviations": True
        },
    )
    linker = nlp.get_pipe("scispacy_linker")
    print("¡Modelo cargado con éxito!")
    return nlp, linker

def extraer_cui(texto: str, nlp) -> dict:
    """Extrae entidades médicas, sus CUIs y sinónimos de un texto."""
    doc = nlp(texto)
    resultados = []
    for ent in doc.ents:
        kb_ents = getattr(ent._, "kb_ents", None)
        if not kb_ents:
            continue
        cui = kb_ents[0][0]
        resultados.append(cui)
    return resultados

# --- 1. Función de Normalización del Prompt (Nueva) ---

def normalizar_prompt_a_cui(texto_paciente: str, nlp) -> str:
    """
    Toma un texto, detecta entidades médicas y las reemplaza por su CUI.

    Args:
        texto_paciente: El perfil del paciente en lenguaje natural.
        nlp: El pipeline de spaCy cargado con el linker UMLS.

    Returns:
        Un string donde las entidades médicas han sido sustituidas por sus CUIs.
    """
    
    doc = nlp(texto_paciente)
    texto_modificado = texto_paciente
    
    # Obtenemos las entidades y las ordenamos por su posición inicial en orden inverso
    # Esto es CRÍTICO para que los reemplazos no invaliden los índices de las siguientes entidades
    entidades_orden_inverso = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
    
    entidades_encontradas = 0
    for ent in entidades_orden_inverso:
        # Comprobamos si la entidad tiene candidatos en la base de conocimiento (UMLS)
        if hasattr(ent._, "kb_ents") and ent._.kb_ents:
            # Seleccionamos el CUI del primer candidato (el de mayor probabilidad)
            cui = ent._.kb_ents[0][0]
            
            # Reemplazamos el texto de la entidad por su CUI en la cadena
            inicio = ent.start_char
            fin = ent.end_char
            texto_modificado = texto_modificado[:inicio] + cui + texto_modificado[fin:]
            entidades_encontradas += 1

    return texto_modificado

# ──────────────────── Ingesta Synthea CSV ─────────────────────────────────
def ingest_synthea_csv(dir_csv: str):
    
    def safe(val, slice_len=None):
        if pd.notna(val):
            val_str = str(val)
            return val_str[:slice_len] if slice_len else val_str
        return "unknown"

    patients = pd.read_csv(Path(dir_csv) / "patients.csv", dtype=str)
    cond = pd.read_csv(Path(dir_csv) / "conditions.csv", dtype=str)
    allergies = pd.read_csv(Path(dir_csv) / "allergies.csv", dtype=str)
    care = pd.read_csv(Path(dir_csv) / "careplans.csv", dtype=str)
    dev = pd.read_csv(Path(dir_csv) / "devices.csv", dtype=str)
    encounters = pd.read_csv(Path(dir_csv) / "encounters.csv", dtype=str)
    imaging = pd.read_csv(Path(dir_csv) / "imaging_studies.csv", dtype=str)
    immun = pd.read_csv(Path(dir_csv) / "immunizations.csv", dtype=str)
    meds = pd.read_csv(Path(dir_csv) / "medications.csv", dtype=str)
    obs = pd.read_csv(Path(dir_csv) / "observations.csv", dtype=str)
    proc = pd.read_csv(Path(dir_csv) / "procedures.csv", dtype=str)

    textos, ids = [], []
    for _, p in patients.iterrows():
        pid = p["Id"]
        textos.append(f"Patient {safe(p['GENDER'])} born {safe(p['BIRTHDATE'], 10)} race {safe(p['RACE'])}")
        ids.append(f"ehr_{pid}_demo")

        for idx, c in cond[cond["PATIENT"] == pid].iterrows():
            textos.append(f"Condition {safe(c['DESCRIPTION'])} started {safe(c['START'], 10)}")
            ids.append(f"ehr_{pid}_cond_{idx}")

        for idx, e in encounters[encounters["PATIENT"] == pid].iterrows():
            textos.append(f"Encounter {safe(e['ENCOUNTERCLASS'])} for {safe(e['DESCRIPTION'])} from {safe(e['START'], 10)} to {safe(e['STOP'], 10)}")
            ids.append(f"ehr_{pid}_enc_{idx}")

        for idx, a in allergies[allergies["PATIENT"] == pid].iterrows():
            textos.append(f"Allergy to {safe(a['DESCRIPTION'])} recorded on {safe(a['START'], 10)}")
            ids.append(f"ehr_{pid}_alg_{idx}")

        for idx, c in care[care["PATIENT"] == pid].iterrows():
            textos.append(f"Care plan: {safe(c['DESCRIPTION'])} from {safe(c['START'], 10)} to {safe(c['STOP'], 10)}")
            ids.append(f"ehr_{pid}_care_{idx}")

        for idx, d in dev[dev["PATIENT"] == pid].iterrows():
            textos.append(f"Device: {safe(d['DESCRIPTION'])} implanted on {safe(d['START'], 10)}")
            ids.append(f"ehr_{pid}_dev_{idx}")

        for idx, im in imaging[imaging["PATIENT"] == pid].iterrows():
            textos.append(f"Imaging study: {safe(im['BODYSITE_DESCRIPTION'])} on {safe(im['DATE'], 10)}")
            ids.append(f"ehr_{pid}_img_{idx}")

        for idx, im in immun[immun["PATIENT"] == pid].iterrows():
            textos.append(f"Immunization: {safe(im['DESCRIPTION'])} on {safe(im['DATE'], 10)}")
            ids.append(f"ehr_{pid}_imm_{idx}")

        for idx, m in meds[meds["PATIENT"] == pid].iterrows():
            textos.append(f"Medication: {safe(m['DESCRIPTION'])} from {safe(m['START'], 10)} to {safe(m['STOP'], 10)}")
            ids.append(f"ehr_{pid}_med_{idx}")

        for idx, o in obs[obs["PATIENT"] == pid].iterrows():
            textos.append(f"Observation: {safe(o['DESCRIPTION'])} = {safe(o['VALUE'])} on {safe(o['DATE'], 10)}")
            ids.append(f"ehr_{pid}_obs_{idx}")

        for idx, pr in proc[proc["PATIENT"] == pid].iterrows():
            textos.append(f"Procedure: {safe(pr['DESCRIPTION'])} on {safe(pr['DATE'], 10)}")
            ids.append(f"ehr_{pid}_proc_{idx}")

    print(f"✔️  Synthea fragmentos: {len(textos)}")
    return textos, ids


# ──────────────────── Ingesta PubMed-RCT CSV ──────────────────────────────
def ingest_pubmed_csv(csv_path: str):
    df = pd.read_csv(csv_path, dtype=str)

    # Verifica que existen las columnas requeridas
    if "abstract_text" not in df.columns or "target" not in df.columns:
        raise ValueError("El CSV debe contener las columnas 'abstract_text' y 'target'.")

    # Elimina filas con texto o target vacío
    df = df[df["abstract_text"].notna() & df["target"].notna()]

    # Asegura que todo es texto plano
    df["abstract_text"] = df["abstract_text"].astype(str)
    df["target"] = df["target"].astype(str)

    # Concatena target al texto → formato: "TARGET: texto..."
    textos = [f"{row['target']}: {row['abstract_text']}" for _, row in df.iterrows()]
    ids    = [f"pm_{i}" for i in range(len(textos))]

    print(f"✔️  PubMed-RCT fragmentos: {len(textos)}")
    return textos, ids


# ──────────────────── Ingesta ClinicalTrials.gov ───────────────────────────
import requests
from urllib.parse import quote
from itertools import combinations
from typing import List, Tuple
from itertools import combinations, product
from urllib.parse import quote
import requests
from typing import List, Tuple
from itertools import chain

def synonim_cui(cui: str, linker) -> List[str]:
    """
    Devuelve una lista de sinónimos (incluyendo el nombre canónico)
    para un CUI usando linker.kb.
    """
    entity = linker.kb.cui_to_entity.get(cui)
    if not entity:
        return [cui]  # fallback si no se encuentra el CUI

    # Combina nombre canónico + alias únicos
    sinonimos = set(entity.aliases)
    sinonimos.add(entity.canonical_name)
    return list(sinonimos)


def ingest_clinical_trials(nlp, linker, terms: List[str], limit: int = 200):
    """
    terms = lista de descripciones en lenguaje natural, ej. ['lung cancer','ibuprofen']
    Implementa búsqueda jerárquica: primero term[0]; luego filtra con term[1:], etc.
    """
    def consulta(cui, linker):
        qs = synonim_cui(cui, linker)
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
            chain.from_iterable(synonim_cui(t, linker)[:2] for t in terms[1:]))
    )
    for t in primarios:
        if any(r in t for r in resto):
            filtrados.append(t)

    finales = filtrados if filtrados else primarios
    return [(f"{' & '.join(terms)}", finales[:limit])]



# ──────────────────── Ingesta unificada ────────────────────────────────────
TUI_PRIORIDAD = {
    "T047": 1,  # Disease or Syndrome
    "T191": 1,  # Neoplastic Process
    "T121": 2,  # Pharmacologic Substance
    "T200": 2,  # Clinical Drug
    "T061": 3,  # Therapeutic/Preventive Procedure
    "T060": 3,  # Diagnostic Procedure
    "T123": 4,  # Biologically Active Substance
    "T109": 5,  # Organic Chemical
    "T103": 6,  # Chemical
    "T074": 7,  # Medical Device
    "T170": 8,  # Intellectual Product
    "T078": 9   # Idea or Concept
}

def prioridad_cui(cui: str, linker) -> int:
    entity = linker.kb.cui_to_entity.get(cui)
    if not entity or not entity.types:
        return 999  # máximo si no tiene TUI
    return min(TUI_PRIORIDAD.get(tui, 999) for tui in entity.types)


def ingest_trials(nlp, linker, trial_term: str, trial_limit: int = 10):

    # extrae CUIs ➜ descripciones
    cuis = extraer_cui(trial_term, nlp)
    terms = sorted(cuis, key=lambda c: prioridad_cui(c, linker))

    res_trials = ingest_clinical_trials(nlp, linker, terms, trial_limit)


    trial_textos = []
    trial_ids = []
    for i, (query, titles) in enumerate(res_trials):
        for j, title in enumerate(titles):
            trial_textos.append(title)
            trial_ids.append(f"ct_{i}_{j}")

    return trial_textos, trial_ids

def pre_ingest(synthea_path: str, pubmed_path: str):
    textos, ids = [], []

    textos += (ts := ingest_synthea_csv(synthea_path))[0];   ids += ts[1]
    textos += (tp := ingest_pubmed_csv(pubmed_path))[0];     ids += tp[1]

    return textos, ids

# ──────────────────── Vectorización / Índice ───────────────────────────────
def build_vector_index(textos: List[str], ids: List[str]):
    model = SentenceTransformer(MODEL_NAME)
    emb   = model.encode(textos, normalize_embeddings=True)
    
    if _HAS_FAISS:
        idx = faiss.IndexFlatIP(DIM); idx.add(emb.astype(np.float32))
        faiss.write_index(idx, VEC_FAISS); print("✅ FAISS guardado")
    elif _HAS_HNSW:
        p = hnswlib.Index(space="cosine", dim=DIM)
        p.init_index(max_elements=len(emb), ef_construction=200, M=16)
        p.add_items(emb, np.arange(len(emb))); p.save_index(VEC_HNSW)
        print("✅ HNSW guardado")
    else:
        raise ImportError("Instala faiss-cpu o hnswlib")

    return model

# ──────────────────── BM25 local ───────────────────────────────────────────
def get_cui_docs(textos, nlp):
    cleaned = [str(t).lower() for t in textos]

    # Aplica normalización a CUI para cada documento
    cui_docs = [normalizar_prompt_a_cui(texto, nlp) for texto in cleaned]
    return cui_docs

def bm25_scores(cui_docs: List[str], query: str, nlp, min_overlap: int = 2):
    # docs → lista-de-listas de tokens
    docs_tok = [d.split() for d in cui_docs]

    # query normalizada + tokens
    q_tok = normalizar_prompt_a_cui(query.lower(), nlp).split()

    # → descarta docs con < min_overlap CUIs de la query  ← NUEVO
    mask = [sum(t in d for t in q_tok) >= min_overlap for d in docs_tok]
    docs_filt = [d for d, keep in zip(docs_tok, mask) if keep]

    if not docs_filt:                                 # fallback suave
        docs_filt, mask = docs_tok, [True]*len(docs_tok)

    bm25 = BM25Okapi(docs_filt)
    scores = bm25.get_scores(q_tok)

    # Rellenamos con 0 los descartados para mantener índice estable
    full = np.zeros(len(cui_docs))
    full[np.nonzero(mask)] = scores
    return full



# ──────────────────── Vector search local ──────────────────────────────────
def vec_scores(query: str, model: SentenceTransformer, top_n=180):
    q = model.encode([query], normalize_embeddings=True).astype(np.float32)
    ids = np.load(DOC_IDS_NPY)

    if Path(VEC_FAISS).exists() and _HAS_FAISS:
        idx = faiss.read_index(VEC_FAISS)
        D, I = idx.search(q, top_n)
        sims = 1 - D[0]
        return {
            str(ids[i]): float(sims[j])
            for j, i in enumerate(I[0])
            if i < len(ids)
        }

    # HNSW fallback
    idx = hnswlib.Index(space="cosine", dim=DIM)
    idx.load_index(VEC_HNSW)
    lab, dist = idx.knn_query(q, k=top_n)
    sims = 1 - dist[0]
    return {
        str(ids[i]): float(sims[j])
        for j, i in enumerate(lab[0])
        if i < len(ids)
    }


# ──────────────────── RRF fusion ───────────────────────────────────────────
def rrf_fuse(bm25, vec, doc_ids, k=20, rrf_k=25):
    scr = {}
    # BM25 ranks
    for rk, idx in enumerate(np.argsort(bm25)[::-1], 1):
        scr[doc_ids[idx]] = scr.get(doc_ids[idx], 0) + 1/(rrf_k+rk)
    # Vector ranks
    for rk, doc in enumerate(sorted(vec, key=vec.get, reverse=True), 1):
        scr[doc] = scr.get(doc, 0) + 1/(rrf_k+rk)
    return sorted(scr.items(), key=lambda x: x[1], reverse=True)[:k]

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
        cui_docs = get_cui_docs(txt, nlp)
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
    
    ct_txt, ct_ids = ingest_trials(nlp, linker,
                  "35-year-old male with lung cancer and ibuprofen 3 times per week",
                  10000)
    ct_cuis = get_cui_docs(ct_txt, nlp)
    build_vector_index(ct_txt + txt, ct_ids + ids)
     
    model = SentenceTransformer(MODEL_NAME)
    print("modelado")
    bm = bm25_scores(list(ct_cuis) + list(cui_docs), args.prompt, nlp)
    print("BMeado")
    vec   = vec_scores(args.prompt, model)
    print("scoreado")
    res   = rrf_fuse(bm, vec, ct_ids + ids, k=args.k)
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
if __name__ == "__main__":
    nlp, linker = crear_pipeline_umls()
    if len(sys.argv) == 1:
        sys.argv.extend(["search", "35-year-old male with lung cancer and ibuprofen 3 times per week"])
    cli(nlp, linker)
