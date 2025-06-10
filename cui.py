import spacy
import scispacy                    # importa el paquete principal
import scispacy.umls_linking       # <-- añade esta línea: registra 'scispacy_linker'
from urllib.parse import quote
import requests
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def crear_pipeline_umls():
    nlp = spacy.load("en_core_sci_sm")
    # ahora la fábrica 'scispacy_linker' ya está registrada
    nlp.add_pipe(
        "scispacy_linker",
        last=True,
        config={
            "linker_name": "umls",
            "resolve_abbreviations": True
        },
    )
    linker = nlp.get_pipe("scispacy_linker")
    return nlp, linker

def extraer_cui_y_sinonimos(texto: str, nlp, linker) -> dict:
    """
    Procesa el texto con nlp, itera sobre doc.ents y para cada entidad
    con ent._.kb_ents no vacío extrae:
      - CUI = ent._.kb_ents[0][0]
      - sinónimos = lista de aliases desde linker.kb.cui_to_entity[cui].aliases
    Devuelve dict: {ent.text: {"cui": ..., "sinonimos": [...]}, ...}
    """
    doc = nlp(texto)
    resultados = {}
    for ent in doc.ents:
        # comprobamos si hay candidatos UMLS
        kb_ents = getattr(ent._, "kb_ents", None)
        if not kb_ents:
            continue
        cui = kb_ents[0][0]
        # extraer sinónimos de la KB
        entity = linker.kb.cui_to_entity.get(cui)
        if not entity:
            continue
        sinonimos = list(set(entity.aliases))
        resultados[ent.text] = {"cui": cui, "sinonimos": sinonimos}
    return resultados

def buscar_trials_por_sinonimos(sinonimos: list, limite: int = 5):
    """
    Forma una query OR con hasta 5 sinónimos y consulta la API v2 de ClinicalTrials.gov.
    Imprime los resultados en consola.
    """
    if not sinonimos:
        print("⚠️  No hay sinónimos para buscar.")
        return
    # tomamos los primeros 5 sinónimos para no crear URL excesivamente larga
    terms = [f'"{s}"' for s in sinonimos[:5]]
    query = " OR ".join(terms)
    url = f"https://clinicaltrials.gov/api/v2/studies?query.term={quote(query)}&pageSize={limite}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        estudios = r.json().get("studies", [])
    except Exception as e:
        print(f"❌ Error al consultar ClinicalTrials para query {query}: {e}")
        return
    if not estudios:
        print(f"⚠️  Sin resultados para: {query}")
        return
    print(f"\n🔎 Ensayos encontrados para: {query}")
    for est in estudios:
        mod = est.get("protocolSection", {})
        id_mod = mod.get("identificationModule", {})
        status_mod = mod.get("statusModule", {})
        nct = id_mod.get("nctId", "?")
        title = id_mod.get("briefTitle", "").strip()
        estado = status_mod.get("overallStatus", "Desconocido")
        print(f"- {nct} [{estado}]: {title[:90]}...")
    print()

def main(nlp, linker):
    print(">>> Entrando en main()")  # debug
    #nlp, linker = crear_pipeline_umls()
    print(">>> Pipeline cargado")    # debug

    texto = "lung cancer and type 1 diabetes."
    print("Texto clínico analizado:\n", texto, "\n")

    entidades = extraer_cui_y_sinonimos(texto, nlp, linker)
    print(f">>> Entidades extraídas: {entidades}")  # debug

    if not entidades:
        print("⚠️  No se detectaron entidades médicas con CUI.")
        return

    for nombre, info in entidades.items():
        print(f"Entidad detectada: {nombre}")
        print(f"   • CUI       : {info['cui']}")
        muestra = info["sinonimos"][:5]
        print(f"   • Sinónimos : {', '.join(muestra)}…")
        buscar_trials_por_sinonimos(info["sinonimos"], limite=5)
    print(">>> Fin de main()")  # debug
