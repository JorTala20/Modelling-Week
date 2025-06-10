import pandas as pd
from pathlib import Path

def cargar_tablas_synthea(directorio: str) -> dict:
    ruta = Path(directorio)
    tablas = {}
    for archivo in ruta.glob("*.csv"):
        nombre = archivo.stem  # por ejemplo: patients
        tablas[nombre] = pd.read_csv(archivo)
    return tablas


def resumen_paciente(t: dict, patient_id: str) -> dict:
    fila = t["patients"].query("Id == @patient_id").iloc[0]
    demografia = {
        "id": patient_id,
        "sexo": fila["GENDER"],
        "nacimiento": fila["BIRTHDATE"],
        "raza": fila["RACE"]
    }

    dx = t["conditions"].query("PATIENT == @patient_id")[["DESCRIPTION", "START"]]
    medicamentos = t["medications"].query("PATIENT == @patient_id and STOP.isna()")["DESCRIPTION"]

    return {
        "demografia": demografia,
        "diagnosticos": dx.to_dict(orient="records"),
        "medicacion_activa": medicamentos.tolist()
    }

if __name__ == "__main__":
    tablas = cargar_tablas_synthea("synthea/csv")
    paciente_aleatorio = tablas["patients"]["Id"].sample(1).item()
    resumen = resumen_paciente(tablas, paciente_aleatorio)

    print("=== Resumen clínico ===")
    print("Demografía:", resumen["demografia"])
    print("\nDiagnósticos:")
    for d in resumen["diagnosticos"]:
        print(f" - {d['START']}: {d['DESCRIPTION']}")
    print("\nFármacos activos:", ", ".join(resumen["medicacion_activa"]) or "ninguno")


import requests

def buscar_trials_v2(palabra_clave: str, max_resultados: int = 5) -> list:
    url = "https://clinicaltrials.gov/api/v2/studies"

    params = {
        "query.term": palabra_clave,     # término de búsqueda libre
        "pageSize": max_resultados       # número de resultados
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print("❌ Error al consultar la nueva API:", e)
        return []

    data = response.json()
    estudios = data.get("studies", [])

    resultados = []
    for est in estudios:
        resultados.append({
            "nct_id": est.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "Sin ID"),
            "titulo": est.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", "Sin título"),
            "condicion": est.get("protocolSection", {}).get("conditionsModule", {}).get("conditions", []),
            "estado": est.get("protocolSection", {}).get("statusModule", {}).get("overallStatus", "Desconocido"),
        })

    return resultados


print("-"*100)

if __name__ == "__main__":
    enfermedad = "duchenne"
    resultados = buscar_trials_v2(enfermedad, max_resultados=5)

    if resultados:
        print(f"\nEnsayos encontrados para: {enfermedad.upper()}")
        for ensayo in resultados:
            print(f"- {ensayo['nct_id']} [{ensayo['estado']}]: {ensayo['titulo'][:90]}...")
    else:
        print("No se encontraron ensayos o hubo un error.")

print("-"*100)

import spacy
from urllib.parse import quote
import requests

# Cargar modelo biomédico de scispaCy
nlp = spacy.load("en_core_sci_sm")

# Extraer entidades (tipo "ENTITY", no usamos UMLS)
def extraer_entidades(texto: str) -> list:
    doc = nlp(texto)
    return [ent.text for ent in doc.ents if ent.label_ in ["ENTITY"]]

# Buscar ensayos clínicos en la API v2 de ClinicalTrials.gov
def buscar_trials_por_entidad(entidad: str, max_resultados=5):
    query = quote(entidad)
    url = f"https://clinicaltrials.gov/api/v2/studies?query.term={query}&pageSize={max_resultados}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"❌ Error al buscar ensayos para '{entidad}':", e)
        return

    estudios = data.get("studies", [])
    if not estudios:
        print(f"⚠️  No se encontraron ensayos para: {entidad}")
        return

    print(f"\nEnsayos para: {entidad}")
    for estudio in estudios:
        try:
            titulo = estudio["protocolSection"]["identificationModule"]["briefTitle"]
            estado = estudio["protocolSection"]["statusModule"]["overallStatus"]
            nct_id = estudio["protocolSection"]["identificationModule"]["nctId"]
            print(f"- {nct_id} [{estado}]: {titulo[:90]}...")
        except KeyError:
            continue

# MAIN
def main():
    texto = "The patient has Duchenne muscular dystrophy and type 1 diabetes."
    print("🔍 Analizando texto clínico:")
    print(texto)

    entidades = extraer_entidades(texto)

    if entidades:
        print("\n✅ Entidades biomédicas detectadas:")
        for e in entidades:
            print(f" - {e}")
    else:
        print("\n⚠️  No se detectaron entidades biomédicas.")
        return

    for entidad in entidades:
        buscar_trials_por_entidad(entidad, max_resultados=5)

if __name__ == "__main__":
    main()
