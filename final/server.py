import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from utils_server import *
from hs_nuevo_server import load_umls_pipeline, get_documents_hybrid_search
import sys
import uvicorn
import traceback
# Faltan los imports para que funcionen las funciones en si

# Create the FastAPI app instance
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global nlp, linker
    print("Cargando UMLS pipeline...")
    nlp, linker = load_umls_pipeline()
    print("Cargado correctamente.")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            query = await websocket.receive_text()
            try:
                documents = get_documents_hybrid_search(query, nlp, linker)
                classified_docs = get_documents_by_class(documents)
                rag_prompt = generate_prompt(classified_docs, query)
                response = generate_text(rag_prompt, MODEL_ID, HF_TOKEN)
                await websocket.send_text(response)
            except Exception as inner_exc:
                await websocket.send_text(f"Error en el servidor: {str(inner_exc)}")
                traceback.print_exc()  # Imprime el error en consola para debug
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8000)
