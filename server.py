from utils import *

from orquestador import orquestador
from capa_de_componentes import capa_de_componentes
from conversador_final import conversador_final

DB_PATH = r".\db"

app = FastAPI()

# Diccionario para sesiones activas
user_sessions = {}

@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado del servidor."""
    logging.info("Health check llamado")
    return {"status": "ok"}


@app.websocket("/api/v1/chat")
async def websocket_endpoint(websocket: WebSocket, user_id: int = Query(default=None)):
    """
    Maneja la comunicación WebSocket con múltiples clientes.
    """
    await websocket.accept()
    
    conversacion = []
    try:
        while True:
            data = await websocket.receive_json()
            user_prompt = data.get("prompt")

            if not user_prompt:
                continue
            
            conversacion.append({"role":"user", "content": user_prompt})
            # Aqui tirar la función que calcula la respuesta 
            conversacion.append({"role": "assistant", "content": respuesta})
            await websocket.send(respuesta)

    except WebSocketDisconnect:
        print("Client disconnected.")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")