from utils import *



app = FastAPI()

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
            user_prompt = data.get("prompt") #aqui query inicial

            if not user_prompt:
                continue
            
            conversacion.append({"role":"user", "content": user_prompt})
            cui_prompt = cui(user_prompt)
            documents = hybrid_search(cui_prompt)
            classified_docs = get_documents_by_class(documents)
            prompt = generate_prompt(classified_docs, user_prompt)
            respuesta=generate_text(prompt, MODEL_ID, HF_TOKEN)

            await websocket.send(respuesta)

    except WebSocketDisconnect:
        print("Client disconnected.")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
