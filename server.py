import json 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect 
import uvicorn
# Faltan los imports para que funcionen las funciones en si

# Create the FastAPI app instance 
app = FastAPI() 

# WebSocket endpoint 
@app.websocket("/ws") 
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try: 
        while True:
            query = await websocket.receive_text()
            nlp, linker = load_umls_pipeline()
            if len(sys.argv) == 1:
                sys.argv.extend(["search",query])
            cli(nlp, linker)
            classified_docs = get_documents_by_class(documents)
            prompt = generate_prompt(classified_docs, user_prompt)
            response = generate_text(prompt, MODEL_ID, HF_TOKEN)
            await websocket.send_text(response)
    except WebSocketDisconnect: 
        print("Client disconnected")


if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8000)
