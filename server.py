from utils import *

app = FastAPI()

user_sessions = {}


@app.get("/health")
async def health_check():
    logging.info("Health check called")
    return {"status": "ok"}


def init_session(user_id):
    """
    Creates a session for a new user or returns the session from an existing user
    """
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "conversation": []
        }
        logging.info(f"New session created for user_id={user_id}")
    return user_sessions[user_id]


@app.websocket("/api/v1/chat")
async def websocket_endpoint(websocket, user_id):
    await websocket.accept()

    session = init_session(user_id)
    conversation = session["conversation"]

    try:
        while True:
            data = await websocket.receive_json()
            user_prompt = data.get("prompt")
            if not user_prompt:
                continue

            conversation.append({"role": "user", "content": user_prompt})

            cui_prompt = cui(user_prompt)
            documents = hybrid_search(cui_prompt)
            classified_docs = get_documents_by_class(documents)
            prompt = generate_prompt(classified_docs, user_prompt)
            response = generate_text(prompt, MODEL_ID, HF_TOKEN)

            conversation.append({"role": "assistant", "content": response})
            session["conversation"] = conversation
            await websocket.send_json({"user_id": user_id, "response": response})

    except WebSocketDisconnect:
        user_sessions.pop(user_id, None)
        logging.info(f"Disconnected client {user_id} and terminated session.")

    except Exception as e:
        logging.error(f"WebSocket error (user {user_id}): {e}")
