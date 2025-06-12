import json
import os
from pathlib import Path
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from dotenv import load_dotenv
import websockets
import asyncio
import signal
from datetime import datetime

async def test_client(event: asyncio.Event, user_id: int = 1) -> None:
    uri = f"ws://10.8.47.101:8000/api/v1/chat?user_id={user_id}"

    async with websockets.connect(uri) as websocket:
        await websocket.send(user_id)
        try:
            while True:
                user_input = input("Query(write 'finish' to end): ")
                if user_input.lower() == "salir":
                    break

                await websocket.send(user_input)
                
                response = (await websocket.recv()).strip()

                print(f"The assistant's treatment report is: \n{response}")

        finally:
            await websocket.close()

    event.set()

async def main():
    """Lanza el servidor y el cliente de pruebas."""
    event = asyncio.Event()

    client_task = asyncio.create_task(test_client(event))

    await event.wait()
    
    # Cuando el cliente termina, se detiene el servidor
    print("Cerrando el servidor...")
    # Detener Uvicorn manualmente usando señales del sistema
    os.kill(os.getpid(), signal.SIGTERM)
    print("Servidor detenido. ¡Hasta la próxima!")

if __name__ == "__main__":
    asyncio.run(main())
