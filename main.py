# Hay que obtener el ip adress del servidor con ip_adress.py
from utils import *
from server import app

async def test_client(event: asyncio.Event):
    """Cliente de prueba para conectarse al servidor WebSocket."""
    uri = "ws://localhost:8000/api/v1/chat?user_id=1"

    async with websockets.connect(uri) as websocket:
        while True:
            user_input = input("Mensaje (o 'salir' para terminar): ")
            if user_input.lower() == "salir":
                break

            # Enviar mensaje al servidor
            await websocket.send(user_input)

            # Recibir respuesta del servidor
            response_data = await websocket.recv()

            print(f"Respuesta del servidor: {response_data}")
    
    # Notificar que el cliente ha terminado
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
