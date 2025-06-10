from utils import *

async def iniciar_servidor():
    """Inicia el servidor FastAPI."""
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    # Aqui inicializar las funciones requeridas.
    await server.serve()

async def main():
    server_task = asyncio.create_task(iniciar_servidor())