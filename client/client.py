import asyncio
import websockets
import os

async def send_query():
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri, ping_interval=100, ping_timeout=100) as websocket:
        query = input("Query: ")
        await websocket.send(query)
        response = await websocket.recv()
        os.startfile("report.txt")
asyncio.run(send_query())
