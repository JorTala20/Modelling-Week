import asyncio 
import websockets
 
async def send_query(): 
    uri = "ws://localhost:8000/ws" 
    async with websockets.connect(uri) as websocket:
        query = input("Query: ")
        await websocket.send(query)  
        response = await websocket.recv() 
        print(f"Received response: {response}") 

asyncio.get_event_loop().run_until_complete(send_query())