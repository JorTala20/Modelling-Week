from utils import *

BACKEND_WS = "ws://localhost:8000/api/v1/chat"
# You can also point to the container IP or a public URL when you deploy.

async def _connect_ws(user_id):
    """
    Opens a persistent websocket connection and registers the user_id.
    The ws object is stored in st.session_state["ws"].
    """
    uri = f"{BACKEND_WS}?user_id={user_id}"
    ws = await websockets.connect(uri)
    await ws.send(user_id)
    st.session_state["ws"] = ws


async def _send_prompt(prompt):
    """
    Sends `prompt`, waits for a reply, returns the reply string.
    """
    ws = st.session_state["ws"]
    await ws.send(prompt)
    reply = await ws.recv()
    return reply.strip()


async def _close_ws():
    ws = st.session_state.get("ws")
    if ws and not ws.closed:
        await ws.close()
    st.session_state["ws"] = None

st.set_page_config(page_title="My FastAPI Chat", page_icon="💬", layout="wide")
st.title("💬 FastAPI + Streamlit chat demo")

if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())[:8]
if "ws" not in st.session_state:
    st.session_state["ws"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if st.session_state["ws"] is None or st.session_state["ws"].closed:
    asyncio.run(_connect_ws(st.session_state["user_id"]))

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Type your message"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        response = asyncio.run(_send_prompt(prompt))
    except Exception as exc:
        response = f"Error talking to backend: {exc}"

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

st.sidebar.header("Session")
st.sidebar.write(f"User id: `{st.session_state['user_id']}`")
if st.sidebar.button("Disconnect"):
    asyncio.run(_close_ws())
    st.sidebar.success("Disconnected. Reload page to reconnect.")