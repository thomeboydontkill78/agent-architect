import json
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Agent Architect", page_icon="🤖", layout="wide")
st.title("🤖 Agent Architect · 智能客服")
st.caption("ReAct Agent + 自建 RAG 工具 + middleware 动态提示词")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("请输入问题...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        try:
            resp = requests.post(
                f"{API_URL}/stream",
                json={"question": prompt},
                stream=True, timeout=120,
            )
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj["type"] == "tool":
                    full += f"\n> 🔧 {obj['text']}\n\n"
                else:
                    full += obj["text"]
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)
        except Exception as e:
            full = f"⚠️ {e}"
            placeholder.markdown(full)

    st.session_state.messages.append({"role": "assistant", "content": full})
