import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
st.title("🤖 Agent Architect · 智能客服")
st.caption("ReAct Agent + 自建 RAG 工具 + middleware 动态提示词")

prompt = st.chat_input("请输入问题...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            r = requests.post(
                f"{API_URL}/ask", json={"question": prompt}, timeout=120,
            )
            st.markdown(r.json()["answer"])
        except Exception as e:
            st.error(f"调用失败: {e}")
