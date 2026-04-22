import streamlit as st
import requests

API = "http://localhost:8000"

st.set_page_config(page_title="📚 Document Q&A", layout="centered")
st.title("📚 RAG Document Q&A")

# --- Upload ---
st.subheader("1. Upload a Document")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
if uploaded and st.button("Index Document"):
    with st.spinner("Indexing..."):
        res = requests.post(f"{API}/upload", files={"file": uploaded})
    if res.ok:
        st.success(res.json()["message"])
    else:
        st.error(res.text)

st.divider()

# --- Ask ---
st.subheader("2. Ask a Question")
question = st.text_input("Your question")
if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        res = requests.post(f"{API}/ask", json={"question": question})
    if res.ok:
        data = res.json()
        st.markdown(f"**Answer:** {data['answer']}")
        with st.expander("📎 Sources"):
            for s in data["sources"]:
                st.markdown(f"- **{s['source']}** (page {s['page']})")
                st.caption(s["snippet"])
    else:
        st.error(res.text)