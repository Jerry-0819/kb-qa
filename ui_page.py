import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/v1/chat"

def call_api(query: str, mode: str, k: int):
    """Send a request to the FastAPI RAG backend."""
    payload = {
        "query": query,
        "mode": mode,
        "k": k,
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Error calling API: {e}")
        return None

    return resp.json()

st.set_page_config(page_title="Internal Employee Knowledge Base Q&A System", layout="wide")

st.title("Internal Employee Knowledge Base Q&A System")

col1, col2 = st.columns([1, 1])

with col1:
    mode = st.radio(
        "Mode",
        options=["chain", "agent"],
        index=0,
        horizontal=True,
        help="chain = simple RAG chain; agent = tool-using agent with retrieval tool",
    )

with col2:
    k = st.slider(
        "Top-k documents",
        min_value=1,
        max_value=8,
        value=3,
        step=1,
        help="How many chunks to retrieve from FAISS",
    )


query = st.text_area(
    "Ask a question:",
    placeholder="e.g. What are the basic principles of recruitment?",
    height=120,
)

ask_button = st.button("Ask", type="primary")

if ask_button:
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            result = call_api(query=query, mode=mode, k=k)

        if result is not None:
            st.subheader("Answer")
            st.markdown(result.get("answer", "_No answer returned._"))

            matches = result.get("matches", [])
            if matches:
                st.subheader("Retrieved Context")
                for i, m in enumerate(matches, start=1):
                    meta = m.get("meta", {})
                    file = meta.get("file", "unknown")
                    page = meta.get("page", "N/A")
                    score = m.get("score", 0.0)
                    snippet = m.get("snippet", "")

                    with st.expander(f"{i}. {file} (page {page}) · score={score:.3f}"):
                        st.write(snippet)
            else:
                st.info("No matches returned from the retriever.")
