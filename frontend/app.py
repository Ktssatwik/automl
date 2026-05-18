import streamlit as st
import requests

st.set_page_config(page_title="Agentic AutoML POC", layout="wide")
st.title("Agentic AutoML POC")
st.write("Phase 1 setup complete. Frontend is connected to FastAPI backend.")

backend_url = st.text_input("Backend URL", value="http://127.0.0.1:8000")

if st.button("Check Backend Health"):
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        response.raise_for_status()
        st.success(f"Backend healthy: {response.json()}")
    except Exception as exc:
        st.error(f"Backend check failed: {exc}")
