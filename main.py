import streamlit as st
from utils.session_state import initialize_session_state
from src.embeddings.embeddings import load_embeddings
from src.models.vicuna_model import load_llm
from src.text_processing.processor import process_pdf

# Initialize session state
initialize_session_state()

# Page configuration and title (from PDF page 15)
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("PDF RAG Assistant")

st.markdown("""
**AI Chatbot to ask and answering**
**User Guide:**
1. **Upload PDF** Upload your pdf
2. **Ask questions** Enter your question and get answer in real-time
---
""")

# Load models (from PDF page 15)
if not st.session_state.models_loaded:
    st.info("Loading the model...")
    st.session_state.embeddings = load_embeddings()
    st.session_state.llm = load_llm()
    st.session_state.models_loaded = True
    st.success("Model is ready!")
    st.rerun()

uploaded_file = st.file_uploader("Upload file PDF", type="pdf")
if uploaded_file and st.button("Process pdf"):
    with st.spinner("Processing..."):
        st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
        st.success(f"Success! {num_chunks} chunks")

if st.session_state.rag_chain:
    question = st.text_input("Enter your prompt:")
    if question:
        with st.spinner("Answering..."):
            output = st.session_state.rag_chain.invoke(question)
            answer = output.split("Answer:")[1].strip() if "Answer:" in output else output.strip()
            st.write("**Answer:**")
            st.write(answer)