import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# cache the models
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
