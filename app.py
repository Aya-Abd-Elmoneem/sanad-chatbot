import streamlit as st
import os
from PyPDF2 import PdfReader

import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# API KEY
# =========================
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "AIzaSyC5Pf2lZRHTAetZUyH5T0TpaUHYh5V6nu0")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="SAND AI Assistant", layout="wide")
st.title("🌾 SAND AI Assistant for Egyptian Farmers")

# =========================
# PDF Reader
# =========================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# =========================
# Embeddings (STABLE)
# =========================
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =========================
# Create FAISS Vector Store
# =========================
def get_vector_store(text_chunks):
    embeddings = get_embeddings()
    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    db.save_local("faiss_index")

# =========================
# Gemini Model (NO LANGCHAIN WRAPPER)
# =========================
model = genai.GenerativeModel("gemini-1.5-flash")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("📂 Upload PDFs")

    pdf_docs = st.file_uploader(
        "Upload your PDF files",
        accept_multiple_files=True
    )

    if st.button("🔍 Process"):
        if pdf_docs:
            with st.spinner("Reading PDFs..."):
                raw_text = get_pdf_text(pdf_docs)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )

                chunks = splitter.split_text(raw_text)
                get_vector_store(chunks)

            st.success("Processing completed ✅")
        else:
            st.warning("Please upload PDF files first")

# =========================
# Chat Section
# =========================
user_question = st.text_input("💬 Ask SAND:")

if user_question:
    embeddings = get_embeddings()

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(user_question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are "SAND", an Egyptian bank assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{user_question}

Answer:
"""

    response = model.generate_content(prompt)

    st.info(response.text)
