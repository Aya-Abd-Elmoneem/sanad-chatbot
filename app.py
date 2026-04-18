import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# API KEY
# =========================
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# =========================
# Gemini Model (WORKING)
# =========================
model = genai.GenerativeModel("models/gemini-flash-latest")

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="SAND AI Assistant", layout="wide")
st.title("🌾 SAND AI Assistant (PDF Chatbot)")

# =========================
# Read PDF Text
# =========================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# =========================
# Embeddings (Stable)
# =========================
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =========================
# Create FAISS Index
# =========================
def create_vector_store(text_chunks):
    embeddings = get_embeddings()
    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    db.save_local("faiss_index")

# =========================
# Sidebar Upload
# =========================
with st.sidebar:
    st.header("📂 Upload PDFs")

    pdf_docs = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True
    )

    if st.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Reading PDFs..."):
                raw_text = get_pdf_text(pdf_docs)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )

                chunks = splitter.split_text(raw_text)
                create_vector_store(chunks)

            st.success("Processing completed ✅")
        else:
            st.warning("Please upload PDF files first")

# =========================
# Chat Section
# =========================
question = st.text_input("💬 Ask SAND:")

if question:
    embeddings = get_embeddings()

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are "SAND", an Egyptian AI assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(prompt)

    st.info(response.text)
