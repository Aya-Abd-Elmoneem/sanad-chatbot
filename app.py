import streamlit as st
import os
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# =========================
# API KEY (ONLY for LLM)
# =========================
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "AQ.Ab8RN6L7FSURN_RrKq7qULC8DNMMIvQLnhVrnI5g8KY2EC_rKQ")

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="SAND AI Assistant", layout="wide")
st.title("🌾 SAND AI Assistant for Egyptian Farmers")

# =========================
# Extract text from PDFs
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
# Embeddings (FIXED - NO GOOGLE EMBEDDINGS)
# =========================
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =========================
# Create Vector Store
# =========================
def get_vector_store(text_chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# =========================
# QA Chain
# =========================
def get_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(
        template="""
You are "SAND", an Egyptian bank assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("📂 Upload PDF Files")

    pdf_docs = st.file_uploader(
        "Upload your PDF files",
        accept_multiple_files=True
    )

    if st.button("🔍 Process Files"):
        if pdf_docs:
            with st.spinner("Reading PDFs..."):
                raw_text = get_pdf_text(pdf_docs)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )

                text_chunks = splitter.split_text(raw_text)

                get_vector_store(text_chunks)

            st.success("Processing completed successfully ✅")
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

    chain = get_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.info(response["output_text"])
