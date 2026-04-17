import streamlit as st
import os
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# =========================
# API KEY (use Streamlit secrets in production)
# =========================
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "AQ.Ab8RN6K3pk2kVR1cQB1w1yLTI0oslfmupO_syeBFJVII7Oqi3w")

# =========================
# Streamlit Config
# =========================
st.set_page_config(page_title="SAND AI Assistant", layout="wide")
st.title("🌾 SAND AI Assistant for Farmers")

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
# Create Vector Store
# =========================
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# =========================
# QA Chain (Stable Version)
# =========================
def get_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(
        template="""
You are "SAND", an Egyptian bank assistant.

Answer ONLY using the given context.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# =========================
# Sidebar UI
# =========================
with st.sidebar:
    st.header("📂 Upload PDF Files")

    pdf_docs = st.file_uploader(
        "Upload your PDFs",
        accept_multiple_files=True
    )

    if st.button("🔍 Process Files"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )

                text_chunks = splitter.split_text(raw_text)

                get_vector_store(text_chunks)

            st.success("Files processed successfully ✅")
        else:
            st.warning("Please upload PDF files first")

# =========================
# Chat Section
# =========================
user_question = st.text_input("💬 Ask SAND:")

if user_question:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

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
