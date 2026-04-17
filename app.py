import streamlit as st
import os
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# =========================
# API KEY (use Streamlit secrets in production)
# =========================
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "YOUR_API_KEY")

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="SAND AI Assistant", layout="wide")
st.title("🌾 SAND Project: AI Assistant for Egyptian Farmers")

# =========================
# Extract text from PDFs
# =========================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# =========================
# Create vector store
# =========================
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# =========================
# Build QA chain (NEW LANGCHAIN STYLE)
# =========================
def get_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    prompt = ChatPromptTemplate.from_template("""
    You are "SAND", an Egyptian bank assistant.
    Answer only from the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    chain = create_stuff_documents_chain(llm, prompt)
    return chain

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("📂 Upload Data")

    pdf_docs = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True
    )

    if st.button("🔍 Process"):
        if pdf_docs:
            with st.spinner("Reading PDFs..."):
                raw_text = get_pdf_text(pdf_docs)

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )

                chunks = text_splitter.split_text(raw_text)

                get_vector_store(chunks)

            st.success("Processing completed ✅")
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

    response = chain.invoke({
        "context": docs,
        "question": user_question
    })

    st.info(response)
