import streamlit as st
import os
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from langchain.chains import load_qa_chain
from langchain.prompts import PromptTemplate

# =========================
# API KEY
# =========================
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# =========================
# Streamlit Configuration
# =========================
st.set_page_config(page_title="SAND - AI Assistant", layout="wide")
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
# QA Chain (Question Answering)
# =========================
def get_conversational_chain():
    prompt_template = """
    You are an Egyptian bank employee named 'SAND'.
    Help farmers using simple rural Egyptian dialect.

    Answer only from the provided context.

    Context:
    {context}

    Question:
    {question}

    Response:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# =========================
# Sidebar UI
# =========================
with st.sidebar:
    st.header("📂 Data Upload")

    pdf_docs = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True
    )

    if st.button("🔍 Process"):
        if pdf_docs:
            with st.spinner("Reading files..."):
                raw_text = get_pdf_text(pdf_docs)

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )

                chunks = text_splitter.split_text(raw_text)

                get_vector_store(chunks)

            st.success("Processing completed successfully ✅")
        else:
            st.warning("Please upload PDF files first")

# =========================
# Main Chat Interface
# =========================
user_question = st.text_input("💬 Ask SAND:")

if user_question:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.info(response["output_text"])
