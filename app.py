import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import base64
import os
import asyncio
import edge_tts

# =========================
# CONFIG
# =========================
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("models/gemini-1.5-flash")

st.set_page_config(page_title="SANAD AI", layout="wide")

# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "chat_type" not in st.session_state:
    st.session_state.chat_type = None


# =========================
# PDF READER
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
# EMBEDDINGS
# =========================
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================
# FAISS STORE
# =========================
def create_vector_store(text_chunks):
    embeddings = get_embeddings()
    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    db.save_local("faiss_index")


def load_db():
    embeddings = get_embeddings()
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


# =========================
# TEXT → AUDIO (EGYPTIAN VOICE)
# =========================
def text_to_audio(text):
    audio_file = "response.mp3"

    async def generate():
        communicate = edge_tts.Communicate(
            text,
            voice="ar-EG-SalmaNeural"
        )
        await communicate.save(audio_file)

    asyncio.run(generate())
    return audio_file


def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()

    audio_html = f"""
    <audio autoplay controls>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


# =========================
# HOME PAGE (UPDATED UI)
# =========================
def home_page():
    st.markdown("""
        <h1 style='text-align:center; color:#2E8B57;'>🌾 SANAD AI Assistant</h1>
        <p style='text-align:center; font-size:18px;'>
        Choose your AI chatbot
        </p>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        .stButton>button {
            width: 320px;
            height: 75px;
            font-size: 20px;
            border-radius: 15px;
            background-color: #2E8B57;
            color: white;
            font-weight: bold;
            display: block;
            margin: 15px auto;
        }

        .stButton>button:hover {
            background-color: #256b45;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        if st.button("🌱 Agriculture AI"):
            st.session_state.page = "chat"
            st.session_state.chat_type = "agriculture"

        if st.button("📊 Data Science AI"):
            st.session_state.page = "chat"
            st.session_state.chat_type = "data"

        if st.button("🤖 General AI"):
            st.session_state.page = "chat"
            st.session_state.chat_type = "general"


# =========================
# SIDEBAR PDF
# =========================
def sidebar():
    with st.sidebar:
        st.header("📂 Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload PDFs",
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100
                    )

                    chunks = splitter.split_text(raw_text)
                    create_vector_store(chunks)

                st.success("Done ✅")
            else:
                st.warning("Please upload files")


# =========================
# CHAT PAGE
# =========================
def chat_page():
    st.title(f"💬 {st.session_state.chat_type.upper()} CHATBOT")

    if st.button("⬅ Back"):
        st.session_state.page = "home"

    question = st.text_input("Ask your question:")

    if question:
        db = load_db()
        docs = db.similarity_search(question)

        context = "\n\n".join([d.page_content for d in docs])

        system_prompt = {
            "agriculture": "You are an agriculture expert AI assistant.",
            "data": "You are a data science expert AI assistant.",
            "general": "You are a helpful AI assistant."
        }

        prompt = f"""
{system_prompt[st.session_state.chat_type]}

Context:
{context}

Question:
{question}

Answer in a clear, simple way:
"""

        response = model.generate_content(prompt)

        st.success(response.text)

        # AUDIO (Egyptian)
        audio_file = text_to_audio(response.text)
        autoplay_audio(audio_file)


# =========================
# ROUTING
# =========================
if st.session_state.page == "home":
    home_page()

elif st.session_state.page == "chat":
    sidebar()
    chat_page()
