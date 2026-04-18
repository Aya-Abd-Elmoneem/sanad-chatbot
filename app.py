import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import base64
import asyncio
import edge_tts
import re

# =========================
# CONFIG
# =========================
# Note: Ensure st.secrets["GOOGLE_API_KEY"] is set in your Streamlit Cloud settings
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("models/gemini-flash-latest")

# Set page config at the very top level
st.set_page_config(page_title="SANAD AI Assistant", page_icon="🌾", layout="wide", initial_sidebar_state="collapsed")

# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "chat_type" not in st.session_state:
    st.session_state.chat_type = None

# =========================
# PDF & VECTOR DB FUNCTIONS (Keeping your logic)
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

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(text_chunks):
    embeddings = get_embeddings()
    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    db.save_local("faiss_index")

def load_db():
    embeddings = get_embeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# =========================
# AUDIO FUNCTIONS
# =========================
def clean_text_for_tts(text):
    text = re.sub(r"[.,:*()\-\n#]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def text_to_audio(text):
    audio_file = "response.mp3"
    clean_text = clean_text_for_tts(text)
    async def generate():
        communicate = edge_tts.Communicate(clean_text, voice="ar-EG-SalmaNeural")
        await communicate.save(audio_file)
    asyncio.run(generate())
    return audio_file

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(f'<audio autoplay controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

# =========================
# HOME PAGE (PROFESSIONAL DESIGN)
# =========================
def home_page():
    # 1. Custom CSS for the "Winning Design"
    st.markdown("""
        <style>
            /* Main Background and Text */
            .stApp {
                background-color: #0F172A;
                color: #F8FAFC;
            }
            
            /* Header Styling */
            .main-header {
                text-align: center;
                padding: 40px 0;
                background: radial-gradient(circle at top, #1E293B 0%, #0F172A 100%);
            }
            .header-title {
                font-size: 3.5rem;
                font-weight: 800;
                letter-spacing: -1px;
                margin-bottom: 10px;
                color: #FFFFFF;
            }
            .subtitle {
                font-size: 1.5rem;
                color: #94A3B8;
                margin-bottom: 40px;
            }

            /* Card Styling */
            div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
                border: 1px solid #334155;
                border-radius: 20px;
                background: #1E293B;
                padding: 40px 20px;
                transition: transform 0.3s ease, border-color 0.3s ease;
                text-align: center;
            }
            
            /* Icon and Text inside Cards */
            .card-icon { font-size: 4rem; margin-bottom: 20px; }
            .card-title { font-size: 1.4rem; font-weight: 700; color: #F1F5F9; margin-bottom: 10px; }
            .card-desc { font-size: 1rem; color: #94A3B8; margin-bottom: 20px; height: 50px; }

            /* Professional Button Styling inside Cards */
            .stButton > button {
                width: 100%;
                border-radius: 12px;
                border: 1px solid #10B981;
                background-color: transparent;
                color: #10B981;
                font-weight: 600;
                padding: 10px 0;
                transition: 0.3s;
            }
            .stButton > button:hover {
                background-color: #10B981;
                color: white;
                transform: scale(1.02);
            }
        </style>
    """, unsafe_allow_html=True)

    # Header section
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="header-title">🌾 SANAD AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">اختر القسم المناسب للبدء</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Cards Grid
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card-icon">🌱</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">تمويل المحاصيل الزراعية</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">حلول ذكية لدعم إنتاج المحاصيل والمزارعين</div>', unsafe_allow_html=True)
        if st.button("دخول القسم", key="btn_agri"):
            st.session_state.chat_type = "agriculture"
            st.session_state.page = "chat"
            st.rerun()

    with col2:
        st.markdown('<div class="card-icon">📊</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">التمويل والقروض</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">استشارات ائتمانية وتسهيلات مالية للمشاريع</div>', unsafe_allow_html=True)
        if st.button("دخول القسم", key="btn_finance"):
            st.session_state.chat_type = "general" # Mapping to your existing system_prompt keys
            st.session_state.page = "chat"
            st.rerun()

    with col3:
        st.markdown('<div class="card-icon">🐄</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">الثروة الحيوانية والدواجن</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">دعم فني وتمويلي لمشاريع الإنتاج الحيواني</div>', unsafe_allow_html=True)
        if st.button("دخول القسم", key="btn_livestock"):
            st.session_state.chat_type = "agriculture"
            st.session_state.page = "chat"
            st.rerun()

# =========================
# SIDEBAR PDF (Keeping your logic)
# =========================
def sidebar():
    with st.sidebar:
        st.header("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = splitter.split_text(raw_text)
                    create_vector_store(chunks)
                st.success("Done ✅")
            else:
                st.warning("Please upload files")

# =========================
# CHAT PAGE (Keeping your logic)
# =========================
def chat_page():
    st.title(f"💬 {st.session_state.chat_type.upper()} CHATBOT")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    question = st.text_input("Ask your question:")

    if question:
        try:
            db = load_db()
            docs = db.similarity_search(question)
            context = "\n\n".join([d.page_content for d in docs])
        except:
            context = "No PDF context available."

        system_prompt = {
            "agriculture": "You are an agriculture expert AI assistant.",
            "data": "You are a data science expert AI assistant.",
            "general": "You are a helpful AI assistant."
        }

        prompt = f"""
        {system_prompt.get(st.session_state.chat_type, "You are a helpful assistant.")}
        Context: {context}
        Question: {question}
        Answer clearly and simply in Arabic:
        """

        response = model.generate_content(prompt)
        st.success(response.text)

        # AUDIO OUTPUT
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
