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
# 1. CONFIGURATION
# =========================
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("models/gemini-flash-latest")

st.set_page_config(
    page_title="SANAD AI Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# 2. SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "chat_type" not in st.session_state:
    st.session_state.chat_type = None

# =========================
# 3. CORE FUNCTIONS (PDF & TTS)
# =========================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: text += page_text
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

def text_to_audio(text):
    audio_file = "response.mp3"
    clean_text = re.sub(r"[.,:*()\-\n#]", " ", text)
    async def generate():
        communicate = edge_tts.Communicate(clean_text, voice="ar-EG-SalmaNeural")
        await communicate.save(audio_file)
    asyncio.run(generate())
    return audio_file

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

# =========================
# 4. FIXED HOME PAGE (FULL CLICKABLE CARDS)
# =========================
def home_page():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900&display=swap');

            .stApp {
                background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 100%);
                font-family: 'Cairo', sans-serif;
                direction: rtl;
            }

            .main-header {
                text-align: center;
                padding: 50px 0 30px 0;
            }
            
            .title-text {
                font-size: 3.5rem;
                font-weight: 900;
                background: linear-gradient(90deg, #10b981, #34d399, #10b981);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }

            .tagline {
                color: #94a3b8;
                font-size: 1.2rem;
            }

            /* تصميم حاوية البطاقة */
            .card-container {
                position: relative;
                background: rgba(30, 41, 59, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 40px 20px;
                text-align: center;
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
                height: 350px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }

            .card-container:hover {
                transform: translateY(-10px);
                border-color: #10b981;
                box-shadow: 0 15px 30px rgba(0,0,0,0.3);
            }

            /* جعل الزر الشفاف يغطي البطاقة بالكامل دون التأثير على المحتوى */
            .stButton > button {
                position: absolute !important;
                top: 0; left: 0; right: 0; bottom: 0;
                width: 100% !important;
                height: 100% !important;
                background: transparent !important;
                border: none !important;
                color: transparent !important;
                z-index: 100;
                cursor: pointer;
            }

            .icon-circle {
                font-size: 4rem;
                margin-bottom: 20px;
            }

            .card-title {
                font-size: 1.5rem;
                font-weight: 700;
                color: #ffffff;
                margin-bottom: 15px;
            }

            .card-desc {
                font-size: 1rem;
                color: #94a3b8;
                line-height: 1.5;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="main-header">
            <h1 class="title-text">SANAD AI Assistant</h1>
            <p class="tagline">مساعدك الذكي في عالم الزراعة والتمويل</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    # بطاقة قسم تمويل المحاصيل
    with col1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        if st.button("Crop Financing", key="btn_crop"):
            st.session_state.chat_type = "agriculture"
            st.session_state.page = "chat"
            st.rerun()
        st.markdown("""
            <div class="icon-circle">🌱</div>
            <div class="card-title">تمويل المحاصيل</div>
            <div class="card-desc">حلول ذكية لدعم إنتاج المحاصيل والمزارعين والنمو المستدام.</div>
        </div>
        """, unsafe_allow_html=True)

    # بطاقة قسم التمويل والقروض
    with col2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        if st.button("Finance & Loans", key="btn_finance"):
            st.session_state.chat_type = "general"
            st.session_state.page = "chat"
            st.rerun()
        st.markdown("""
            <div class="icon-circle">📈</div>
            <div class="card-title">التمويل والقروض</div>
            <div class="card-desc">استشارات ائتمانية وتسهيلات مالية مبتكرة لمشاريعك.</div>
        </div>
        """, unsafe_allow_html=True)

    # بطاقة قسم الثروة الحيوانية
    with col3:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        if st.button("Livestock", key="btn_livestock"):
            st.session_state.chat_type = "agriculture"
            st.session_state.page = "chat"
            st.rerun()
        st.markdown("""
            <div class="icon-circle">🐄</div>
            <div class="card-title">الثروة الحيوانية</div>
            <div class="card-desc">دعم فني وتمويلي متخصص لمشاريع الإنتاج الحيواني والداجني.</div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# 5. CHAT PAGE & SIDEBAR
# =========================
def sidebar():
    with st.sidebar:
        st.header("📂 إدارة ملفات القسم")
        pdf_docs = st.file_uploader("ارفع ملفات PDF (Knowledge Base)", accept_multiple_files=True)
        if st.button("تحديث قاعدة البيانات"):
            if pdf_docs:
                with st.spinner("جاري تحليل الملفات..."):
                    raw_text = get_pdf_text(pdf_docs)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = splitter.split_text(raw_text)
                    create_vector_store(chunks)
                st.success("تم التحديث بنجاح! ✅")

def chat_page():
    st.markdown(f"<h1 style='text-align: right; color: #10b981;'>💬 مساعد {st.session_state.chat_type.upper()}</h1>", unsafe_allow_html=True)
    if st.button("⬅️ العودة للرئيسية"):
        st.session_state.page = "home"
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("اسأل SANAD..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
                try:
                    db = load_db()
                    docs = db.similarity_search(prompt)
                    context = "\n\n".join([d.page_content for d in docs])
                except: context = "لا توجد ملفات مرفوعة."

                response = model.generate_content(f"أجب بالعربية: {prompt}\n\nالسياق: {context}")
                st.markdown(response.text)
                audio_path = text_to_audio(response.text)
                autoplay_audio(audio_path)
                st.session_state.messages.append({"role": "assistant", "content": response.text})

# =========================
# 6. ROUTING
# =========================
if st.session_state.page == "home":
    home_page()
else:
    sidebar()
    chat_page()
