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
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("models/gemini-flash-latest")

st.set_page_config(page_title="SANAD AI Assistant", page_icon="🌾", layout="wide", initial_sidebar_state="collapsed")

# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "chat_type" not in st.session_state:
    st.session_state.chat_type = None

# =========================
# PDF & AUDIO FUNCTIONS (Keeping your logic)
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

def clean_text_for_tts(text):
    text = re.sub(r"[.,:*()\-\n#]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def text_to_audio(text):
    audio_file = "response.mp3"
    clean_text = clean_text_for_tts(text)
    async def generate():
        communicate = edge_tts.Communicate(clean_text, voice="ar-EG-SalmaNeural")
        await communicate.run() # Fixed: use run() for standard execution
        await communicate.save(audio_file)
    asyncio.run(generate())
    return audio_file

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

# =========================
# HOME PAGE (PROFESSIONAL RTL DESIGN)
# =========================
def home_page():
    # CSS لتحويل الواجهة للعربية وجعلها عصرية
    st.markdown("""
        <style>
            /* اتجاه النص من اليمين لليسار */
            .main {
                direction: rtl;
                text-align: right;
            }
            
            /* خلفية احترافية */
            .stApp {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #ffffff;
            }

            /* تصميم الهيدر */
            .header-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
                padding: 50px 0;
            }
            .title-text {
                font-size: 4rem;
                font-weight: 800;
                background: linear-gradient(90deg, #10b981, #34d399);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }
            .sub-title {
                font-size: 1.5rem;
                color: #94a3b8;
            }

            /* تصميم البطاقات (Cards) */
            [data-testid="stVerticalBlock"] > div > div > div[data-testid="stVerticalBlock"] {
                background: rgba(30, 41, 59, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                padding: 30px;
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                text-align: center;
                backdrop-filter: blur(10px);
            }
            
            [data-testid="stVerticalBlock"] > div > div > div[data-testid="stVerticalBlock"]:hover {
                transform: translateY(-10px);
                border-color: #10b981;
                box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            }

            .card-icon { font-size: 5rem; margin-bottom: 20px; }
            .card-title { font-size: 1.8rem; font-weight: 700; color: #ffffff; margin-bottom: 15px; }
            .card-desc { font-size: 1.1rem; color: #94a3b8; margin-bottom: 30px; line-height: 1.6; }

            /* تحسين شكل الأزرار */
            .stButton > button {
                width: 100%;
                background: linear-gradient(90deg, #10b981, #059669);
                color: white;
                border: none;
                padding: 12px 0;
                border-radius: 12px;
                font-size: 1.2rem;
                font-weight: 600;
                transition: 0.3s;
            }
            .stButton > button:hover {
                box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
                transform: scale(1.05);
            }
        </style>
    """, unsafe_allow_html=True)

    # الهيدر
    st.markdown("""
        <div class="header-container">
            <h1 class="title-text">🌾 SANAD AI Assistant</h1>
            <p class="sub-title">مساعدك الذكي في عالم الزراعة والتمويل</p>
            <p style="color:#64748b; margin-top:10px;">اختر القسم المناسب للبدء</p>
        </div>
    """, unsafe_allow_html=True)

    # شبكة الأقسام
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card-icon">🌱</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">تمويل المحاصيل</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">حلول ذكية لدعم إنتاج المحاصيل والمزارعين والنمو المستدام</div>', unsafe_allow_html=True)
        if st.button("دخول القسم", key="btn_agri"):
            st.session_state.chat_type = "agriculture"
            st.session_state.page = "chat"
            st.rerun()

    with col2:
        st.markdown('<div class="card-icon">📈</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">التمويل والقروض</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">استشارات ائتمانية وتسهيلات مالية مبتكرة لمشاريعك</div>', unsafe_allow_html=True)
        if st.button("دخول القسم", key="btn_finance"):
            st.session_state.chat_type = "general"
            st.session_state.page = "chat"
            st.rerun()

    with col3:
        st.markdown('<div class="card-icon">🐄</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">الثروة الحيوانية</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">دعم فني وتمويلي متخصص لمشاريع الإنتاج الحيواني والداجني</div>', unsafe_allow_html=True)
        if st.button("دخول القسم", key="btn_livestock"):
            st.session_state.chat_type = "agriculture"
            st.session_state.page = "chat"
            st.rerun()

# =========================
# CHAT & SIDEBAR (Keeping your logic)
# =========================
def sidebar():
    with st.sidebar:
        st.header("📂 ملفات المعرفة")
        pdf_docs = st.file_uploader("ارفع ملفات PDF الخاصة بالقسم", accept_multiple_files=True)
        if st.button("معالجة الملفات"):
            if pdf_docs:
                with st.spinner("جاري التحليل..."):
                    raw_text = get_pdf_text(pdf_docs)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = splitter.split_text(raw_text)
                    create_vector_store(chunks)
                st.success("تم التجهيز ✅")
            else: st.warning("يرجى رفع ملفات أولاً")

def chat_page():
    st.markdown(f"<h1 style='text-align: right;'>💬 محادثة: {st.session_state.chat_type}</h1>", unsafe_allow_html=True)
    
    if st.button("⬅️ العودة للرئيسية"):
        st.session_state.page = "home"
        st.rerun()

    question = st.chat_input("اسألني أي شيء عن هذا القسم...")

    if question:
        st.chat_message("user").write(question)
        with st.spinner("يفكر SANAD..."):
            try:
                db = load_db()
                docs = db.similarity_search(question)
                context = "\n\n".join([d.page_content for d in docs])
            except: context = "لا يوجد سياق من الملفات حالياً."

            prompt = f"أجب باللغة العربية بوضوح: {question}\n\nالسياق المتاح: {context}"
            response = model.generate_content(prompt)
            
            st.chat_message("assistant").write(response.text)
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
