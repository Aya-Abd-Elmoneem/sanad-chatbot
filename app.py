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
# 4. IMPROVED HOME PAGE UI
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
                padding: 40px 0;
            }
            
            .title-text {
                font-size: 3.5rem;
                font-weight: 900;
                background: linear-gradient(90deg, #10b981, #34d399, #10b981);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            [data-testid="stColumn"] {
                position: relative;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }

            .card-ui {
                background: rgba(30, 41, 59, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 30px 15px;
                text-align: center;
                backdrop-filter: blur(10px);
                height: 400px;
                width: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                transition: 0.3s ease;
                z-index: 1;
            }

            /* --- التمركز التام للزر في منتصف البطاقة --- */
            .stButton > button {
                position: absolute !important;
                top: 50% !important;
                left: 50% !important;
                transform: translate(-50%, -50%) !important; /* لضمان التمركز في الوسط تماماً */
                width: 100% !important;
                height: 400px !important;
                background: transparent !important;
                border: none !important;
                color: transparent !important;
                z-index: 5 !important;
                cursor: pointer;
            }

            [data-testid="stColumn"]:hover .card-ui {
                border-color: #10b981;
                background: rgba(30, 41, 59, 0.6);
                transform: translateY(-8px);
                box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
            }

            .icon-box { font-size: 4.5rem; margin-bottom: 20px; }
            .card-title { font-size: 1.6rem; font-weight: 700; color: white; margin-bottom: 10px; }
            .card-desc { font-size: 1rem; color: #94a3b8; line-height: 1.5; }

            .stButton { margin: 0 !important; padding: 0 !important; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="main-header">
            <h1 class="title-text">SANAD AI Assistant</h1>
            <p style="color: #94a3b8; font-size: 1.2rem;">مساعدك الذكي في عالم الزراعة والتمويل</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    def create_card(col, icon, title, desc, key, chat_type):
        with col:
            st.markdown(f"""
                <div class="card-ui">
                    <div class="icon-box">{icon}</div>
                    <div class="card-title">{title}</div>
                    <div class="card-desc">{desc}</div>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("", key=key):
                st.session_state.chat_type = chat_type
                st.session_state.page = "chat"
                st.rerun()

    create_card(col1, "🌱", "تمويل المحاصيل", "حلول تمويلية ذكية للمزارعين ودعم الإنتاج المستدام.", "crop_btn", "agriculture")
    create_card(col2, "📈", "التمويل والقروض", "استكشف خيارات القروض والتسهيلات الائتمانية لمشاريعك.", "finance_btn", "general")
    create_card(col3, "🐄", "الثروة الحيوانية", "دعم فني وتمويلي لمشاريع الإنتاج الحيواني والداجني.", "livestock_btn", "agriculture")

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
            else:
                st.warning("يرجى اختيار ملفات أولاً.")

def chat_page():
    st.markdown(f"<h1 style='text-align: right; color: #10b981;'>💬 مساعد {st.session_state.chat_type.upper()}</h1>", unsafe_allow_html=True)
    
    if st.button("⬅️ العودة للرئيسية"):
        st.session_state.page = "home"
        st.rerun()

    st.divider()

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("اسأل SANAD..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
                try:
                    db = load_db()
                    docs = db.similarity_search(prompt)
                    context = "\n\n".join([d.page_content for d in docs])
                except:
                    context = "لا توجد ملفات مرفوعة لهذا القسم."

                sys_msg = "أنت خبير ذكاء اصطناعي في مجال الزراعة والتمويل. أجب باللغة العربية بأسلوب مهني ومختصر."
                full_query = f"{sys_msg}\n\nالسياق: {context}\n\nالسؤال: {prompt}"
                
                response = model.generate_content(full_query)
                st.markdown(response.text)
                
                # TTS
                audio_path = text_to_audio(response.text)
                autoplay_audio(audio_path)
                
                st.session_state.messages.append({"role": "assistant", "content": response.text})

# =========================
# 6. ROUTING
# =========================
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "chat":
    sidebar()
    chat_page()
