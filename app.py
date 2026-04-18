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
# 2. CORE FUNCTIONS
# =========================
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
# 3. ADVANCED UI STYLING
# =========================
def home_page():
    # CSS لتحقيق مظهر البطاقات الزجاجية وتنسيق الأيقونات
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

            /* تصميم البطاقة كحاوية مرئية فقط */
            .card-ui {
                background: rgba(30, 41, 59, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 30px 15px;
                text-align: center;
                backdrop-filter: blur(10px);
                height: 400px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                transition: 0.3s ease;
            }

            .card-ui:hover {
                border-color: #10b981;
                background: rgba(30, 41, 59, 0.6);
                transform: translateY(-5px);
            }

            .icon-box { font-size: 4.5rem; margin-bottom: 20px; }
            .card-title { font-size: 1.6rem; font-weight: 700; color: white; margin-bottom: 10px; }
            .card-desc { font-size: 1rem; color: #94a3b8; line-height: 1.5; }

            /* تعديل الزر ليصبح شفافاً تماماً ومستقلاً فوق الحاوية */
            .clickable-overlay {
                position: relative;
                margin-top: -400px; /* يسحب الزر للأعلى ليغطي الـ UI */
                height: 400px;
                z-index: 10;
            }

            .stButton > button {
                width: 100% !important;
                height: 400px !important;
                background: transparent !important;
                border: none !important;
                color: transparent !important;
                box-shadow: none !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="main-header">
            <h1 class="title-text">SANAD AI Assistant</h1>
            <p style="color: #94a3b8; font-size: 1.2rem;">مساعدك الذكي في عالم الزراعة والتمويل</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    # وظيفة لإنشاء البطاقة لتكرار الكود بسهولة
    def create_clickable_card(col, icon, title, desc, key, chat_type):
        with col:
            # طبقة التصميم المرئي (UI Layer)
            st.markdown(f"""
                <div class="card-ui">
                    <div class="icon-box">{icon}</div>
                    <div class="card-title">{title}</div>
                    <div class="card-desc">{desc}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # طبقة التفاعل الشفافة (Interaction Layer)
            st.markdown('<div class="clickable-overlay">', unsafe_allow_html=True)
            if st.button(f"click_{key}", key=key):
                st.session_state.chat_type = chat_type
                st.session_state.page = "chat"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    create_clickable_card(col1, "🌱", "تمويل المحاصيل", "تقديم حلول تمويلية ذكية للمزارعين ودعم الإنتاج المستدام.", "crop", "agriculture")
    create_clickable_card(col2, "📈", "التمويل والقروض", "استكشف خيارات القروض والتسهيلات الائتمانية لمشاريعك.", "finance", "general")
    create_clickable_card(col3, "🐄", "الثروة الحيوانية", "دعم فني وتمويلي لمشاريع الإنتاج الحيواني والداجني.", "livestock", "agriculture")

# =========================
# 4. CHAT LOGIC (保持不变)
# =========================
def chat_page():
    st.markdown(f"<h1 style='text-align: right; color: #10b981;'>💬 مساعد {st.session_state.chat_type.upper()}</h1>", unsafe_allow_html=True)
    if st.button("⬅️ العودة للرئيسية"):
        st.session_state.page = "home"
        st.rerun()
    
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("اسأل SANAD..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            response = model.generate_content(f"أجب بالعربية: {prompt}")
            st.markdown(response.text)
            audio_path = text_to_audio(response.text)
            autoplay_audio(audio_path)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

# =========================
# 5. ROUTING
# =========================
if "page" not in st.session_state: st.session_state.page = "home"

if st.session_state.page == "home":
    home_page()
else:
    chat_page()
