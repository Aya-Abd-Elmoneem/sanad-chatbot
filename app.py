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
# 3. ADVANCED UI - CARDS AS BUTTONS
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

            /* تنسيق الأعمدة */
            [data-testid="column"] {
                position: relative;
                margin-bottom: 20px;
            }

            /* تنسيق البطاقة */
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
                cursor: pointer;
            }

            /* تأثير hover على البطاقة */
            .card-ui:hover {
                border-color: #10b981;
                background: rgba(30, 41, 59, 0.6);
                transform: translateY(-8px);
                box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
            }

            .icon-box { 
                font-size: 4.5rem; 
                margin-bottom: 20px;
            }
            
            .card-title { 
                font-size: 1.6rem; 
                font-weight: 700; 
                color: white; 
                margin-bottom: 10px;
            }
            
            .card-desc { 
                font-size: 1rem; 
                color: #94a3b8; 
                line-height: 1.5;
            }

            /* تنسيق الأزرار المخفية */
            .card-button {
                width: 100%;
            }
            
            /* إخفاء نص الزر */
            .stButton > button {
                background: transparent !important;
                border: none !important;
                color: transparent !important;
                box-shadow: none !important;
                padding: 0 !important;
                margin: 0 !important;
                min-height: auto !important;
            }
            
            .stButton > button:hover {
                background: transparent !important;
                border: none !important;
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

    def create_card(col, icon, title, desc, key, chat_type):
        with col:
            # إنشاء حاوية للبطاقة
            card_container = st.container()
            
            with card_container:
                # عرض البطاقة باستخدام HTML
                st.markdown(f"""
                    <div class="card-ui">
                        <div class="icon-box">{icon}</div>
                        <div class="card-title">{title}</div>
                        <div class="card-desc">{desc}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # زر مخفي يغطي البطاقة - مع إزالة use_container_width
                if st.button(" ", key=key):
                    # تخزين البيانات في session state
                    st.session_state.chat_type = chat_type
                    st.session_state.page = "chat"
                    st.rerun()

    # إنشاء البطاقات الثلاث
    create_card(col1, "🌱", "تمويل المحاصيل", "حلول تمويلية ذكية للمزارعين ودعم الإنتاج المستدام.", "crop_btn", "agriculture")
    create_card(col2, "📈", "التمويل والقروض", "استكشف خيارات القروض والتسهيلات الائتمانية لمشاريعك.", "finance_btn", "general")
    create_card(col3, "🐄", "الثروة الحيوانية", "دعم فني وتمويلي لمشاريع الإنتاج الحيواني والداجني.", "livestock_btn", "agriculture")

# =========================
# 4. CHAT PAGE - SIMPLE VERSION
# =========================
def chat_page():
    # زر العودة
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅️ العودة للرئيسية", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    
    with col2:
        st.markdown(f"<h1 style='text-align: right; color: #10b981; margin: 0;'>💬 مساعد {st.session_state.chat_type}</h1>", unsafe_allow_html=True)
    
    # واجهة الدردشة البسيطة
    st.markdown("---")
    
    # تهيئة session state للدردشة
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # عرض رسائل الدردشة
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # إدخال المستخدم
    if prompt := st.chat_input("اكتب سؤالك هنا..."):
        # إضافة رسالة المستخدم
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # إنشاء رد من المساعد
        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
                try:
                    # استخدام نموذج Gemini للرد
                    response = model.generate_content(prompt)
                    assistant_response = response.text
                    st.markdown(assistant_response)
                    
                    # تحويل النص إلى صوت
                    audio_file = text_to_audio(assistant_response)
                    autoplay_audio(audio_file)
                    
                    # حفظ الرد
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                except Exception as e:
                    st.error(f"حدث خطأ: {str(e)}")

# =========================
# 5. MAIN ROUTING
# =========================
# تهيئة session state
if "page" not in st.session_state:
    st.session_state.page = "home"

# توجيه الصفحات
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "chat":
    chat_page()
