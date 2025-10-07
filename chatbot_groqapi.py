import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Session state initialization
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

# Functions
@st.cache_resource
def load_embeddings():
    """Load Vietnamese embedding model"""
    return HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder"
    )

def get_llm():
    """
    Get Groq LLM instance
    API key from environment variable or Streamlit secrets
    """
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    
    if not api_key:
        st.error("⚠️ Chưa cấu hình Groq API key!")
        st.info("""
        **Cách lấy API key miễn phí:**
        1. Truy cập: https://console.groq.com
        2. Đăng ký tài khoản (miễn phí)
        3. Tạo API key
        4. Thêm vào file `.env`: `GROQ_API_KEY=gsk_xxx`
        
        **Hoặc trong Streamlit Cloud:**
        Settings → Secrets → Thêm `GROQ_API_KEY = "gsk_xxx"`
        """)
        st.stop()
    
    return ChatGroq(
        model="llama-3.1-70b-versatile",  # Model 70B, mạnh hơn Vicuna-7B nhiều
        temperature=0,
        api_key=api_key,
        max_tokens=512
    )

def process_pdf(uploaded_file):
    """Process uploaded PDF and create RAG chain"""
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Split documents semantically
        semantic_splitter = SemanticChunker(
            embeddings=st.session_state.embeddings,
            buffer_size=1,
            breakpoint_threshold_type="percentile", 
            breakpoint_threshold_amount=95,
            min_chunk_size=500,
            add_start_index=True
        )
        
        docs = semantic_splitter.split_documents(documents)
        
        # Create vector database
        vector_db = Chroma.from_documents(
            documents=docs, 
            embedding=st.session_state.embeddings
        )
        retriever = vector_db.as_retriever()
        
        # Get prompt template
        prompt = hub.pull("rlm/rag-prompt")
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create RAG chain with Groq
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt 
            | get_llm()  # ← Dùng Groq API thay vì local model
            | StrOutputParser()
        )
        
        return rag_chain, len(docs)
    
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# UI
def main():
    st.set_page_config(
        page_title="PDF RAG Assistant", 
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF RAG Assistant")

    st.markdown("""
    **Ứng dụng AI giúp bạn hỏi đáp trực tiếp với nội dung tài liệu PDF bằng tiếng Việt**

    **Cách sử dụng đơn giản:**
    1. **Upload PDF** → Chọn file PDF từ máy tính và nhấn "Xử lý PDF"  
    2. **Đặt câu hỏi** → Nhập câu hỏi về nội dung tài liệu và nhận câu trả lời ngay lập tức

    ---
    """)

    # Load embeddings
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Đang tải embedding model..."):
            st.session_state.embeddings = load_embeddings()
            st.session_state.models_loaded = True
        st.success("✅ Embedding model đã sẵn sàng!")

    # Upload PDF
    uploaded_file = st.file_uploader("📄 Upload file PDF", type="pdf")
    
    if uploaded_file and st.button("🔄 Xử lý PDF", type="primary"):
        with st.spinner("⏳ Đang xử lý PDF..."):
            try:
                st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
                st.success(f"✅ Hoàn thành! Đã chia thành {num_chunks} chunks")
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý PDF: {str(e)}")

    # Q&A
    st.markdown("---")
    
    if st.session_state.rag_chain:
        question = st.text_input(
            "💬 Đặt câu hỏi về tài liệu:", 
            placeholder="VD: Tài liệu này nói về gì?"
        )
        
        if question:
            with st.spinner("🤔 Đang suy nghĩ..."):
                try:
                    output = st.session_state.rag_chain.invoke(question)
                    
                    # Extract answer
                    if 'Answer:' in output:
                        answer = output.split('Answer:')[1].strip()
                    else:
                        answer = output.strip()
                    
                    st.markdown("### 💡 Trả lời:")
                    st.write(answer)
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi trả lời: {str(e)}")
    else:
        st.info("ℹ️ Vui lòng upload và xử lý PDF trước khi đặt câu hỏi")

if __name__ == "__main__":
    main()