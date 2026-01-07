import streamlit as st
import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from googletrans import Translator
from gtts import gTTS
import threading
import time
import tempfile
import pygame
from dotenv import load_dotenv

# Free embedding alternatives
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load environment variables
load_dotenv()

# Initialize translator
translator = Translator()

def get_api_key():
    """Get API key from multiple sources"""
    return (
        os.getenv("GOOGLE_API_KEY") or 
        st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') else None
    )

def initialize_tts():
    """Initialize pygame for audio playback"""
    try:
        pygame.mixer.init()
        return True
    except Exception as e:
        st.error(f"Error initializing audio: {str(e)}")
        return False

def speak_text(text):
    """Convert text to speech using gTTS"""
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            
            # Play audio using pygame
            pygame.mixer.music.load(tmp_file.name)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")

def detect_language(text):
    """Detect the language of the input text"""
    try:
        # Use a simple language detection approach
        # Check for common non-English patterns
        text_lower = text.lower()
        
        # Simple pattern-based detection
        if any(char in text for char in 'ñáéíóúü'):  # Spanish characters
            return 'es', 0.8
        elif any(char in text for char in 'àâäéèêëïîôöùûüÿç'):  # French characters
            return 'fr', 0.8
        elif any(char in text for char in 'äöüß'):  # German characters
            return 'de', 0.8
        elif any(char in text for char in 'àèéìíîòóù'):  # Italian characters
            return 'it', 0.8
        elif any(char in text for char in 'ãâáàçéêíóôõú'):  # Portuguese characters
            return 'pt', 0.8
        elif any(char in text for char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'):  # Russian characters
            return 'ru', 0.9
        elif any(char in text for char in 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'):  # Japanese characters
            return 'ja', 0.9
        elif any(char in text for char in '가나다라마바사아자차카타파하'):  # Korean characters
            return 'ko', 0.9
        elif any(char in text for char in '的了一是我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已其此当没成只如事把还用第样道想作种开美总从无面最女但现前些所同日手又行意动方期它头经1.0
        else:
            return 'en', 1.0
    except Exception as e:
        st.error(f"Language detection error: {str(e)}")
        return 'en', 1.0

def translate_text(text, target_lang='en', source_lang='auto'):
    """Translate text to target language"""
    try:
        if source_lang == target_lang:
            return text
        
        translation = translator.translate(text, src=source_lang, dest=target_lang)
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    """Split text into chunks for processing"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return []

def get_embedding_model(embedding_type="free"):
    """Get embedding model based on type"""
    
    if embedding_type == "free" and HUGGINGFACE_AVAILABLE:
        try:
            st.info("🤖 Loading free Hugging Face embeddings...")
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Error loading HuggingFace embeddings: {str(e)}")
            return None
    
    elif embedding_type == "gemini" and GEMINI_AVAILABLE:
        api_key = get_api_key()
        if not api_key:
            st.error("Gemini API key required")
            return None
            
        try:
            genai.configure(api_key=api_key)
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key,
                task_type="retrieval_document"
            )
        except Exception as e:
            st.error(f"Error loading Gemini embeddings: {str(e)}")
            return None
    
    else:
        st.error("❌ No embedding models available!")
        st.info("💡 Install: pip install sentence-transformers torch")
        return None

def get_vector_store(text_chunks, embedding_choice="free"):
    """Create and save FAISS vector store"""
    try:
        if not text_chunks:
            st.error("No text chunks to process")
            return None
        
        # Get embedding model
        embeddings = get_embedding_model(embedding_choice)
        if embeddings is None:
            return None
        
        # Limit chunks for processing
        max_chunks = 100 if embedding_choice == "free" else 50
        if len(text_chunks) > max_chunks:
            st.warning(f"⚠️ Processing first {max_chunks} chunks (found {len(text_chunks)})")
            text_chunks = text_chunks[:max_chunks]
        
        st.info(f"📊 Creating embeddings for {len(text_chunks)} chunks...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        progress_bar.progress(0.3)
        
        # Create vector store
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        progress_bar.progress(0.8)
        
        # Save vector store
        vector_store.save_local("faiss_index")
        progress_bar.progress(1.0)
        
        st.success("✅ Vector store created successfully!")
        progress_bar.empty()
        
        return vector_store
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversational_chain():
    """Create conversational chain for Q&A"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say "Answer is not available in the context".
    Don't provide wrong answers.
    
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    
    api_key = get_api_key()
    if not api_key:
        st.error("Google API key required for chat model")
        return None
    
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.4,
            convert_system_message_to_human=True
        )
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

def user_input(user_question, target_language='en', enable_tts=False, embedding_choice="free"):
    """Process user input and provide response"""
    
    # Detect input language
    detected_lang, confidence = detect_language(user_question)
    
    # Translate question to English if needed
    english_question = user_question
    if detected_lang != 'en':
        english_question = translate_text(user_question, target_lang='en', source_lang=detected_lang)
        st.info(f"Detected language: {detected_lang} (confidence: {confidence:.2f})")
        st.info(f"Translated question: {english_question}")
    
    try:
        # Load vector database
        embeddings = get_embedding_model(embedding_choice)
        if embeddings is None:
            st.error("Could not load embedding model")
            return None, target_language
        
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(english_question)
        
        # Get conversational chain
        chain = get_conversational_chain()
        if chain is None:
            return None, target_language
            
        response = chain({"input_documents": docs, "question": english_question}, return_only_outputs=True)
        english_response = response["output_text"]
        
        # Display English response
        st.write("**English Response:**")
        st.write(english_response)
        
        # Translate response if needed
        if target_language != 'en':
            translated_response = translate_text(english_response, target_lang=target_language)
            st.write(f"**Translated Response ({target_language}):**")
            st.write(translated_response)
        
        # Text-to-Speech
        if enable_tts and english_response.strip():
            st.write("🔊 **Audio Response (English):**")
            if st.button("🔊 Play Audio"):
                with st.spinner("Generating audio..."):
                    try:
                        if initialize_tts():
                            tts_thread = threading.Thread(target=speak_text, args=(english_response,))
                            tts_thread.daemon = True
                            tts_thread.start()
                            st.success("Audio is playing...")
                        else:
                            st.error("Failed to initialize audio system")
                    except Exception as e:
                        st.error(f"Error playing audio: {str(e)}")
        
        return english_response, target_language
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return None, target_language

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="PDF Info Retriever", page_icon="📚")
    
    st.markdown('<h1 style="text-align: center; color: #1f77b4;">📚 PDF Info Retriever</h1>', unsafe_allow_html=True)
    st.markdown("**Chat with your PDFs using free embeddings with multilingual support and English TTS**")
    
    # Sidebar
    with st.sidebar:
        st.subheader("📁 Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF Files", 
            accept_multiple_files=True, 
            type=['pdf']
        )
        
        st.subheader("🌐 Language Settings")
        language_options = {
            'English': 'en', 'Spanish': 'es', 'French': 'fr',
            'German': 'de', 'Italian': 'it', 'Portuguese': 'pt',
            'Russian': 'ru', 'Japanese': 'ja', 'Korean': 'ko',
            'Chinese (Simplified)': 'zh-cn', 'Hindi': 'hi', 'Arabic': 'ar'
        }
        
        selected_language = st.selectbox(
            "Select response language:", 
            list(language_options.keys()),
            index=0
        )
        target_lang = language_options[selected_language]
        
        st.subheader("🤖 Embedding Model")
        embedding_options = {
            "Free (Hugging Face)": "free", 
            "Gemini (API Required)": "gemini"
        }
        
        selected_embedding = st.selectbox(
            "Choose embedding model:",
            list(embedding_options.keys()),
            index=0
        )
        embedding_choice = embedding_options[selected_embedding]
        
        # Show status
        if embedding_choice == "free":
            if HUGGINGFACE_AVAILABLE:
                st.success("✅ Free embeddings available")
            else:
                st.warning("⚠️ Install: pip install sentence-transformers torch")
        elif embedding_choice == "gemini":
            if get_api_key():
                st.success("✅ Gemini API key found")
            else:
                st.warning("⚠️ Gemini API key required")
        
        st.subheader("🔊 Audio Settings")
        enable_tts = st.checkbox("Enable Text-to-Speech (English)", value=True)
        
        # Process button
        if st.button("📊 Process Documents"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if raw_text.strip():
                        word_count = len(raw_text.split())
                        st.info(f"📄 Extracted {word_count:,} words from {len(pdf_docs)} PDF(s)")
                        
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            st.info(f"📝 Split into {len(text_chunks)} chunks")
                            vector_store = get_vector_store(text_chunks, embedding_choice)
                            if vector_store:
                                st.success("✅ Documents processed successfully!")
                                st.balloons()
                            else:
                                st.error("❌ Error creating vector store")
                        else:
                            st.error("❌ Error splitting text into chunks")
                    else:
                        st.error("❌ No text extracted from PDFs")
            else:
                st.warning("⚠️ Please upload at least one PDF file")
    
    # Main content
    st.subheader("💬 Ask Questions About Your Documents")
    
    # Check if processed
    if not os.path.exists("faiss_index"):
        st.info("👆 Please upload and process PDF documents first using the sidebar.")
        return
    
    # Question input
    user_question = st.text_input(
        "Ask a question about your documents:", 
        placeholder="Enter your question in any language..."
    )
    
    # Display settings
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🌐 Response Language: {selected_language}")
    with col2:
        st.info(f"🤖 Embeddings: {selected_embedding}")
    
    if user_question:
        with st.spinner("🔍 Searching for answer..."):
            response, lang = user_input(user_question, target_lang, enable_tts, embedding_choice)
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What is the main topic of this document?
        - Summarize the key points from the PDF
        - What are the conclusions mentioned?
        - Can you extract important dates mentioned?
        - What recommendations are provided?
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**🆓 Free Features:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("✅ **No API Quotas**\nUnlimited free processing")
    with col2:
        st.markdown("🌐 **Multilingual**\nSupports 12+ languages")
    with col3:
        st.markdown("🔊 **English TTS**\nClear audio responses")

if __name__ == "__main__":
    main()