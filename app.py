import streamlit as st
import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
from gtts import gTTS
import threading
import time
import tempfile
import base64
from dotenv import load_dotenv
import requests
import json
from datetime import datetime, timedelta
import hashlib
import io

# Translation imports
try:
    from deep_translator import GoogleTranslator, MyMemoryTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    try:
        from googletrans import Translator
        DEEP_TRANSLATOR_AVAILABLE = False
        FALLBACK_TRANSLATOR = True
    except ImportError:
        DEEP_TRANSLATOR_AVAILABLE = False
        FALLBACK_TRANSLATOR = False

# Free embedding alternatives
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Load environment variables
load_dotenv()

# Initialize translator
if DEEP_TRANSLATOR_AVAILABLE:
    # Using deep-translator for better reliability
    translator = None
elif FALLBACK_TRANSLATOR:
    # Fallback to googletrans
    translator = Translator()
else:
    translator = None

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# Rate limiting configuration
class RateLimiter:
    def __init__(self, max_requests_per_minute=5):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.request_cache = {}
    
    def can_make_request(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < timedelta(minutes=1)]
        return len(self.requests) < self.max_requests
    
    def add_request(self):
        self.requests.append(datetime.now())
    
    def get_cache_key(self, prompt, context):
        """Create cache key for request"""
        content = f"{prompt}_{context[:100]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_response(self, cache_key):
        """Get cached response if available and recent (within 1 hour)"""
        if cache_key in self.request_cache:
            response, timestamp = self.request_cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=1):
                return response
        return None
    
    def cache_response(self, cache_key, response):
        """Cache API response"""
        self.request_cache[cache_key] = (response, datetime.now())

# Initialize rate limiter
rate_limiter = RateLimiter(max_requests_per_minute=3)  # Conservative limit

def get_openrouter_api_key():
    """Get OpenRouter API key from multiple sources"""
    return (
        os.getenv("OPENROUTER_API_KEY") or 
        st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, 'secrets') else None
    )

def generate_audio(text):
    """Generate audio from text using gTTS and return base64 encoded audio"""
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode()
        
        return audio_base64
        
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def display_audio_player(audio_base64):
    """Display an audio player with the given base64 audio"""
    if audio_base64:
        audio_html = f"""
        <audio controls style="width: 100%;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

def call_ai_api(prompt, context="", api_key=None, model_choice="grok", retry_count=0, max_retries=3):
    """Call AI model via OpenRouter API with rate limiting and retry logic"""
    if not api_key:
        return None, "API key not found", "❌"
    
    # Create cache key with model info
    cache_key = rate_limiter.get_cache_key(f"{model_choice}_{prompt}", context)
    
    # Check cache first
    cached_response = rate_limiter.get_cached_response(cache_key)
    if cached_response:
        return cached_response, "Cached Response", "📱"
    
    # Check rate limit
    if not rate_limiter.can_make_request():
        return None, "Rate limited", "⏳"
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app.com",
        "X-Title": "PDF Info Retriever"
    }
    
    # Model configuration
    models = {
        "grok": {
            "name": "anthropic/claude-sonnet-4.5",
            "display_name": "Claude Sonnet 4.5",
            "max_tokens": 1000,
            "emoji": "🤖"
        },
        "gemini": {
            "name": "anthropic/claude-sonnet-4.5", 
            "display_name": "Claude Sonnet 4.5",
            "max_tokens": 800,
            "emoji": "🚀"
        }
    }
    
    selected_model = models.get(model_choice, models["grok"])
    
    # Combine context and prompt for better results
    full_prompt = f"""Context from PDF document:
{context[:3000]}  

User Request: {prompt}

Please provide a comprehensive and accurate response based on the context provided."""
    
    data = {
        "model": selected_model["name"],
        "messages": [
            {
                "role": "user", 
                "content": full_prompt
            }
        ],
        "max_tokens": selected_model["max_tokens"],
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        # Add request to rate limiter
        rate_limiter.add_request()
        
        # Make request with timeout
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 429:
            return None, "Rate limited by API", "⏳"
        elif response.status_code == 402:
            return None, "Credit limit reached", "💳"
        elif response.status_code in [500, 502, 503, 504]:
            if retry_count < max_retries:
                wait_time = (2 ** retry_count) + 1
                time.sleep(wait_time)
                return call_ai_api(prompt, context, api_key, model_choice, retry_count + 1, max_retries)
            else:
                return None, "Server error", "❌"
        
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            api_response = result['choices'][0]['message']['content']
            # Cache successful response
            rate_limiter.cache_response(cache_key, api_response)
            return api_response, selected_model["display_name"], selected_model["emoji"]
        else:
            return None, "No response from API", "❌"
            
    except requests.exceptions.Timeout:
        return None, "Request timeout", "⏰"
    except requests.exceptions.ConnectionError:
        return None, "Connection error", "🌐"
    except Exception as e:
        return None, f"Error: {str(e)}", "❌"

def detect_language(text):
    """Detect the language of the input text"""
    try:
        text_lower = text.lower()
        
        # Simple pattern-based detection
        if any(char in text for char in 'ñáéíóúü'):
            return 'es', 0.8
        elif any(char in text for char in 'àâäéèêëïîôöùûüÿç'):
            return 'fr', 0.8
        elif any(char in text for char in 'äöüß'):
            return 'de', 0.8
        elif any(char in text for char in 'àèéìíîòóù'):
            return 'it', 0.8
        elif any(char in text for char in 'ãâáàçéêíóôõú'):
            return 'pt', 0.8
        elif any(char in text for char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'):
            return 'ru', 0.9
        elif any(char in text for char in 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'):
            return 'ja', 0.9
        elif any(char in text for char in '가나다라마바사아자차카타파하'):
            return 'ko', 0.9
        elif any(char in text for char in 'अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह'):  # Marathi/Hindi Devanagari
            return 'mr', 0.9
        else:
            return 'en', 1.0
    except Exception as e:
        return 'en', 1.0

def translate_text(text, target_lang='en', source_lang='auto'):
    """Translate text using multiple translation backends with robust error handling"""
    try:
        if target_lang == source_lang or target_lang == 'en' and source_lang == 'auto':
            return text
        
        if not text or text.strip() == "":
            return text
        
        # Limit text length for better performance
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        translated_text = None
        
        # Method 1: Try deep-translator (most reliable)
        if DEEP_TRANSLATOR_AVAILABLE:
            try:
                # Try GoogleTranslator from deep-translator first
                google_translator = GoogleTranslator(source=source_lang, target=target_lang)
                translated_text = google_translator.translate(text)
                
                if translated_text and translated_text.strip() and translated_text != text:
                    return translated_text
                    
            except Exception as e:
                st.warning(f"Google Translator failed: {str(e)}")
            
            try:
                # Fallback to MyMemoryTranslator
                mymemory_translator = MyMemoryTranslator(source=source_lang, target=target_lang)
                translated_text = mymemory_translator.translate(text)
                
                if translated_text and translated_text.strip() and translated_text != text:
                    return translated_text
                    
            except Exception as e:
                st.warning(f"MyMemory Translator failed: {str(e)}")
        
        # Method 2: Try legacy googletrans (fallback)
        elif FALLBACK_TRANSLATOR and translator:
            try:
                translation = translator.translate(text, src=source_lang, dest=target_lang)
                if translation and hasattr(translation, 'text') and translation.text:
                    return translation.text
            except Exception as e:
                st.warning(f"Legacy Google Translate failed: {str(e)}")
        
        # Method 3: Try OpenRouter translation as last resort (if API key available)
        openrouter_key = get_openrouter_api_key()
        if openrouter_key:
            try:
                translation_prompt = f"Translate this text from {source_lang} to {target_lang}. Only return the translated text, nothing else:\n\n{text}"
                
                result = call_ai_api(
                    prompt=translation_prompt,
                    context="",
                    api_key=openrouter_key,
                    model_choice="grok"
                )
                
                if result[0] and result[0].strip():
                    return result[0].strip()
                    
            except Exception as e:
                pass  # Silent fallback
        
        # If all methods fail, return original text
        st.info(f"📝 Translation unavailable, showing original text")
        return text
        
    except Exception as e:
        st.warning(f"Translation error: {str(e)}")
        return text

def get_language_name(code):
    """Get language name from code"""
    lang_names = {
        'en': 'English', 'hi': 'Hindi', 'mr': 'Marathi', 'es': 'Spanish',
        'fr': 'French', 'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
        'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic'
    }
    return lang_names.get(code, code.upper())

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

def get_embedding_model():
    """Get embedding model - only free HuggingFace model available"""
    
    if HUGGINGFACE_AVAILABLE:
        try:
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Error loading HuggingFace embeddings: {str(e)}")
            return None
    else:
        st.error("❌ Install: pip install sentence-transformers")
        return None

def get_vector_store(text_chunks):
    """Create and save FAISS vector store"""
    try:
        if not text_chunks:
            st.error("No text chunks to process")
            return None
        
        # Get embedding model
        embeddings = get_embedding_model()
        if embeddings is None:
            return None
        
        # Limit chunks for processing
        max_chunks = 100
        if len(text_chunks) > max_chunks:
            st.warning(f"⚠️ Processing first {max_chunks} chunks (found {len(text_chunks)})")
            text_chunks = text_chunks[:max_chunks]
        
        # Create progress bar
        progress_bar = st.progress(0)
        progress_bar.progress(0.3)
        
        # Create vector store
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        progress_bar.progress(0.8)
        
        # Save vector store
        vector_store.save_local("faiss_index")
        progress_bar.progress(1.0)
        
        progress_bar.empty()
        return vector_store
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def process_question(user_question, target_language='en', model_choice="grok"):
    """Process user input and return response"""
    
    openrouter_key = get_openrouter_api_key()
    
    if not openrouter_key:
        return "🔑 **OpenRouter API Key Required!** Please add your API key to continue.", None, None
    
    # Detect input language
    detected_lang, confidence = detect_language(user_question)
    
    # Translate question to English if needed
    english_question = user_question
    if detected_lang != 'en' and confidence > 0.7:
        try:
            translated = translate_text(user_question, target_lang='en', source_lang=detected_lang)
            if translated and translated != user_question and len(translated.strip()) > 0:
                english_question = translated
                lang_name = get_language_name(detected_lang)
                st.info(f"🌐 Detected {lang_name} → Translated to English")
            else:
                st.info(f"🌐 Detected {get_language_name(detected_lang)}, processing as-is")
        except Exception as e:
            st.warning(f"Input translation failed: {str(e)}")
    
    try:
        # Load vector database
        embeddings = get_embedding_model()
        if embeddings is None:
            return "Could not load embedding model. Please install sentence-transformers.", None, None
        
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(english_question, k=5)
        
        # Prepare context
        context_chunks = []
        for doc in docs:
            if hasattr(doc, 'page_content'):
                doc_text = doc.page_content.strip()
            elif hasattr(doc, 'content'):
                doc_text = doc.content.strip()
            elif isinstance(doc, dict) and 'page_content' in doc:
                doc_text = doc['page_content'].strip()
            elif isinstance(doc, dict) and 'content' in doc:
                doc_text = doc['content'].strip()
            else:
                doc_text = str(doc).strip()
            
            if doc_text:
                context_chunks.append(doc_text)
        
        # Combine context
        combined_context = "\n\n".join(context_chunks[:4])[:3000]
        
        # Call AI API
        result = call_ai_api(
            prompt=english_question,
            context=combined_context,
            api_key=openrouter_key,
            model_choice=model_choice
        )
        
        if result[0] is None:
            return f"❌ {result[1]}", None, None
        
        english_response, model_name, model_emoji = result
        
        # Translate response if needed
        final_response = english_response
        if target_language != 'en':
            try:
                translated_response = translate_text(english_response, target_lang=target_language, source_lang='en')
                if translated_response and translated_response != english_response and len(translated_response.strip()) > 0:
                    final_response = translated_response
                    target_lang_name = get_language_name(target_language)
                    st.success(f"✅ Response translated to {target_lang_name}")
                else:
                    st.info("📝 Showing response in English")
            except Exception as e:
                st.warning(f"Response translation failed: {str(e)}")
        
        return final_response, model_name, model_emoji
        
    except Exception as e:
        return f"Error processing question: {str(e)}", None, None

def display_chat_message(role, content, model_info=None, audio_data=None):
    """Display a chat message with proper styling"""
    
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(content)
    else:
        avatar = "🤖" if model_info and "Grok" in model_info else "🚀"
        with st.chat_message("assistant", avatar=avatar):
            if model_info:
                st.caption(f"Responded by {model_info}")
            st.write(content)
            
            # Add audio player if available
            if audio_data:
                st.caption("🔊 Audio Response:")
                display_audio_player(audio_data)

def main():
    """Main Streamlit application with chatbot UI"""
    st.set_page_config(page_title="PDF Chat Bot", page_icon="🤖", layout="wide")
    
    # Custom CSS for better chatbot styling
    st.markdown("""
    <style>
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    
    .stChatMessage {
        margin-bottom: 1rem;
    }
    
    .sidebar .sidebar-content {
        padding-top: 2rem;
    }
    
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 PDF Chat Bot</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Chat with your PDFs using Claude Sonnet 4.5</p>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("🤖 AI Model")
        model_options = {
            "🤖 Claude Sonnet 4.5": "grok"
        }
        
        selected_model_display = st.selectbox(
            "Choose AI Model:",
            list(model_options.keys()),
            index=0
        )
        selected_model = model_options[selected_model_display]
        
        # Language settings
        st.subheader("🌐 Language")
        language_options = {
            'English': 'en', 'Spanish': 'es', 'French': 'fr',
            'German': 'de', 'Italian': 'it', 'Portuguese': 'pt',
            'Russian': 'ru', 'Japanese': 'ja', 'Korean': 'ko',
            'Hindi': 'hi', 'Arabic': 'ar', 'Marathi': 'mr'
        }
        
        selected_language = st.selectbox(
            "Response Language:",
            list(language_options.keys()),
            index=0
        )
        target_lang = language_options[selected_language]
        
        # TTS settings
        st.subheader("🔊 Audio")
        enable_tts = st.checkbox("Enable Text-to-Speech", value=True)
        
        
        openrouter_key = get_openrouter_api_key()
        
        
        # Document upload section
        st.subheader("📁 Upload Documents")
        pdf_docs = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload your PDF documents to start chatting"
        )
        
        if st.button("📊 Process Documents", type="primary"):
            if not openrouter_key:
                st.error("❌ API key required")
            elif pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if raw_text.strip():
                        word_count = len(raw_text.split())
                        st.success(f"📄 Extracted {word_count:,} words")
                        
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            st.info(f"📝 Created {len(text_chunks)} chunks")
                            vector_store = get_vector_store(text_chunks)
                            if vector_store:
                                st.session_state.documents_processed = True
                                st.success("✅ Documents ready!")
                                st.balloons()
                            else:
                                st.error("❌ Vector store error")
                        else:
                            st.error("❌ Text chunking failed")
                    else:
                        st.error("❌ No text extracted")
            else:
                st.warning("⚠️ Upload PDF files first")
        
        # Clear chat button
        st.divider()
        if st.button("🗑️ Clear Chat", help="Clear all chat messages"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    if not openrouter_key:
        st.error("🔑 **OpenRouter API Key Required**")
        st.info("Please add your OpenRouter API key to the environment variables to start chatting with your PDFs.")
        return
    
    if not st.session_state.documents_processed:
        st.info("📁 **Upload and Process Documents First**")
        st.write("Please upload your PDF documents using the sidebar to start chatting.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(
            message["role"], 
            message["content"], 
            message.get("model_info"),
            message.get("audio_data")
        )
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt
        })
        
        # Display user message
        display_chat_message("user", prompt)
        
        # Generate response
        with st.spinner(f"🤔 {selected_model_display} is thinking..."):
            response, model_name, model_emoji = process_question(
                prompt, 
                target_lang, 
                selected_model
            )
        
        # Generate audio if TTS is enabled
        audio_data = None
        if enable_tts and response and model_name:
            with st.spinner("🔊 Generating audio..."):
                # Limit audio to first 200 words
                tts_text = ' '.join(response.split()[:200])
                if len(response.split()) > 200:
                    tts_text += "... (truncated for audio)"
                audio_data = generate_audio(tts_text)
        
        # Display assistant response
        if model_name:
            full_model_info = f"{model_emoji} {model_name}"
        else:
            full_model_info = None
            
        display_chat_message("assistant", response, full_model_info, audio_data)
        
        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "model_info": full_model_info,
            "audio_data": audio_data
        })
    
    # Quick action buttons
    if st.session_state.documents_processed:
        st.divider()
        st.write("**💡 Quick Actions:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📝 Summarize", help="Get a summary of the documents"):
                st.chat_input("Please provide a comprehensive summary of the main points in the document.")
        
        with col2:
            if st.button("❓ Generate Questions", help="Generate questions about the content"):
                st.chat_input("Generate 5 important questions based on this document.")
        
        with col3:
            if st.button("🔍 Key Points", help="Extract key points"):
                st.chat_input("What are the key points and main takeaways from this document?")
        
        with col4:
            if st.button("📊 Analysis", help="Analyze the content"):
                st.chat_input("Provide a detailed analysis of the content and its implications.")

if __name__ == "__main__":
    main()