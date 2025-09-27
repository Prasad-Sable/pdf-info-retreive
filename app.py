import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import base64
import tempfile
from gtts import gTTS
import io

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="Document Genie with TTS", layout="wide")

# Language mapping for TTS
LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Chinese (Mandarin)': 'zh',
    'Arabic': 'ar',
    'Dutch': 'nl',
    'Turkish': 'tr',
    'Polish': 'pl',
    'Bengali': 'bn',
    'Gujarati': 'gu',
    'Kannada': 'kn',
    'Malayalam': 'ml',
    'Marathi': 'mr',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Urdu': 'ur',
    'Thai': 'th',
    'Vietnamese': 'vi',
    'Indonesian': 'id',
    'Filipino': 'tl'
}

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    """Create vector store using Gemini embeddings"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        task_type="retrieval_document"
    )
    
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    return vector_store

def get_conversational_chain():
    """Create conversational chain using Gemini"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say "answer is not available in the context".
    Don't provide wrong answers.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def text_to_speech_bytes(text, language='en', slow=False):
    """Convert text to speech and return audio bytes"""
    try:
        # Clean text for TTS (remove markdown formatting)
        cleaned_text = text.replace("*", "").replace("#", "").replace("```", "").strip()
        
        # Limit text length for TTS (gTTS has limits)
        if len(cleaned_text) > 5000:
            cleaned_text = cleaned_text[:5000] + "..."
            
        if not cleaned_text:
            return None
            
        # Create TTS object
        tts = gTTS(text=cleaned_text, lang=language, slow=slow)
        
        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.read()
        
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def display_audio_player(audio_bytes, key=None):
    """Display audio player using Streamlit's audio component"""
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")

def clear_chat_history():
    """Clear chat history"""
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
    ]

def user_input(user_question):
    """Process user question and generate response"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            task_type="retrieval_query"
        )
        
        new_db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )
        
        return response["output_text"]
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    """Main Streamlit application"""
    st.header("🎤 Chat with PDF using Gemini + Text-to-Speech")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
        ]
    
    # Sidebar for TTS settings
    st.sidebar.header("🔊 Text-to-Speech Settings")
    
    # Language selection
    selected_language_name = st.sidebar.selectbox(
        "Select Language for TTS:",
        list(LANGUAGES.keys()),
        index=0  # Default to English
    )
    selected_language_code = LANGUAGES[selected_language_name]
    
    # Speech speed
    slow_speech = st.sidebar.checkbox("Slow Speech", value=False)
    
    # Auto-play setting
    auto_tts = st.sidebar.checkbox("Auto-play responses", value=True)
    
    st.sidebar.markdown("---")
    
    # File uploader
    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDF Files and Click on the Submit & Process Button", 
        accept_multiple_files=True
    )
    
    if st.sidebar.button("Submit & Process"):
        with st.spinner("Processing PDFs..."):
            try:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Done! You can now ask questions about your PDFs.")
            except Exception as e:
                st.error(f"❌ Error processing PDFs: {str(e)}")
    
    # Clear chat history button
    st.sidebar.button('🗑️ Clear Chat History', on_click=clear_chat_history)
    
    # Display chat messages with TTS
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Add TTS for assistant messages (except welcome message)
            if (message["role"] == "assistant" and 
                message["content"] != "Upload some PDFs and ask me a question" and
                "error occurred" not in message["content"].lower()):
                
                # Create a unique key for each message
                tts_key = f"tts_{i}"
                
                # TTS button and audio player
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    if st.button("🔊 Play", key=f"btn_{tts_key}"):
                        with st.spinner("Generating audio..."):
                            audio_bytes = text_to_speech_bytes(
                                message["content"], 
                                selected_language_code, 
                                slow_speech
                            )
                            if audio_bytes:
                                # Store audio in session state for this message
                                st.session_state[f"audio_{tts_key}"] = audio_bytes
                            else:
                                st.error("Failed to generate audio")
                
                with col2:
                    # Display audio player if audio exists for this message
                    if f"audio_{tts_key}" in st.session_state:
                        display_audio_player(st.session_state[f"audio_{tts_key}"])
    
    # User input
    if prompt := st.chat_input("Ask a question about your PDFs..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                st.write(response)
                
                # Auto-generate TTS if enabled and response is valid
                if (auto_tts and response and 
                    "error occurred" not in response.lower() and
                    "not available in the context" not in response.lower()):
                    
                    with st.spinner("Generating audio..."):
                        audio_bytes = text_to_speech_bytes(
                            response, 
                            selected_language_code, 
                            slow_speech
                        )
                        if audio_bytes:
                            st.success("🔊 Audio ready!")
                            display_audio_player(audio_bytes)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the interface
        st.rerun()
    
    # Help sections
    with st.expander("ℹ️ Supported Languages"):
        st.write("This app supports text-to-speech in the following languages:")
        
        # Display languages in a nice format
        langs = list(LANGUAGES.keys())
        for i in range(0, len(langs), 3):
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            for j, col in enumerate(cols):
                if i + j < len(langs):
                    col.write(f"• {langs[i + j]}")
    
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Steps:
        1. **Upload PDFs**: Use the sidebar to upload your PDF documents
        2. **Process**: Click "Submit & Process" to index your documents
        3. **Select Language**: Choose your preferred TTS language from the sidebar
        4. **Ask Questions**: Type questions about your PDFs in the chat
        5. **Listen**: Enable auto-play or click "🔊 Play" buttons for audio
        
        ### Features:
        - 🌍 **25+ Languages supported**
        - 🎛️ **Speed control** (normal/slow)
        - 🔄 **Auto-play option**
        - 📱 **Works on all devices**
        
        ### Tips:
        - Keep questions clear and specific
        - Audio works best with shorter responses
        - Use different languages to practice pronunciation
        """)

if __name__ == '__main__':
    main()