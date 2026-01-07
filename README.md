# 🤖 PDF Chat Bot

A powerful AI-powered chatbot that allows you to have conversations with your PDF documents using **Claude Sonnet 4.5** via OpenRouter API.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 📄 **Multi-PDF Support** - Upload and chat with multiple PDF documents simultaneously
- 🤖 **Claude Sonnet 4.5** - Powered by Anthropic's advanced AI model via OpenRouter
- 🌐 **Multi-Language Support** - Ask questions and get responses in 12+ languages
- � **Text-to-Speech** - Listen to AI responses with built-in audio playback
- 💬 **Chat Interface** - Modern conversational UI with chat history
- 🚀 **Fast Embeddings** - Uses HuggingFace embeddings for efficient document search
- 📊 **Smart Caching** - Rate limiting and response caching for optimal performance

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core programming language |
| Streamlit | Web application framework |
| LangChain | LLM orchestration |
| FAISS | Vector similarity search |
| HuggingFace | Text embeddings |
| OpenRouter | AI model API gateway |
| Claude Sonnet 4.5 | Large language model |
| gTTS | Text-to-speech |

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/Prasad-Sable/pdf-info-retreiver.git
cd pdf-info-retreiver
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n pdfchat python=3.8 -y
conda activate pdfchat

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the root directory:

```ini
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

> 💡 Get your OpenRouter API key from [openrouter.ai](https://openrouter.ai)

### Step 5: Run the Application

```bash
streamlit run app.py
```

Open your browser and navigate to: **http://localhost:8501**

## 🚀 Usage

1. **Upload PDFs** - Use the sidebar to upload one or more PDF documents
2. **Process Documents** - Click "Process Documents" to index your files
3. **Ask Questions** - Type your questions in the chat input
4. **Get Responses** - Receive AI-powered answers based on your documents
5. **Listen** - Enable Text-to-Speech to hear the responses

## 🌍 Supported Languages

- 🇺🇸 English
- 🇪🇸 Spanish
- 🇫🇷 French
- 🇩🇪 German
- 🇮🇹 Italian
- 🇵🇹 Portuguese
- 🇷🇺 Russian
- 🇯🇵 Japanese
- 🇰🇷 Korean
- 🇮🇳 Hindi
- 🇮🇳 Marathi
- 🇸🇦 Arabic

## 📁 Project Structure

```
pdf-info-retreiver/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
├── faiss_index/       # Vector store (auto-generated)
├── README.md          # This file
└── LICENSE            # MIT License
```

## ⚙️ Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | ✅ Yes |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Anthropic](https://anthropic.com) for Claude AI
- [OpenRouter](https://openrouter.ai) for API gateway
- [Streamlit](https://streamlit.io) for the web framework
- [LangChain](https://langchain.com) for LLM orchestration
- [HuggingFace](https://huggingface.co) for embeddings

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/Prasad-Sable">Prasad Sable</a>
</p>