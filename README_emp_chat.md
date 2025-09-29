# 🧠 Empathic Mental Health Chatbot

A compassionate AI chatbot that provides mental health support using conversation memory and research-based knowledge from PDF documents.

## ✨ Features

- **🔄 Conversation Memory**: Remembers previous exchanges for contextual responses
- **📚 Knowledge-Based**: Grounds responses in mental health research papers
- **💝 Empathetic Design**: Specifically prompt-engineered for compassionate mental health support
- **🌐 Web Interface**: Easy-to-use chat interface via Gradio
- **🔍 Semantic Search**: Finds relevant information using vector similarity
- **⚡ Fast Performance**: Cached vector database for quick responses

## 🏗️ Architecture

```
PDF Documents → Text Chunks → Vector Embeddings → ChromaDB
                                      ↓
User Question → Context Retrieval → LLM (LLaMA 3.3) → Empathetic Response
                                      ↑
                              Conversation History
```

### Technology Stack

- **LLM**: Groq's LLaMA 3.3 70B model
- **Vector Database**: ChromaDB with HuggingFace embeddings
- **PDF Processing**: LangChain community loaders
- **Web Interface**: Gradio
- **Architecture**: RAG (Retrieval-Augmented Generation) with conversation memory

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API key (free at [Groq Console](https://console.groq.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Empathic-Chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_emp_chat.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GROQ_API=your_groq_api_key_here
   ```

4. **Add mental health documents**
   Place PDF files in `./content/data/` directory:
   ```
   content/
   └── data/
       ├── mental_health_guide.pdf
       ├── therapy_techniques.pdf
       └── coping_strategies.pdf
   ```

5. **Run the chatbot**
   ```bash
   python emp_chat.py
   ```

The chatbot will be available at `http://127.0.0.1:7860`

## 📁 Project Structure

```
Empathic-Chatbot/
├── emp_chat.py                 # Main chatbot application
├── requirements_emp_chat.txt   # Python dependencies
├── .env                        # Environment variables (create this)
└── content/
    ├── data/                   # PDF documents (add your files here)
    │   ├── mental_health_guide.pdf
    │   └── therapy_manual.pdf
    └── chroma_db/             # Vector database (auto-created)
        └── [database files]
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API` | Your Groq API key for LLaMA access | Yes |

### Customizable Parameters

In `emp_chat.py`, you can modify:

```python
# Text chunking parameters
chunk_size = 500      # Characters per chunk
chunk_overlap = 50    # Overlap between chunks

# Conversation memory
recent_history = history[-3:]  # Number of exchanges to remember

# LLM parameters
temperature = 0       # Response consistency (0-1)
```

## 💬 How It Works

### 1. Knowledge Base Creation
- Loads PDF documents from `./content/data/`
- Splits documents into 500-character chunks with 50-character overlap
- Converts text to vector embeddings using sentence-transformers
- Stores vectors in ChromaDB for fast similarity search

### 2. Conversation Processing
- User sends message through Gradio interface
- System formats conversation history (last 3 exchanges)
- Vector database searches for relevant PDF content
- LLM receives: user question + relevant context + conversation history
- Generates empathetic response based on all inputs

### 3. Response Generation
- Uses Groq's LLaMA 3.3 70B model
- Prompt-engineered for mental health support
- References previous conversation when relevant
- Grounds advice in provided research documents

## 📚 Adding Knowledge

To expand the chatbot's knowledge base:

1. **Add PDF files** to `./content/data/`
2. **Delete the vector database**: `rm -rf ./content/chroma_db/`
3. **Restart the application**: `python emp_chat.py`

The chatbot will automatically rebuild the knowledge base with all PDF files.

### Recommended Document Types

- ✅ Mental health treatment guides
- ✅ Therapy technique manuals
- ✅ Coping strategy documents
- ✅ Crisis intervention protocols
- ✅ Evidence-based research papers
- ✅ Self-help resources

## 🎯 Usage Examples

### Basic Conversation
```
User: I'm feeling really anxious about work
Bot: I understand that work-related anxiety can feel overwhelming. Can you tell me more about what specific aspects of work are causing you the most stress?

User: My deadlines are impossible to meet
Bot: I can see how the anxiety you mentioned earlier about work is connected to these tight deadlines. Based on stress management techniques, here are some strategies that might help...
```

### Memory Demonstration
The chatbot remembers previous topics and builds on them:
- References earlier mentioned problems
- Follows up on suggested strategies
- Maintains consistent therapeutic approach

## ⚠️ Important Disclaimers

- **Not a replacement for professional therapy**
- **For crisis situations, seek immediate professional help**
- **This is a supportive tool, not medical advice**
- **Always consult licensed mental health professionals for serious concerns**

## 🔍 Troubleshooting

### Common Issues

**"No PDF documents found"**
- Check that PDF files are in `./content/data/`
- Ensure files have `.pdf` extension

**"Import errors"**
- Run: `pip install -r requirements_emp_chat.txt`
- Check Python version (3.8+ required)

**"API key error"**
- Verify `.env` file exists with correct `GROQ_API` key
- Check Groq API key is valid

**"Empty responses"**
- Ensure vector database was created successfully
- Check that PDF files contain readable text

### Performance Tips

- **First run**: Takes longer to create vector database
- **Subsequent runs**: Loads existing database (much faster)
- **Adding documents**: Delete `chroma_db` folder to rebuild
- **Memory usage**: Limit conversation history if needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add documentation for new features
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: Framework for building AI applications
- **Groq**: Fast LLaMA model inference
- **ChromaDB**: Vector database technology
- **Gradio**: User-friendly web interfaces
- **HuggingFace**: Sentence transformer models

## 📞 Support

For technical issues or questions:
- Create an issue in the repository
- Check the troubleshooting section above
- Review the inline code documentation

---

**Remember**: This chatbot provides general support. For urgent mental health concerns, please contact a licensed professional or crisis helpline immediately.