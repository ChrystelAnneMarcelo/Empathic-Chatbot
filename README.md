# Empathic Mental Health Chatbot

A compassionate AI chatbot that provides mental health support using conversation memory and research-based knowledge from PDF documents.


### Technology Stack

- **LLM**: Groq's LLaMA 3.3 70B model
- **Vector Database**: ChromaDB with HuggingFace embeddings
- **PDF Processing**: LangChain community loaders
- **Web Interface**: Gradio
- **Architecture**: RAG (Retrieval-Augmented Generation) with conversation memory

## Quick Start

### Prerequisites

- Python 3.8+
- Groq API key (free at [Groq Console](https://console.groq.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Empathic-Chatbot
   ```

2. **Install dependencies in your virtual environment**
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

## Configuration

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

**Remember**: This chatbot provides general support. For urgent mental health concerns, please contact a licensed professional or crisis helpline immediately.
