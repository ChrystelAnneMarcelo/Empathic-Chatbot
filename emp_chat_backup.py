"""
EMPATHIC MENTAL HEALTH CHATBOT WITH CONVERSATION MEMORY
=====================================================

This chatbot uses RAG (Retrieval-Augmented Generation) architecture to provide
empathetic mental health support based on PDF documents and conversation history.

Architecture Overview:
1. PDF Documents → Text Chunks → Vector Embeddings → ChromaDB (Knowledge Base)
2. Use    # Information modal content
    help_text = """
    ## How to Use This Chatbot

    **For Emotional Support:**
    - Share your feelings openly - I'll listen and validate
    - I remember our conversation to provide better support

    **For Advice:**
    - Ask specific questions like "How do I deal with anxiety?"
    - Request tips, strategies, or suggestions

    **Emergency Resources:**
    - **Crisis Text Line**: Text HOME to 741741
    - **National Suicide Prevention Lifeline**: 988
    - **Emergency**: 911

    **Remember**: This is supportive AI, not professional therapy.
    """arch → Relevant Context + Conversation History
3. Context + Query + History → LLM → Empathetic Response

Key Features:
- Remembers conversation context for empathetic responses
- Grounds responses in mental health research papers
- Uses Groq's LLaMA 3.3 70B model for natural conversation
- Gradio web interface for easy interaction
"""

from dotenv import load_dotenv 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # Convert text to vectors
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader  # Load PDF files
from langchain_community.vectorstores import Chroma  # Vector database for similarity search
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split documents into chunks
from langchain_groq import ChatGroq  # Groq API client for LLaMA model
import os  # Operating system interface
import gradio as gr  # Web interface framework

# Load environment variables (API keys) from .env file
load_dotenv()

def initialize_llm():
	"""
	Initialize the Large Language Model (LLM) using Groq's API.
	
	This function creates a connection to Groq's LLaMA 3.3 70B model which will:
	- Generate empathetic responses based on context
	- Maintain conversation flow
	- Provide mental health guidance
	
	Returns:
		ChatGroq: Configured LLM instance ready for generating responses
	
	Note: Requires GROQ_API environment variable to be set
	"""
	llm = ChatGroq(
		temperature=0,
		groq_api_key=os.getenv("GROQ_API"), 
		model_name="llama-3.3-70b-versatile"
	)
	return llm

def vector_db():
	"""
	Create or load ChromaDB vector database from PDF documents.
	
	This is the KNOWLEDGE BASE of your chatbot. It:
	1. Loads all PDF files from ./content/data/ directory
	2. Splits documents into 500-character chunks with 50-char overlap
	3. Converts text chunks into numerical vectors (embeddings)
	4. Stores vectors in ChromaDB for fast similarity search
	
	The Process:
	PDF → Text → Chunks → Embeddings → Vector Database
	
	Returns:
		Chroma: Vector database instance for document retrieval
		None: If no PDFs found or error occurs
	"""
	try:
		# Load all PDF files from the data directory
		loader = DirectoryLoader("./content/data/", glob='*.pdf', loader_cls=PyPDFLoader)
		documents = loader.load()  # Extract text content from PDFs
		
		# Check if any documents were loaded
		if not documents:
			print("WARNING: No PDF documents found in ./content/data/")
			return None
			
		print(f"Successfully loaded {len(documents)} PDF documents")
		
		# Split documents into smaller chunks for better retrieval
		# chunk_size=500: Each piece is ~500 characters (roughly 1-2 paragraphs)
		# chunk_overlap=50: Overlap prevents context loss between chunks
		text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
		texts = text_splitter.split_documents(documents)
		print(f"Created {len(texts)} text chunks")
		
		# Create embeddings (convert text to numerical vectors for similarity search)
		# This model converts text meaning into 384-dimensional vectors
		embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
		
		# Create vector database and store all text chunks with their embeddings
		vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./content/chroma_db')
		vector_db.persist()  # Save to disk so we don't rebuild every time

		print("ChromaDB created and data saved")
		return vector_db
		
	except Exception as e:
		print(f"ERROR: Error creating vector database: {str(e)}")
		print("Make sure you have PDF files in ./content/data/ and all required packages installed")
		return None

def setup_qa_chain(vector_db, llm):
	"""
	Create a custom Question-Answering chain that maintains conversation history.
	
	This is the BRAIN of your chatbot that:
	1. Takes user questions and conversation history
	2. Searches vector database for relevant mental health information
	3. Combines context + history + question into a prompt
	4. Generates empathetic responses using the LLM
	
	The QA Process:
	User Question → Vector Search → Context + History → LLM → Response
	
	Args:
		vector_db: ChromaDB instance for document retrieval
		llm: Large Language Model for response generation
		
	Returns:
		HistoryAwareQAChain: Custom chain that handles conversation memory
	"""
	retriever = vector_db.as_retriever()  # Create search interface for vector database
	
	# Create a custom chain that can handle conversation history
	class HistoryAwareQAChain:
		"""
		Custom QA chain that remembers conversation context.
		
		Unlike standard QA chains, this one:
		- Maintains conversation memory
		- Provides contextual responses
		- References previous exchanges
		- Builds therapeutic rapport over time
		"""
		
		def __init__(self, llm, retriever):
			self.llm = llm  # Language model for generation
			self.retriever = retriever  # Vector database retriever
			
			# Create prompt template for history-aware responses
			# This template structures how the AI sees the conversation
			self.prompt_template = """You are a warm, empathetic mental health supporter who balances emotional validation with helpful guidance.

Previous conversation: {conversation_history}

Mental health context: {context}

Current message: {query}

RESPONSE GUIDELINES:
- For emotional expressions WITHOUT advice requests: Focus on validation ("I hear you", "That sounds really difficult") and ask gentle follow-up questions
- For advice requests (even with emotion): Provide brief validation THEN offer practical, helpful guidance from the mental health resources
- For pure advice requests: Give warm, practical guidance while maintaining supportive tone
- Keep emotional validation responses shorter (2-3 sentences), advice responses can be more detailed
- Reference previous conversation naturally, like a caring friend would
- Be human, warm, and present - not clinical or robotic
- When giving advice, make it actionable and easy to understand

Respond appropriately:"""
		
		def invoke(self, inputs):
			"""
			Process user input and generate contextual response.
			
			Steps:
			1. Extract user query and conversation history
			2. Detect emotional tone for appropriate response style
			3. Search vector database for relevant documents
			4. Combine all information into structured prompt
			5. Generate empathetic response using LLM
			
			Args:
				inputs: Dictionary containing 'query' and 'conversation_history'
				
			Returns:
				Dictionary with 'result' key containing bot response
			"""
			query = inputs["query"]  # Current user message
			conversation_history = inputs.get("conversation_history", "No previous conversation.")
			
			# Detect emotional tone and advice-seeking to adjust response style
			sad_keywords = ['sad', 'depressed', 'hopeless', 'crying', 'hurt', 'pain', 'lost', 'empty', 'worthless', 'alone']
			vulnerable_keywords = ['scared', 'afraid', 'worried', 'anxious', 'confused', 'overwhelmed', 'struggling']
			advice_keywords = ['what should i do', 'how do i', 'help me', 'advice', 'suggestions', 'what can i do', 'how can i', 'tips', 'strategies', 'recommend']
			
			query_lower = query.lower()
			is_emotional = any(word in query_lower for word in sad_keywords + vulnerable_keywords)
			is_seeking_advice = any(phrase in query_lower for phrase in advice_keywords)
			
			# Search vector database for relevant mental health information
			# This finds PDF chunks most similar to the user's question
			relevant_docs = self.retriever.get_relevant_documents(query)
			context = "\\n".join([doc.page_content for doc in relevant_docs])
			
			# Adjust prompt based on emotional tone AND advice-seeking
			if is_emotional and not is_seeking_advice:
				# Pure emotional expression - prioritize validation
				emotion_guidance = "\\n\\nEMOTIONAL RESPONSE NEEDED: The user is expressing vulnerability but not explicitly asking for advice. Start with validation and emotional support, then gently ask if they'd like to talk more about it or if there's anything specific that might help."
				context = context + emotion_guidance
			elif is_emotional and is_seeking_advice:
				# Emotional + advice request - validate first, then provide guidance
				emotion_guidance = "\\n\\nEMOTIONAL + ADVICE RESPONSE: The user is vulnerable AND asking for help. Start with brief validation ('I hear how difficult this is for you'), then provide gentle, practical guidance based on the mental health resources."
				context = context + emotion_guidance
			elif is_seeking_advice:
				# Pure advice request - provide helpful guidance
				advice_guidance = "\\n\\nADVICE RESPONSE: The user is specifically asking for guidance. Provide helpful, practical advice based on the mental health resources while maintaining a warm, supportive tone."
				context = context + advice_guidance
			
			# Create complete prompt by filling in the template
			full_prompt = self.prompt_template.format(
				conversation_history=conversation_history,  # What we've talked about
				context=context,  # Relevant info from mental health PDFs + emotional guidance
				query=query  # Current user question
			)
			
			# Generate response using the LLM
			response = self.llm.invoke(full_prompt)
			return {"result": response.content}
	
	return HistoryAwareQAChain(llm, retriever)

# ============================================================================
# MAIN EXECUTION: Initialize the chatbot system
# ============================================================================

print("Initializing Chatbot..........")
llm = initialize_llm()  # Set up connection to Groq's LLaMA model

db_path = "./content/chroma_db"  # Path where vector database is stored

# Check if vector database already exists (saves time on subsequent runs)
if not os.path.exists(db_path):
	# Database doesn't exist, create new one from PDF files
	vector_db = vector_db()
else:
	# Database exists, load it directly (much faster than rebuilding)
	print("Loading existing vector database...")
	embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
	vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

# Create the QA chain that combines retrieval + generation + history
qa_chain = setup_qa_chain(vector_db, llm)

def format_conversation_history(history):
	"""
	Convert Gradio's conversation history into readable text format.
	
	This function takes the chat history from Gradio and formats it so the AI
	can understand previous conversations. It helps maintain context across
	multiple exchanges for more empathetic responses.
	
	Args:
		history: List of conversation exchanges from Gradio interface
		
	Returns:
		str: Formatted conversation history as readable text
		
	Example:
		Input: [{"role": "user", "content": "I'm anxious"}, {"role": "assistant", "content": "I understand..."}]
		Output: "User: I'm anxious\nChatbot: I understand..."
	"""
	if not history:
		return "No previous conversation."
	
	# Get last 3 exchanges for context (prevents prompt from becoming too long)
	recent_history = history[-3:] if len(history) > 3 else history
	formatted_history = []
	
	# Process each message in the conversation history
	for message in recent_history:
		if isinstance(message, dict):
			# New message format used by Gradio (role-content structure)
			role = message.get('role', 'unknown')
			content = message.get('content', '')
			if role == 'user':
				formatted_history.append(f"User: {content}")
			elif role == 'assistant':
				formatted_history.append(f"Chatbot: {content}")
		else:
			# Handle older tuple format (user_message, bot_response)
			if len(message) >= 2:
				user_msg, bot_msg = message[0], message[1]
				formatted_history.append(f"User: {user_msg}")
				formatted_history.append(f"Chatbot: {bot_msg}")
	
	return "\n".join(formatted_history)

def response(input, history):
	"""
	Main response function called by Gradio when user sends a message.
	
	This is the ENTRY POINT for user interactions. It:
	1. Validates user input
	2. Formats conversation history for context
	3. Calls the QA chain to generate response
	4. Returns empathetic response based on PDFs + conversation memory
	
	Args:
		input (str): Current user message
		history (list): Previous conversation exchanges from Gradio
		
	Returns:
		str: Bot's empathetic response
		
	Flow:
		User Message → Format History → QA Chain → PDF Context + History → LLM → Response
	"""
	# Basic input validation
	if not input.strip():
		return "Provide valid input please."
	
	# Convert Gradio's history format into readable conversation context
	conversation_context = format_conversation_history(history)
	
	# Create enhanced query with both current question and conversation history
	enhanced_query = {
		"query": input,  # What the user just asked
		"conversation_history": conversation_context  # What we've talked about before
	}
	
	# Use the QA chain to generate context-aware response
	# This searches PDFs, combines with history, and generates empathetic response
	response = qa_chain.invoke(enhanced_query)
	return response["result"]

# ============================================================================
# WEB INTERFACE: Create Gradio app for user interaction
# ============================================================================

def exit_application():
    """Function to handle application exit"""
    import sys
    import os
    print("\nThank you for using the Empathic Mental Health Chatbot!")
    print("Take care of yourself!")
    os._exit(0)

# Create the web interface using Gradio Blocks
with gr.Blocks(theme = 'earneleh/paris') as app:
    # Header and description
    gr.Markdown("# Empathetic Mental Health Support")
    gr.Markdown("I'm here to listen and support you. I prioritize understanding your feelings before offering any guidance.")
    gr.Markdown("**Note**: I focus on emotional validation first, especially when you're feeling vulnerable.")

    # Main chat interface
    # fn=response: Function to call when user sends message
    # type="messages": Use modern message format for better conversation flow
    chatbot = gr.ChatInterface(fn=response, title="Empathic Listener", type="messages")

    # Simple exit button
    exit_btn = gr.Button("Exit", variant="stop")
    
    # Exit button functionality
    exit_btn.click(
        fn=exit_application,
        inputs=None,
        outputs=None
    )

    gr.Markdown("**Remember**: For crisis situations or urgent concerns, please reach out to a mental health professional or crisis helpline immediately.")

# Create the web interface using Gradio Blocks
with gr.Blocks(theme = 'earneleh/paris') as app:
    # Header and description
    gr.Markdown("# Empathetic Mental Health Support")
    gr.Markdown("I'm here to listen and support you. I prioritize understanding your feelings before offering any guidance.")
    gr.Markdown("**Note**: I focus on emotional validation first, especially when you're feeling vulnerable.")

    # Main chat interface
    # fn=response: Function to call when user sends message
    # type="messages": Use modern message format for better conversation flow
    chatbot = gr.ChatInterface(fn=response, title="Empathic Listener", type="messages")

    # Control buttons section
    with gr.Row():
        with gr.Column(scale=1):
            exit_btn = gr.Button("Exit Safely", variant="stop", size="sm")
        with gr.Column(scale=1):
            info_btn = gr.Button("Help", variant="secondary", size="sm")
        with gr.Column(scale=2):
            status_text = gr.Markdown("**Status**: Ready to listen and support you")

    # Information modal content
    help_text = """
    ## � How to Use This Chatbot

    **For Emotional Support:**
    - Share your feelings openly - I'll listen and validate
    - I remember our conversation to provide better support

    **For Advice:**
    - Ask specific questions like "How do I deal with anxiety?"
    - Request tips, strategies, or suggestions

    **Emergency Resources:**
    - 🆘 **Crisis Text Line**: Text HOME to 741741
    - 📞 **National Suicide Prevention Lifeline**: 988
    - 🏥 **Emergency**: 911

    **Remember**: This is supportive AI, not professional therapy.
    """

    # Create modal for help information
    with gr.Column(visible=False) as help_modal:
        gr.Markdown(help_text)
        close_help_btn = gr.Button("Close", variant="secondary")

    # Button functionality
    def show_help():
        return gr.update(visible=True)
    
    def hide_help():
        return gr.update(visible=False)

    # Connect button events
    exit_btn.click(
        fn=exit_application,
        inputs=None,
        outputs=None
    )
    
    info_btn.click(
        fn=show_help,
        outputs=help_modal
    )
    
    close_help_btn.click(
        fn=hide_help,
        outputs=help_modal
    )

    gr.Markdown("**Remember**: For crisis situations or urgent concerns, please reach out to a mental health professional or crisis helpline immediately.")

app.launch(inbrowser=True) 
