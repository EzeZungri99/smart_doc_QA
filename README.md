# Smart Document QA

A document question and answer system using embeddings and semantic search with Ollama.

## What It Does

This system allows you to ask questions about any text document and get AI-powered answers based on the document's content. It works by:

1. **Breaking down** your document into meaningful chunks
2. **Converting** text into numerical vectors that understand meaning
3. **Finding** the most relevant parts of your document for each question
4. **Generating** accurate answers using a local AI model
5. **Tracking** all interactions for analysis

## Quick Start

### 1. Install Dependencies

```bash
# Install system dependencies
sudo apt update && sudo apt install python3-pip

# Install Python dependencies
./install_deps.sh
```

### 2. Setup Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Download the AI model (in another terminal)
ollama pull llama2
```

### 3. Use the System

```bash
# Ask a question about a document
python smartqa.py example.txt "What is artificial intelligence?"

# Interactive mode for multiple questions
python smartqa.py example.txt

# Web interface
./run_web_app.sh
# Then open http://localhost:8501
```

## Installation Details

### Prerequisites
- Ubuntu/Debian Linux
- Python 3.8+
- Internet connection (for initial setup)

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ezezungri/smart_doc_QA.git
   cd smart_doc_QA
   ```

2. **Install Python dependencies**
   ```bash
   ./install_deps.sh
   ```

3. **Verify installation**
   ```bash
   python3 -c "import sentence_transformers, faiss, numpy, requests; print('✅ All dependencies installed')"
   ```

## Usage

### Command Line Interface

**Single Question:**
```bash
python smartqa.py document.txt "Your question here?"
```

**Interactive Mode:**
```bash
python smartqa.py document.txt
# Then type questions one by one
# Type 'exit' to quit
```

### Web Interface

```bash
./run_web_app.sh
```
Then open your browser to `http://localhost:8501`

- Upload your document
- Ask questions in the chat interface
- View response statistics and citations

## Testing

### Run All Tests
```bash
python3 run_all_tests.py
```

### Individual Component Tests
```bash
# Test text chunking
python3 -c "from tests.test_chunker import test_chunker_basic; test_chunker_basic()"

# Test embedding generation
python3 -c "from tests.test_embedder import test_embedder_with_chunker; test_embedder_with_chunker()"

# Test semantic search
python3 -c "from tests.test_retriever import test_retriever_with_chunker_and_embedder; test_retriever_with_chunker_and_embedder()"

# Test LLM responses
python3 -c "from tests.test_llm import test_llm_basic; test_llm_basic()"

# Test logging system
python3 -c "from tests.test_logger import test_log_interaction; test_log_interaction()"
```

### Test Specific Scenarios
```bash
# Test handling of questions without relevant information
python3 -c "from tests.test_llm import test_llm_with_completely_unrelated_question; test_llm_with_completely_unrelated_question()"

# Test personal questions
python3 -c "from tests.test_llm import test_llm_with_personal_question; test_llm_with_personal_question()"

# Test empty document handling
python3 -c "from tests.test_llm import test_llm_with_empty_chunks; test_llm_with_empty_chunks()"
```

## Project Structure

```
smart_doc_QA/
├── smartqa/                 # Core system modules
│   ├── chunker.py          # Divides documents into chunks
│   ├── embedder.py         # Converts text to vectors
│   ├── retriever.py        # Finds relevant information
│   ├── llm.py             # Generates AI responses
│   └── logger.py          # Tracks interactions
├── tests/                  # Test files
├── web_app.py             # Streamlit web interface
├── smartqa.py             # Command line interface
├── install_deps.sh        # Dependency installer
├── run_web_app.sh         # Web app launcher
├── example.txt            # Sample document
└── requirements.txt       # Python dependencies
```

## Technical Decisions

### **Offline AI Usage**
- **Decision**: Use local Ollama instead of online APIs
- **Rationale**: Simpler configuration, higher speed, no token limits

### **45-Second Timeout**
- **Decision**: Set 45-second timeout for LLM responses
- **Rationale**: Proven performance for complex questions, prevents indefinite waits

### **Standardized "No Information" Response**
- **Decision**: Use fixed response when no relevant information is available
- **Rationale**: Prevents hallucination, maintains accuracy

### **Complete JSONL Logging**
- **Decision**: Implement comprehensive logging in JSONL format
- **Rationale**: History without database, user query tracking, debugging capability

### **Paragraph-Based Chunking**
- **Decision**: Split documents by paragraphs rather than character count
- **Rationale**: Clarity and depth, coherent responses, semantic integrity

### **Streamlit for Web UI**
- **Decision**: Use Streamlit instead of traditional web frameworks
- **Rationale**: Simple UI design, immediate application, fast loading

## Troubleshooting

### Common Issues

**Ollama not running:**
```bash
ollama serve
```

**Model not found:**
```bash
ollama pull llama2
```

**Dependencies missing:**
```bash
./install_deps.sh
```

**Permission denied on scripts:**
```bash
chmod +x install_deps.sh run_web_app.sh
```

### Performance Tips

- **First run**: May be slower as models load
- **Large documents**: Consider splitting into smaller files
- **Multiple questions**: Use interactive mode to avoid reloading models
- **Web interface**: Faster for multiple questions on same document

### Logs and Debugging

- **Interaction logs**: `qa_history.jsonl`
- **View statistics**: Check the logs file for usage data
- **Test individual components**: Use the individual test commands above

## Features

- **Intelligent Document Processing**: Breaks documents into meaningful chunks
- **Semantic Search**: Finds relevant information using AI embeddings
- **Local AI**: Uses Ollama for privacy and speed
- **Multiple Interfaces**: CLI and web interface
- **Comprehensive Logging**: Track all interactions and performance
- **Robust Testing**: Individual and comprehensive test suites
- **Easy Setup**: Automated dependency installation

