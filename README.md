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

### 1. Install Poetry
```bash
pip install poetry
```

### 2. Clone the project
```bash
git clone https://github.com/EzeZungri99/smart_doc_QA.git
cd smart_doc_QA
```

### 3. Install dependencies
```bash
poetry install
```

### 4. Setup Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull llama2
```

## Usage

### Command Line Interface (CLI)

**Single question:**
```bash
# Single question
poetry run python smartqa.py --input example.txt --ask "What is AI?"

**Interactive mode (multiple questions):**
```bash
poetry run python smartqa.py --input example.txt
```

### Web Interface
```bash
./run_web_app.sh
# Open http://localhost:8501
```
Then open your browser to `http://localhost:8501`

- Upload your document
- Ask questions in the chat interface
- View response statistics and citations

## Testing

### Run All Tests
```bash
# All tests
poetry run pytest
```

### Individual Component Tests
```bash
# Test text chunking
poetry run pytest tests/test_chunker.py -v

# Test embedding generation
poetry run pytest tests/test_embedder.py -v

# Test semantic search
poetry run pytest tests/test_retriever.py -v

# Test LLM responses
poetry run pytest tests/test_llm.py -v

# Test logging system
poetry run pytest tests/test_logger.py -v

# CLI tests 
poetry run pytest tests/test_cli.py -v
```

### Run tests with more detail
```bash
poetry run pytest -v --tb=long
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
├── run_web_app.sh         # Web app launcher
└── pyproject.toml         # Poetry configuration
```

## Environment Variables

- `EMBEDDING_MODEL`: Hugging Face model for embeddings (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `OLLAMA_MODEL`: Ollama model name (default: `llama2`)
- `FAISS_INDEX_TYPE`: FAISS index type (default: `IndexFlatIP`)

Example:
```bash
export EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
export OLLAMA_MODEL="llama2:7b"
poetry run python smartqa.py --input document.txt --ask "Question"
```

## Development

```bash
# Add dependency
poetry add package-name

# Add dev dependency
poetry add --dev package-name

# Update dependencies
poetry update

# Show installed packages
poetry show
```

## Troubleshooting

- **Ollama not running**: `ollama serve`
- **Model missing**: `ollama pull llama2`
- **Dependencies**: `poetry install`
- **Permissions**: `chmod +x run_web_app.sh`

## Design Decisions

### Offline AI Usage
- **Decision**: Use local Ollama instead of online APIs
- **Rationale**: Simpler configuration, higher speed, no token limits

### 45-Second Timeout
- **Decision**: Set 45-second timeout for LLM responses
- **Rationale**: Proven performance for complex questions, prevents indefinite waits

### Standardized "No Information" Response
- **Decision**: Use fixed response when no relevant information is available
- **Rationale**: Prevents hallucination, maintains accuracy

### Complete JSONL Logging
- **Decision**: Implement comprehensive logging in JSONL format
- **Rationale**: History without database, user query tracking, debugging capability

### Paragraph-Based Chunking
- **Decision**: Split documents by paragraphs rather than character count
- **Rationale**: Clarity and depth, coherent responses, semantic integrity

### Streamlit for Web UI
- **Decision**: Use Streamlit instead of traditional web frameworks
- **Rationale**: Simple UI design, immediate application, fast loading

## Requirements

- Python 3.11+
- Ollama running
- Internet (initial setup only)

