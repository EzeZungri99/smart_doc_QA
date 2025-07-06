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
git clone https://github.com/your-username/smart_doc_QA.git
cd smart_doc_QA
```

### 3. Install dependencies
```bash
poetry install
```

### 4. Setup Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Download the model (in another terminal)
ollama pull llama2
```

## Usage

### Command Line Interface (CLI)

**Single question:**
```bash
poetry run python smartqa.py --input document.txt --ask "What is artificial intelligence?"
```

**Interactive mode (multiple questions):**
```bash
poetry run python smartqa.py --input document.txt
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
├── example.txt            # Sample document
└── pyproject.toml         # Poetry configuration
```

## Useful Commands

### Development
```bash
# Install dependencies
poetry install

# Add new dependency
poetry add package-name

# Add development dependency
poetry add --dev package-name

# View installed dependencies
poetry show

# Update dependencies
poetry update
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=smartqa

# Run specific tests
poetry run pytest tests/test_chunker.py::test_chunker_basic
```

### Execution
```bash
# CLI with specific document
poetry run python smartqa.py --input my_document.txt --ask "My question"

# Web app
poetry run streamlit run web_app.py

# Or use the script
./run_web_app.sh
```

## Troubleshooting

### Ollama not running
```bash
ollama serve
```

### Model not found
```bash
ollama pull llama2
```

### Missing dependencies
```bash
poetry install
```

### Permission denied on scripts
```bash
chmod +x run_web_app.sh
```

### Disk space error
- Clean temporary files
- Remove old virtual environments
- Free space in `/tmp`

## Features

- **Intelligent document processing**: Divides documents into meaningful chunks
- **Semantic search**: Finds relevant information using AI embeddings
- **Local AI**: Uses Ollama for privacy and speed
- **Multiple Interfaces**: CLI and web interface
- **Comprehensive Logging**: Track all interactions and performance
- **Robust Testing**: Individual and comprehensive test suites
- **Easy Setup**: Automated dependency installation

## Requirements

- Python 3.11 or higher
- Ollama installed and running
- Internet connection (only for initial installation)

