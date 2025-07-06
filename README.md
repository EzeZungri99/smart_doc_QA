# Smart Document QA

A document question and answer system using embeddings and semantic search with Ollama.

## How It Works

This system processes documents through several steps:

1. **Text Chunking**: Divides large documents into smaller, manageable pieces
2. **Embedding Generation**: Converts text chunks into numerical vectors that capture meaning
3. **Semantic Search**: Finds the most relevant chunks for any question using vector similarity
4. **Answer Generation**: Uses Ollama (local LLM) to generate answers based on relevant context
5. **Logging**: Tracks tokens, latency, and usage statistics automatically

### Technical Flow

```
Document → Chunks → Embeddings → Search Index → Question → Relevant Chunks → LLM → Answer
                ↓
            QALogger → qa_history.jsonl
```

## Features

- **Intelligent Chunking**: Divides documents into meaningful segments
- **Semantic Embeddings**: Uses Sentence Transformers for vector representations
- **Semantic Search**: Finds relevant information using FAISS
- **Answer Generation**: Uses Ollama (local) to generate context-based answers
- **Citation Tracking**: Get source chunks with relevance scores
- **Comprehensive Logging**: Track tokens, latency, and usage statistics
- **Interactive Mode**: REPL interface for multiple questions
- **Performance Monitoring**: Real-time response times and token usage
- **No Virtual Environment**: Works directly with system Python
- **English Responses**: All answers are generated in English

## Installation

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-pip
```

### Install all Python dependencies with the provided script

```bash
./install_deps.sh
```

### Clone and Use

```bash
git clone https://github.com/ezezungri/smart_doc_QA.git
cd smart_doc_QA
```

## Usage

### Command Line Interface

The main way to use the system is through the CLI:

```bash
# Ask a specific question
python3 smartqa.py --input document.txt --ask "What is artificial intelligence?"

# Interactive mode
python3 smartqa.py --input document.txt

# View statistics
python3 smartqa.py --stats
```

### Interactive Mode

The interactive mode allows you to have a conversation with your document. It's perfect for exploring and asking multiple questions without reloading the system.

#### How it works:
1. **One-time setup**: Loads and processes the document once
2. **REPL interface**: Enter questions one by one
3. **Multiple questions**: Ask as many questions as you want
4. **Efficient**: No need to reload the document for each question

#### Commands:
- Type your question and press Enter
- Type `stats` to show system statistics
- Type `exit`, `quit`, or `q` to exit
- Press `Ctrl+C` to force exit

#### Example interactive session:
```bash
❓ Question: What is artificial intelligence?
📝 ANSWER: Artificial intelligence (AI) is a branch of computer science...
🔢 Tokens used: 45
⏱️  Response time: 1234ms

❓ Question: stats
📊 QA System Statistics:
========================================
Total interactions: 3
Total tokens used: 156
Total cost: $0.0000
Average latency: 987ms
Log file: qa_history.jsonl
========================================

❓ Question: exit
👋 Goodbye!
```

#### When to use each mode:
- **`--ask` mode**: For a quick, specific question
- **Interactive mode**: For exploring the document with multiple questions
- **`--stats` mode**: For viewing usage statistics

### Programmatic Usage

You can also use the system programmatically:

```python
from smartqa import TextChunker, Embedder, Retriever, LLMResponseGenerator

# 1. Prepare text
text = "Artificial intelligence is a branch of computer science..."

# 2. Create chunks
chunker = TextChunker()
chunks = chunker.create_chunks(text)

# 3. Setup embeddings and search
embedder = Embedder()
retriever = Retriever(embedder)
retriever.add_chunks(chunks)

# 4. Ask a question
query = "What is artificial intelligence?"
results = retriever.search(query, k=3)

# 5. Generate answer
llm = LLMResponseGenerator()
response = llm.generate_response(query, results)

print(response["answer"])
print(f"Tokens used: {response['tokens_used']}")
print(f"Response time: {response.get('latency_ms', 0)}ms")
```

## Testing

### Run All Tests

```bash
# Run the test script
python3 run_all_tests.py

# Or use pytest directly
python3 -m pytest tests/ -v
```

### Individual Tests

```bash
# Test chunking
python3 -c "from tests.test_chunker import test_chunker_basic; test_chunker_basic()"

# Test embeddings
python3 -c "from tests.test_embedder import test_embedder_with_chunker; test_embedder_with_chunker()"

# Test retrieval
python3 -c "from tests.test_retriever import test_retriever_with_chunker_and_embedder; test_retriever_with_chunker_and_embedder()"

# Test LLM
python3 -c "from tests.test_llm import test_llm_with_relevant_answer; test_llm_with_relevant_answer()"

# Test logger
python3 -c "from tests.test_logger import test_log_interaction; test_log_interaction()"
```

### Test Components

Each component can be tested independently:

- **Chunker**: Divides text into chunks
- **Embedder**: Converts text to vectors
- **Retriever**: Finds relevant chunks
- **LLM**: Generates answers
- **Logger**: Tracks interactions and statistics

## Ollama Setup

The system uses Ollama for answer generation. Make sure Ollama is installed and running:

```bash
# Install Ollama (if not installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Download a model (in another terminal)
ollama pull llama2
```

## Project Structure

```
smart_doc_QA/
├── smartqa/
│   ├── __init__.py          # Module exports
│   ├── chunker.py           # Text chunking
│   ├── embedder.py          # Embedding generation
│   ├── retriever.py         # Semantic search
│   ├── llm.py              # Answer generation
│   └── logger.py           # Logging system
├── tests/                   # Unit tests
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_retriever.py
│   ├── test_llm.py
│   ├── test_logger.py
│   └── test_cli.py
├── smartqa.py              # CLI interface
├── run_all_tests.py        # Test runner
├── example.txt             # Sample document
├── requirements.txt        # Dependencies
├── pyproject.toml         # Project configuration
├── qa_history.jsonl       # Interaction logs
└── README.md
```


## Troubleshooting

### Ollama Connection Issues
- Make sure Ollama is running: `ollama serve`
- Check if the model is downloaded: `ollama list`
- Verify the model name matches what's in the code

### Import Errors
- Ensure all dependencies are installed: `./install_deps.sh`
- Check that you're in the correct directory

### Performance Issues
- First runs may be slower as models are loaded
- Reduce the number of chunks retrieved by changing `k` in the search
- Use smaller documents for faster processing

**Examples:**

```bash
# Reduce chunks retrieved (default k=3, change to k=1 for faster processing)
python3 smartqa.py --input large_document.txt --ask "What is AI?"

# For very large documents, split them into smaller files
python3 smartqa.py --input chapter1.txt --ask "What is machine learning?"
python3 smartqa.py --input chapter2.txt --ask "What are neural networks?"

# Use interactive mode for multiple questions (avoids reloading models)
python3 smartqa.py --input document.txt
# Then ask multiple questions in the interactive session

# View usage statistics
python3 smartqa.py --stats
```
