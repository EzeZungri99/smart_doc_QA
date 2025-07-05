# Smart Document QA

A document question and answer system using embeddings and semantic search with Ollama.

## How It Works

This system processes documents through several steps:

1. **Text Chunking**: Divides large documents into smaller, manageable pieces
2. **Embedding Generation**: Converts text chunks into numerical vectors that capture meaning
3. **Semantic Search**: Finds the most relevant chunks for any question using vector similarity
4. **Answer Generation**: Uses Ollama (local LLM) to generate answers based on relevant context

### Technical Flow

```
Document → Chunks → Embeddings → Search Index → Question → Relevant Chunks → LLM → Answer
```

## Features

- **Intelligent Chunking**: Divides documents into meaningful segments
- **Semantic Embeddings**: Uses Sentence Transformers for vector representations
- **Semantic Search**: Finds relevant information using FAISS
- **Answer Generation**: Uses Ollama (local) to generate context-based answers
- **No Virtual Environment**: Works directly with system Python
- **English Responses**: All answers are generated in English

## Installation

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-pip

# Install Python dependencies (globally)
pip3 install sentence-transformers faiss-cpu numpy requests pytest
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
```

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
```

### Test Components

Each component can be tested independently:

- **Chunker**: Divides text into chunks
- **Embedder**: Converts text to vectors
- **Retriever**: Finds relevant chunks
- **LLM**: Generates answers

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
│   └── llm.py              # Answer generation
├── tests/                   # Unit tests
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_retriever.py
│   └── test_llm.py
├── smartqa.py              # CLI interface
├── run_all_tests.py        # Test runner
├── example.txt             # Sample document
├── requirements.txt        # Dependencies
├── pyproject.toml         # Project configuration
└── README.md
```

## Example Document

Create a text file (e.g., `example.txt`) with content like:

```
Artificial intelligence (AI) is a branch of computer science that seeks to create systems capable of performing tasks that normally require human intelligence.

Machine learning is a subcategory of AI that allows computers to learn and improve automatically without being explicitly programmed. Machine learning algorithms build a mathematical model based on sample data to make predictions or decisions.

Chatbots are AI programs designed to simulate human conversations. They use natural language processing to understand and respond to user queries. Modern chatbots can handle complex queries and provide useful responses.
```

Then run:
```bash
python3 smartqa.py --input example.txt --ask "What is artificial intelligence?"
```

## Troubleshooting

### Ollama Connection Issues
- Make sure Ollama is running: `ollama serve`
- Check if the model is downloaded: `ollama list`
- Verify the model name matches what's in the code

### Import Errors
- Ensure all dependencies are installed: `pip3 install sentence-transformers faiss-cpu numpy requests pytest`
- Check that you're in the correct directory

### Performance Issues
- First runs may be slower as models are loaded
- Reduce the number of chunks retrieved by changing `k` in the search
- Use smaller documents for faster processing
