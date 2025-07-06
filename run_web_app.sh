#!/bin/bash

echo "🚀 Starting Smart Document QA Web App..."
echo "=========================================="

if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing Streamlit..."
    pip3 install --user streamlit
    echo "✅ Streamlit installed!"
fi

if ! curl -s http://localhost:11434/api/tags >/dev/null; then
    echo "⚠️  Warning: Ollama is not running"
    echo "   Please start Ollama with: ollama serve"
    echo "   And download a model with: ollama pull llama2"
    echo ""
fi

echo "🌐 Starting web server..."
echo "   URL: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

~/.local/bin/streamlit run web_app.py --server.port 8501 --server.address localhost 