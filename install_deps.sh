#!/bin/bash

echo "🔧 Installing dependencies for Smart Document QA..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Installing..."
    sudo apt update
    sudo apt install -y python3 python3-pip
fi

echo "📦 Installing Python dependencies..."
pip3 install --user --break-system-packages sentence-transformers faiss-cpu numpy requests pytest

echo "✅ Verifying installation..."
python3 -c "
import sentence_transformers
import faiss
import numpy
import requests
import pytest
print('✅ All dependencies installed correctly')
"

echo "🎉 Installation completed!"