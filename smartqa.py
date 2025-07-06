#!/usr/bin/env python3
"""
CLI for Smart Document QA
Usage: python3 smartqa.py --input <txt_file> [--ask "question"]
"""

import argparse
import sys
from pathlib import Path
from smartqa import TextChunker, Embedder, Retriever, LLMResponseGenerator


def load_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)


def setup_qa_system(text: str):
    print("🔧 Setting up QA system...")
    
    print("📄 Dividing text into chunks...")
    chunker = TextChunker()
    chunks = chunker.create_chunks(text)
    print(f"   ✅ Created {len(chunks)} chunks")
    
    print("🔢 Setting up embeddings and search...")
    embedder = Embedder()
    retriever = Retriever(embedder)
    retriever.add_chunks(chunks)
    print("   ✅ System configured")
    
    return retriever


def ask_question(retriever, question: str):
    print(f"\n❓ Question: {question}")
    print("🔍 Searching for relevant information...")
    
    search_results = retriever.search(question, k=3)
    
    if not search_results:
        print("❌ No relevant information found to answer your question.")
        return
    
    print(f"   ✅ Found {len(search_results)} relevant chunks")
    
    print("🤖 Generating response...")
    llm = LLMResponseGenerator()
    response = llm.generate_response(question, search_results)
    
    print("\n" + "="*60)
    print("📝 ANSWER:")
    print("="*60)
    print(response["answer"])
    print(f"\n🔢 Tokens used: {response['tokens_used']}")
    
    if response["citations"]:
        print(f"\n📚 Sources ({len(response['citations'])}):")
        for i, citation in enumerate(response["citations"], 1):
            print(f"   {i}. Chunk {citation['chunk_id']}: {citation['text']}")
            print(f"      Relevance: {citation['relevance_score']}")


def interactive_mode(retriever):
    print("\n" + "="*60)
    print("🎯 INTERACTIVE MODE")
    print("="*60)
    print("Type your questions (or 'exit' to quit):")
    
    while True:
        try:
            question = input("\n❓ Question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            ask_question(retriever, question)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Smart Document QA - Document question and answer system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python3 smartqa.py --input document.txt --ask "What is AI?"
  python3 smartqa.py --input document.txt
        """
    )
    
    parser.add_argument(
        "--input", 
        required=True,
        help="Text file to process (required)"
    )
    
    parser.add_argument(
        "--ask",
        help="Specific question to answer (optional, enters interactive mode if not provided)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ Error: File '{args.input}' does not exist")
        sys.exit(1)
    
    print("🚀 Smart Document QA")
    print("="*60)
    
    print(f"📖 Loading file: {args.input}")
    text = load_text_file(args.input)
    print(f"   ✅ File loaded ({len(text)} characters)")
    
    retriever = setup_qa_system(text)
    
    if args.ask:
        ask_question(retriever, args.ask)
    else:
        interactive_mode(retriever)


if __name__ == "__main__":
    main()
