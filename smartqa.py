#!/usr/bin/env python3
"""
CLI for Smart Document QA
Usage: python3 smartqa.py --input <txt_file> [--ask "question"]
"""

import argparse
import sys
from pathlib import Path
from smartqa import TextChunker, Embedder, Retriever, LLMResponseGenerator
from smartqa.logger import QALogger


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


def ask_question(retriever, question: str, input_file: str = "unknown"):
    print(f"\n❓ Question: {question}")
    print("🔍 Searching for relevant information...")
    
    search_results = retriever.search(question, k=3)
    
    if not search_results:
        print("❌ No relevant information found to answer your question.")
        return
    
    print(f"   ✅ Found {len(search_results)} relevant chunks")
    
    print("🤖 Generating response...")
    llm = LLMResponseGenerator()
    response = llm.generate_response(question, search_results, input_file)
    
    print("\n" + "="*60)
    print("📝 ANSWER:")
    print("="*60)
    print(response["answer"])
    print(f"\n🔢 Tokens used: {response['tokens_used']}")
    print(f"⏱️  Response time: {response.get('latency_ms', 0)}ms")
    
    if response["citations"]:
        print(f"\n📚 Sources ({len(response['citations'])}):")
        for i, citation in enumerate(response["citations"], 1):
            print(f"   {i}. Chunk {citation['chunk_id']}: {citation['text']}")
            print(f"      Relevance: {citation['relevance_score']}")


def interactive_mode(retriever, input_file: str):
    print("\n" + "="*60)
    print("🎯 INTERACTIVE MODE")
    print("="*60)
    print("Type your questions (or 'exit' to quit, 'stats' for statistics):")
    
    while True:
        try:
            question = input("\n❓ Question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'stats':
                logger = QALogger()
                logger.print_stats()
                continue
            
            if not question:
                continue
            
            ask_question(retriever, question, input_file)
            
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
  python3 smartqa.py --stats
        """
    )
    
    parser.add_argument(
        "--input", 
        help="Text file to process (required unless --stats)"
    )
    
    parser.add_argument(
        "--ask",
        help="Specific question to answer (optional, enters interactive mode if not provided)"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show system statistics"
    )
    
    args = parser.parse_args()
    
    if args.stats:
        logger = QALogger()
        logger.print_stats()
        return
    
    if not args.input:
        print("❌ Error: --input is required (use --stats to see statistics)")
        sys.exit(1)
    
    if not Path(args.input).exists():
        print(f"❌ Error: File '{args.input}' not found")
        sys.exit(1)
    
    print("🚀 Smart Document QA")
    print("="*60)
    
    print(f"📖 Loading file: {args.input}")
    text = load_text_file(args.input)
    print(f"   ✅ File loaded ({len(text)} characters)")
    
    retriever = setup_qa_system(text)
    
    if args.ask:
        ask_question(retriever, args.ask, args.input)
    else:
        interactive_mode(retriever, args.input)
    
    print("\n" + "="*60)
    logger = QALogger()
    logger.print_stats()


if __name__ == "__main__":
    main()
