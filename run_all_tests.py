#!/usr/bin/env python3
"""
Script to run all project tests
"""

import sys
import traceback

def run_test(test_name, test_func):
    try:
        print(f"\n🧪 Running: {test_name}")
        print("=" * 50)
        
        original_print = print
        
        llm_responses = []
        
        def capture_print(*args, **kwargs):
            original_print(*args, **kwargs)
            if args and isinstance(args[0], str) and "Answer:" in args[0]:
                llm_responses.append(args[0])
        
        import builtins
        builtins.print = capture_print
        
        test_func()
        
        builtins.print = original_print
        
        print(f"✅ {test_name} - PASSED")
        
        if llm_responses:
            print("\n🤖 AI Responses:")
            print("-" * 30)
            for response in llm_responses:
                print(response)
            print("-" * 30)
        
        return True
    except Exception as e:
        print(f"❌ {test_name} - FAILED")
        print(f"   Error: {str(e)}")
        traceback.print_exc()
        return False

def main():
    print("🚀 Starting Smart Document QA project tests")
    print("=" * 60)
    
    try:
        from tests.test_chunker import test_chunker_basic
        from tests.test_embedder import test_embedder_with_chunker
        from tests.test_retriever import test_retriever_with_chunker_and_embedder
        from tests.test_llm import test_llm_with_relevant_answer, test_llm_without_relevant_answer, test_llm_with_empty_chunks
        from tests.test_cli import test_cli_help, test_cli_missing_input, test_cli_file_not_found, test_cli_basic_functionality
    except ImportError as e:
        print(f"❌ Error importing tests: {e}")
        return
    
    tests = [
        ("Basic Chunker", test_chunker_basic),
        ("Embedder with chunker", test_embedder_with_chunker),
        ("Complete Retriever", test_retriever_with_chunker_and_embedder),
        ("LLM with relevant answer", test_llm_with_relevant_answer),
        ("LLM without relevant answer", test_llm_without_relevant_answer),
        ("LLM with empty chunks", test_llm_with_empty_chunks),
        ("CLI help", test_cli_help),
        ("CLI missing input", test_cli_missing_input),
        ("CLI file not found", test_cli_file_not_found),
        ("CLI basic functionality", test_cli_basic_functionality),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Tests passed: {passed}")
    print(f"❌ Tests failed: {failed}")
    print(f"📈 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed successfully!")
        print("✅ Project is working correctly")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        print("🔧 Check the errors above for details")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 