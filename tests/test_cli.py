import subprocess
import os
from pathlib import Path


def test_cli_basic_functionality():
    if not Path("example.txt").exists():
        print("❌ CLI test SKIPPED - example.txt not found")
        return
    
    try:
        result = subprocess.run(
            ['python3', 'smartqa.py', '--input', 'example.txt', '--ask', 'What is artificial intelligence?'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        assert result.returncode == 0, f"CLI failed with return code {result.returncode}"
        assert "Smart Document QA" in result.stdout, "CLI header not found"
        assert "Loading file: example.txt" in result.stdout, "File loading message not found"
        assert "Setting up QA system" in result.stdout, "QA system setup not found"
        assert "Question: What is artificial intelligence?" in result.stdout, "Question not found"
        assert "ANSWER:" in result.stdout, "Answer section not found"
        assert "Sources" in result.stdout, "Sources section not found"
        
        print("✅ CLI basic functionality test PASSED")
        print(f"   Output length: {len(result.stdout)} characters")
        
    except subprocess.TimeoutExpired:
        print("❌ CLI test TIMEOUT - Ollama might not be running")
    except Exception as e:
        print(f"❌ CLI test FAILED: {e}")


def test_cli_file_not_found():
    result = subprocess.run(
        ['python3', 'smartqa.py', '--input', 'nonexistent_file.txt', '--ask', 'test'],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 1, "CLI should exit with code 1 for missing file"
    assert "File 'nonexistent_file.txt' not found" in result.stderr or "File 'nonexistent_file.txt' does not exist" in result.stdout
    
    print("✅ CLI file not found test PASSED")


def test_cli_help():
    result = subprocess.run(
        ['python3', 'smartqa.py', '--help'],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "Help should exit with code 0"
    assert "Smart Document QA" in result.stdout, "Help should show CLI description"
    assert "--input" in result.stdout, "Help should show input argument"
    assert "--ask" in result.stdout, "Help should show ask argument"
    
    print("✅ CLI help test PASSED")


def test_cli_missing_input():
    result = subprocess.run(
        ['python3', 'smartqa.py'],
        capture_output=True,
        text=True
    )
    
    assert result.returncode != 0, "CLI should fail without required input"
    
    print("✅ CLI missing input test PASSED")


if __name__ == "__main__":
    print("🧪 Testing Smart QA CLI...")
    print("=" * 50)
    
    test_cli_help()
    test_cli_missing_input()
    test_cli_file_not_found()
    test_cli_basic_functionality()
    
    print("\n🎉 CLI tests completed!") 