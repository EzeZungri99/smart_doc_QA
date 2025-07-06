import pytest
import subprocess
import tempfile
import os


def test_cli_help():
    result = subprocess.run(
        ['python3', 'smartqa.py', '--help'],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"CLI help failed with return code {result.returncode}"
    assert "Smart Document QA" in result.stdout, "CLI header not found"
    assert "--input" in result.stdout, "Input argument not found in help"
    assert "--ask" in result.stdout, "Ask argument not found in help"
    
    print("✅ CLI help test PASSED")


def test_cli_missing_input():
    result = subprocess.run(
        ['python3', 'smartqa.py'],
        capture_output=True,
        text=True
    )

    assert result.returncode != 0, "CLI should fail without input file"
    assert "error" in result.stdout.lower() or "usage" in result.stdout.lower(), "Should show error or usage"
    
    print("✅ CLI missing input test PASSED")


def test_cli_file_not_found():
    result = subprocess.run(
        ['python3', 'smartqa.py', '--input', 'nonexistent_file.txt', '--ask', 'test'],
        capture_output=True,
        text=True
    )

    assert result.returncode == 1, "CLI should exit with code 1 for missing file"
    assert "File 'nonexistent_file.txt' not found" in result.stdout, "Should show error message in stdout"
    
    print("✅ CLI file not found test PASSED")


def test_cli_basic_functionality():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write("Artificial intelligence is a branch of computer science.")
        tmp_file_path = tmp_file.name

    try:
        result = subprocess.run(
            ['python3', 'smartqa.py', '--input', tmp_file_path, '--ask', 'What is AI?'],
            capture_output=True,
            text=True,
            timeout=60
        )

        assert result.returncode == 0, f"CLI failed with return code {result.returncode}"
        assert "ANSWER" in result.stdout or "Answer" in result.stdout, "Should show answer"
        
        print("✅ CLI basic functionality test PASSED")
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path) 