"""Test the code-parser integration for token-efficient code indexing."""
import pytest
from pathlib import Path
from mini_coding_agent import CodeParser


@pytest.fixture
def sample_python_file(tmp_path):
    """Create a sample Python file for testing."""
    content = '''"""Sample module for testing code parser."""

import os
import sys


class UserService:
    """Service for managing users."""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def create_user(self, name, email):
        """Create a new user."""
        if not name or not email:
            raise ValueError("Name and email are required")
        return {"id": 1, "name": name, "email": email}
    
    def get_user(self, user_id):
        """Get a user by ID."""
        return {"id": user_id, "name": "Test"}
    
    def delete_user(self, user_id):
        """Delete a user."""
        pass


class AuthHandler:
    """Handle authentication."""
    
    def login(self, username, password):
        """Authenticate a user."""
        return True
    
    def logout(self, session_id):
        """End a session."""
        pass


def helper_function(x, y):
    """A top-level helper function."""
    return x + y


async def async_fetch_data(url):
    """Fetch data asynchronously."""
    return {"data": "test"}
'''
    filepath = tmp_path / "sample.py"
    filepath.write_text(content)
    return filepath


def test_parse_python_file(sample_python_file):
    """Test parsing a Python file extracts classes and methods correctly."""
    result = CodeParser.parse_file(sample_python_file)
    
    assert result is not None
    assert result["file"] == str(sample_python_file)
    assert result["language"] == "python"
    
    # Check classes
    assert len(result["classes"]) == 2
    
    # UserService should have 4 methods
    user_service = next(c for c in result["classes"] if c["name"] == "UserService")
    assert user_service["kind"] == "class"
    assert len(user_service["methods"]) == 4
    method_names = [m["name"] for m in user_service["methods"]]
    assert "__init__" in method_names
    assert "create_user" in method_names
    assert "get_user" in method_names
    assert "delete_user" in method_names
    
    # AuthHandler should have 2 methods
    auth_handler = next(c for c in result["classes"] if c["name"] == "AuthHandler")
    assert len(auth_handler["methods"]) == 2


def test_parse_directory(sample_python_file, tmp_path):
    """Test parsing a directory recursively."""
    # Create another file in a subdirectory
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    subfile = subdir / "other.py"
    subfile.write_text("def other_func():\n    pass\n")
    
    results = CodeParser.parse_directory(tmp_path)
    
    assert len(results) == 2
    files = {r["file"] for r in results}
    assert str(sample_python_file) in files
    assert str(subfile) in files


def test_to_index_summary(sample_python_file):
    """Test generating compact index summary."""
    result = CodeParser.parse_file(sample_python_file)
    summary = CodeParser.to_index_summary([result])
    
    assert "sample.py:" in summary
    assert "UserService (4 methods)" in summary
    assert "AuthHandler (2 methods)" in summary
    assert "helper_function" in summary
    assert "async_fetch_data" in summary


def test_find_symbol_class(sample_python_file):
    """Test finding a class by name."""
    result = CodeParser.parse_file(sample_python_file)
    found = CodeParser.find_symbol([result], "UserService")
    
    assert found is not None
    assert found["kind"] == "class"
    assert found["name"] == "UserService"
    assert found["file"] == str(sample_python_file)
    assert "line_start" in found
    assert "line_end" in found


def test_find_symbol_method(sample_python_file):
    """Test finding a method by name."""
    result = CodeParser.parse_file(sample_python_file)
    found = CodeParser.find_symbol([result], "create_user")
    
    assert found is not None
    assert found["kind"] == "method"
    assert found["name"] == "create_user"
    assert found["parent"] == "UserService"
    assert found["file"] == str(sample_python_file)


def test_find_symbol_function(sample_python_file):
    """Test finding a top-level function."""
    result = CodeParser.parse_file(sample_python_file)
    found = CodeParser.find_symbol([result], "helper_function")
    
    assert found is not None
    assert found["kind"] == "function"
    assert found["name"] == "helper_function"
    assert found["file"] == str(sample_python_file)


def test_find_symbol_not_found(sample_python_file):
    """Test searching for non-existent symbol."""
    result = CodeParser.parse_file(sample_python_file)
    found = CodeParser.find_symbol([result], "NonExistentClass")
    
    assert found is None


def test_find_symbols_in_file(sample_python_file):
    """Test finding multiple symbols at once."""
    results = CodeParser.find_symbols_in_file(
        sample_python_file, 
        ["UserService", "create_user", "helper_function", "NotExist"]
    )
    
    assert len(results) == 3
    assert "UserService" in results
    assert "create_user" in results
    assert "helper_function" in results
    assert "NotExist" not in results


def test_empty_file(tmp_path):
    """Test parsing an empty file."""
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")
    
    result = CodeParser.parse_file(empty_file)
    assert result is not None
    assert result["classes"] == []
    assert result["functions"] == []


def test_unsupported_extension(tmp_path):
    """Test that unsupported extensions return None."""
    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("Hello world")
    
    result = CodeParser.parse_file(txt_file)
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
