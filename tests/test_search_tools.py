"""Tests for code search tools (grep_pattern, find_definitions)."""

import pytest
from src.agents.tools import InMemoryFileBackend, LocalFileBackend
from src.agents.tools.search import (
    grep_pattern,
    find_definitions,
    _find_python_definitions,
    _find_js_definitions,
    _build_method_set,
    create_search_tools,
)
import ast


# =============================================================================
# grep_pattern tests
# =============================================================================


class TestGrepPattern:
    """Tests for the grep_pattern function."""

    def test_basic_match(self, tmp_path):
        """Test finding a simple pattern in a file."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "main.py").write_text("def hello():\n    print('hello world')\n")

        backend = LocalFileBackend(repo)
        result = grep_pattern(backend, "hello")

        assert result.error is None
        assert result.total_matches >= 1
        assert any("hello" in m.content for m in result.matches)

    def test_invalid_regex(self):
        """Test that invalid regex returns an error, not an exception."""
        backend = InMemoryFileBackend(files={"test.py": "content"})
        result = grep_pattern(backend, "[invalid")

        assert result.error is not None
        assert "Invalid regex" in result.error
        assert result.matches == []
        assert result.total_matches == 0

    def test_max_results_limits_output(self, tmp_path):
        """Test that max_results limits the returned matches."""
        repo = tmp_path / "repo"
        repo.mkdir()
        # Create a file with many matches
        lines = [f"match line {i}" for i in range(20)]
        (repo / "data.py").write_text("\n".join(lines))

        backend = LocalFileBackend(repo)
        result = grep_pattern(backend, "match", max_results=5)

        assert len(result.matches) == 5
        assert result.total_matches >= 5

    def test_no_matches(self, tmp_path):
        """Test searching for a pattern that doesn't exist."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "main.py").write_text("def foo(): pass\n")

        backend = LocalFileBackend(repo)
        result = grep_pattern(backend, "nonexistent_xyz")

        assert result.error is None
        assert result.matches == []
        assert result.total_matches == 0

    def test_specific_file_path(self, tmp_path):
        """Test searching within a specific file."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("target line\n")
        (repo / "b.py").write_text("target line\n")

        backend = LocalFileBackend(repo)
        result = grep_pattern(backend, "target", path="a.py")

        assert result.total_matches == 1
        assert all(m.path == "a.py" for m in result.matches)

    def test_context_lines(self, tmp_path):
        """Test that context lines are included."""
        repo = tmp_path / "repo"
        repo.mkdir()
        content = "line1\nline2\nMATCH\nline4\nline5\n"
        (repo / "test.py").write_text(content)

        backend = LocalFileBackend(repo)
        result = grep_pattern(backend, "MATCH", context_lines=1)

        assert len(result.matches) == 1
        match = result.matches[0]
        assert "line2" in match.context_before
        assert "line4" in match.context_after

    def test_case_insensitive(self, tmp_path):
        """Test that search is case-insensitive."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "test.py").write_text("Hello World\n")

        backend = LocalFileBackend(repo)
        result = grep_pattern(backend, "hello world")

        assert result.total_matches == 1

    def test_empty_file(self, tmp_path):
        """Test searching an empty file."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "empty.py").write_text("")

        backend = LocalFileBackend(repo)
        result = grep_pattern(backend, "anything")

        assert result.error is None
        assert result.matches == []

    def test_files_searched_count(self, tmp_path):
        """Test that files_searched count is accurate."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("content a")
        (repo / "b.py").write_text("content b")

        backend = LocalFileBackend(repo)
        result = grep_pattern(backend, "content")

        assert result.files_searched == 2


# =============================================================================
# _find_python_definitions tests
# =============================================================================


class TestFindPythonDefinitions:
    """Tests for Python definition finding."""

    def test_find_function(self):
        """Test finding a simple function."""
        code = "def hello_world():\n    pass\n"
        result = _find_python_definitions(code, "hello", "test.py")

        assert len(result) == 1
        assert result[0].name == "hello_world"
        assert result[0].def_type == "function"
        assert result[0].line_num == 1

    def test_find_async_function(self):
        """Test finding an async function."""
        code = "async def fetch_data():\n    pass\n"
        result = _find_python_definitions(code, "fetch", "test.py")

        assert len(result) == 1
        assert result[0].name == "fetch_data"
        assert result[0].def_type == "function"
        assert "async def" in result[0].signature

    def test_find_class(self):
        """Test finding a class definition."""
        code = "class MyService:\n    pass\n"
        result = _find_python_definitions(code, "service", "test.py")

        assert len(result) == 1
        assert result[0].name == "MyService"
        assert result[0].def_type == "class"

    def test_find_method(self):
        """Test finding a method inside a class."""
        code = "class Foo:\n    def bar(self):\n        pass\n"
        result = _find_python_definitions(code, "bar", "test.py")

        assert len(result) == 1
        assert result[0].name == "bar"
        assert result[0].def_type == "method"

    def test_filter_by_type(self):
        """Test filtering by definition type."""
        code = "def foo(): pass\nclass Foo: pass\n"
        funcs = _find_python_definitions(code, "foo", "test.py", def_type="function")
        classes = _find_python_definitions(code, "foo", "test.py", def_type="class")

        assert len(funcs) == 1
        assert funcs[0].def_type == "function"
        assert len(classes) == 1
        assert classes[0].def_type == "class"

    def test_partial_name_match(self):
        """Test that partial name matching works."""
        code = "def create_user(): pass\ndef delete_user(): pass\ndef get_item(): pass\n"
        result = _find_python_definitions(code, "user", "test.py")

        assert len(result) == 2
        names = {r.name for r in result}
        assert names == {"create_user", "delete_user"}

    def test_syntax_error_returns_empty(self):
        """Test that invalid Python returns empty list, not an exception."""
        code = "def broken(:\n"
        result = _find_python_definitions(code, "broken", "test.py")

        assert result == []

    def test_signature_extraction(self):
        """Test that function signatures are extracted."""
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        result = _find_python_definitions(code, "add", "test.py")

        assert len(result) == 1
        assert "a: int" in result[0].signature
        assert "b: int" in result[0].signature
        assert "-> int" in result[0].signature

    def test_no_matches(self):
        """Test searching for a name that doesn't exist."""
        code = "def foo(): pass\n"
        result = _find_python_definitions(code, "nonexistent", "test.py")

        assert result == []


# =============================================================================
# _build_method_set tests
# =============================================================================


class TestBuildMethodSet:
    """Tests for the _build_method_set helper."""

    def test_identifies_methods(self):
        """Test that methods inside classes are identified."""
        code = "class A:\n    def method(self): pass\ndef standalone(): pass\n"
        tree = ast.parse(code)
        method_ids = _build_method_set(tree)

        # There should be exactly one method
        assert len(method_ids) == 1

    def test_nested_classes(self):
        """Test methods in nested classes."""
        code = (
            "class Outer:\n"
            "    def outer_method(self): pass\n"
            "    class Inner:\n"
            "        def inner_method(self): pass\n"
        )
        tree = ast.parse(code)
        method_ids = _build_method_set(tree)

        assert len(method_ids) == 2

    def test_no_classes(self):
        """Test with no classes — should return empty set."""
        code = "def foo(): pass\ndef bar(): pass\n"
        tree = ast.parse(code)
        method_ids = _build_method_set(tree)

        assert len(method_ids) == 0


# =============================================================================
# _find_js_definitions tests
# =============================================================================


class TestFindJsDefinitions:
    """Tests for JavaScript/TypeScript definition finding."""

    def test_find_function_declaration(self):
        """Test finding a function declaration."""
        code = "function handleClick() {\n  console.log('clicked');\n}\n"
        result = _find_js_definitions(code, "handle", "app.js")

        assert len(result) == 1
        assert result[0].name == "handleClick"
        assert result[0].def_type == "function"

    def test_find_async_function(self):
        """Test finding an async function."""
        code = "async function fetchData() {\n  return await api.get();\n}\n"
        result = _find_js_definitions(code, "fetch", "app.js")

        assert len(result) == 1
        assert result[0].name == "fetchData"

    def test_find_class(self):
        """Test finding a class declaration."""
        code = "class UserService {\n  constructor() {}\n}\n"
        result = _find_js_definitions(code, "user", "app.js")

        assert len(result) == 1
        assert result[0].name == "UserService"
        assert result[0].def_type == "class"

    def test_find_arrow_function(self):
        """Test finding an arrow function."""
        code = "const processData = (items) => {\n  return items.map(x => x);\n};\n"
        result = _find_js_definitions(code, "process", "app.js")

        assert len(result) == 1
        assert result[0].name == "processData"
        assert result[0].def_type == "function"

    def test_arrow_function_no_false_positive(self):
        """Test that parenthesized expressions aren't matched as arrow functions."""
        code = "const result = (1 + 2);\n"
        result = _find_js_definitions(code, "result", "app.js")

        # Should NOT match as a function since there's no =>
        func_matches = [r for r in result if r.def_type == "function"]
        assert len(func_matches) == 0

    def test_find_exported_function(self):
        """Test finding an exported function."""
        code = "export function createApp() {\n  return new App();\n}\n"
        result = _find_js_definitions(code, "create", "app.js")

        assert len(result) == 1
        assert result[0].name == "createApp"

    def test_find_exported_arrow_function(self):
        """Test finding an exported arrow function."""
        code = "export const handler = (req, res) => {\n  res.send('ok');\n};\n"
        result = _find_js_definitions(code, "handler", "app.js")

        assert len(result) == 1
        assert result[0].name == "handler"

    def test_arrow_function_without_parens(self):
        """Test finding arrow function with single param (no parens)."""
        code = "const double = x => x * 2;\n"
        result = _find_js_definitions(code, "double", "app.js")

        assert len(result) == 1
        assert result[0].name == "double"
        assert result[0].def_type == "function"

    def test_filter_by_type(self):
        """Test filtering JS definitions by type."""
        code = "function foo() {}\nclass Foo {}\n"
        funcs = _find_js_definitions(code, "foo", "app.js", def_type="function")
        classes = _find_js_definitions(code, "foo", "app.js", def_type="class")

        assert len(funcs) == 1
        assert funcs[0].def_type == "function"
        assert len(classes) == 1
        assert classes[0].def_type == "class"

    def test_no_matches(self):
        """Test searching for a name that doesn't exist."""
        code = "function foo() {}\n"
        result = _find_js_definitions(code, "nonexistent", "app.js")

        assert result == []

    def test_object_variable(self):
        """Test finding a const object declaration."""
        code = "const config = {\n  port: 3000,\n};\n"
        result = _find_js_definitions(code, "config", "app.js")

        assert len(result) == 1
        assert result[0].name == "config"
        assert result[0].def_type == "variable"


# =============================================================================
# find_definitions integration tests
# =============================================================================


class TestFindDefinitions:
    """Integration tests for find_definitions using the full pipeline."""

    def test_finds_python_definitions(self, tmp_path):
        """Test finding definitions across Python files."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "models.py").write_text("class User:\n    def save(self): pass\n")
        (repo / "views.py").write_text("def get_user(): pass\n")

        backend = LocalFileBackend(repo)
        result = find_definitions(backend, "user")

        assert result.error is None
        names = {d.name for d in result.definitions}
        assert "User" in names
        assert "get_user" in names

    def test_finds_js_definitions(self, tmp_path):
        """Test finding definitions in JS files."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "app.js").write_text("function createApp() {\n  return {};\n}\n")

        backend = LocalFileBackend(repo)
        result = find_definitions(backend, "create")

        assert result.error is None
        assert len(result.definitions) == 1
        assert result.definitions[0].name == "createApp"

    def test_empty_repo(self, tmp_path):
        """Test searching an empty repository."""
        repo = tmp_path / "repo"
        repo.mkdir()

        backend = LocalFileBackend(repo)
        result = find_definitions(backend, "anything")

        assert result.error is None
        assert result.definitions == []


# =============================================================================
# create_search_tools tests
# =============================================================================


class TestCreateSearchTools:
    """Tests for the LangChain tool factory."""

    def test_creates_two_tools(self, tmp_path):
        """Test that factory creates grep and find_definitions tools."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "test.py").write_text("content")

        backend = LocalFileBackend(repo)
        tools = create_search_tools(backend)

        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert "grep_in_files" in tool_names
        assert "find_code_definitions" in tool_names

    def test_grep_tool_invocation(self, tmp_path):
        """Test that the grep tool can be invoked."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "test.py").write_text("hello world\n")

        backend = LocalFileBackend(repo)
        tools = create_search_tools(backend)
        grep_tool = next(t for t in tools if t.name == "grep_in_files")

        result = grep_tool.invoke({"pattern": "hello"})
        assert result["total_matches"] >= 1

    def test_find_defs_tool_invocation(self, tmp_path):
        """Test that the find_definitions tool can be invoked."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "utils.py").write_text("def helper(): pass\n")

        backend = LocalFileBackend(repo)
        tools = create_search_tools(backend)
        find_tool = next(t for t in tools if t.name == "find_code_definitions")

        result = find_tool.invoke({"name": "helper"})
        assert len(result["definitions"]) == 1
