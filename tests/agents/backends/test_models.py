"""Tests for backend data models."""

from src.agents.backends.models import SymbolInfo, SymbolDetail, FileOutline, Reference


class TestSymbolInfo:
    def test_basic_function(self):
        sym = SymbolInfo(
            name="authenticate",
            kind="function",
            line=10,
            line_end=25,
            signature="def authenticate(token: str) -> bool",
        )
        assert sym.name == "authenticate"
        assert sym.kind == "function"
        assert sym.line == 10
        assert sym.line_end == 25
        assert sym.children == []
        assert sym.decorators == []
        assert sym.docstring is None

    def test_class_with_children(self):
        method = SymbolInfo(
            name="process",
            kind="method",
            line=15,
            line_end=25,
            signature="def process(self, data: str) -> None",
        )
        cls = SymbolInfo(
            name="Handler",
            kind="class",
            line=10,
            line_end=30,
            signature="class Handler(BaseHandler):",
            children=[method],
            decorators=["@dataclass"],
            docstring="Handles incoming requests.",
        )
        assert len(cls.children) == 1
        assert cls.children[0].name == "process"
        assert cls.decorators == ["@dataclass"]

    def test_to_dict(self):
        sym = SymbolInfo(
            name="foo",
            kind="function",
            line=1,
            line_end=5,
            signature="def foo()",
        )
        d = sym.to_dict()
        assert d["name"] == "foo"
        assert d["kind"] == "function"
        assert "children" in d


class TestSymbolDetail:
    def test_includes_body(self):
        detail = SymbolDetail(
            name="authenticate",
            kind="function",
            line=10,
            line_end=20,
            signature="def authenticate(token: str) -> bool",
            body='def authenticate(token: str) -> bool:\n    return validate(token)',
            parent=None,
            char_count=60,
        )
        assert detail.body.startswith("def authenticate")
        assert detail.char_count == 60
        assert detail.parent is None

    def test_method_with_parent(self):
        detail = SymbolDetail(
            name="process",
            kind="method",
            line=15,
            line_end=25,
            signature="def process(self) -> None",
            body="def process(self) -> None:\n    pass",
            parent="Handler",
            char_count=35,
        )
        assert detail.parent == "Handler"


class TestFileOutline:
    def test_empty_file(self):
        outline = FileOutline(
            path="src/empty.py",
            language="python",
            imports=[],
            symbols=[],
            line_count=0,
        )
        assert outline.path == "src/empty.py"
        assert outline.symbols == []

    def test_file_with_symbols(self):
        sym = SymbolInfo(
            name="main", kind="function", line=5, line_end=10,
            signature="def main()",
        )
        outline = FileOutline(
            path="src/main.py",
            language="python",
            imports=["os", "sys"],
            symbols=[sym],
            line_count=15,
        )
        assert len(outline.imports) == 2
        assert len(outline.symbols) == 1

    def test_to_dict(self):
        outline = FileOutline(
            path="src/app.py",
            language="python",
            imports=["flask"],
            symbols=[],
            line_count=20,
        )
        d = outline.to_dict()
        assert d["path"] == "src/app.py"
        assert d["language"] == "python"
        assert d["imports"] == ["flask"]


class TestReference:
    def test_basic_reference(self):
        ref = Reference(
            path="src/routes.py",
            line=34,
            context="user = handler.authenticate(token)",
        )
        assert ref.path == "src/routes.py"
        assert ref.line == 34

    def test_to_dict(self):
        ref = Reference(path="test.py", line=1, context="import foo")
        d = ref.to_dict()
        assert d["path"] == "test.py"
        assert d["line"] == 1
