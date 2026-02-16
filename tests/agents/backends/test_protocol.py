"""Tests for FileAnalysisBackend protocol."""

from src.agents.backends.protocol import FileAnalysisBackend
from src.agents.backends.models import SymbolInfo, SymbolDetail, FileOutline, Reference


class FakeAnalysisBackend:
    """Minimal implementation to verify protocol conformance."""

    def get_outline(self, file_path: str, source: str) -> FileOutline:
        return FileOutline(path=file_path, language="python")

    def read_symbol(self, file_path: str, symbol_name: str, source: str) -> SymbolDetail | None:
        return None

    def find_references(self, symbol_name: str, file_backend, scope: str | None = None) -> list[Reference]:
        return []


class TestFileAnalysisBackendProtocol:
    def test_fake_backend_satisfies_protocol(self):
        backend = FakeAnalysisBackend()
        assert isinstance(backend, FileAnalysisBackend)

    def test_class_without_methods_fails_protocol(self):
        class Incomplete:
            pass

        assert not isinstance(Incomplete(), FileAnalysisBackend)
