"""Data models for semantic file analysis."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SymbolInfo:
    """A symbol (function, class, method, variable) found in a file."""

    name: str
    kind: str  # "function", "class", "method", "variable"
    line: int
    line_end: int
    signature: str
    children: list[SymbolInfo] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "kind": self.kind,
            "line": self.line,
            "line_end": self.line_end,
            "signature": self.signature,
            "children": [c.to_dict() for c in self.children],
            "decorators": self.decorators,
        }
        if self.docstring:
            result["docstring"] = self.docstring
        return result


@dataclass
class SymbolDetail(SymbolInfo):
    """A symbol with its full source code body."""

    body: str = ""
    parent: str | None = None
    char_count: int = 0


@dataclass
class FileOutline:
    """Outline of a file: imports and symbols without bodies."""

    path: str
    language: str
    imports: list[str] = field(default_factory=list)
    symbols: list[SymbolInfo] = field(default_factory=list)
    line_count: int = 0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "language": self.language,
            "imports": self.imports,
            "symbols": [s.to_dict() for s in self.symbols],
            "line_count": self.line_count,
        }


@dataclass
class Reference:
    """A reference to a symbol in a file."""

    path: str
    line: int
    context: str

    def to_dict(self) -> dict:
        return {"path": self.path, "line": self.line, "context": self.context}
