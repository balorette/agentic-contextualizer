"""Python parser using stdlib ast module."""

from __future__ import annotations

import ast
from ..models import SymbolInfo, SymbolDetail


class PythonParser:
    """Parse Python source using stdlib ast."""

    def get_symbols(self, source: str, file_path: str) -> list[SymbolInfo]:
        """Extract top-level symbols from Python source."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        lines = source.splitlines()
        symbols: list[SymbolInfo] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                symbols.append(self._func_to_symbol(node, lines))
            elif isinstance(node, ast.ClassDef):
                symbols.append(self._class_to_symbol(node, lines))

        return symbols

    def get_imports(self, source: str) -> list[str]:
        """Extract import module names from Python source."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return imports

    def extract_symbol(self, source: str, symbol_name: str) -> SymbolDetail | None:
        """Extract a specific symbol's full body."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        lines = source.splitlines()

        # Search top-level and class methods
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name == symbol_name:
                    parent = self._find_parent_class(tree, node)
                    return self._node_to_detail(node, lines, parent)
            elif isinstance(node, ast.ClassDef):
                if node.name == symbol_name:
                    return self._node_to_detail(node, lines, parent=None)

        return None

    def _func_to_symbol(self, node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]) -> SymbolInfo:
        """Convert a function AST node to SymbolInfo."""
        return SymbolInfo(
            name=node.name,
            kind="function",
            line=node.lineno,
            line_end=node.end_lineno or node.lineno,
            signature=self._extract_signature(node, lines),
            decorators=self._extract_decorators(node),
            docstring=self._extract_docstring(node),
        )

    def _class_to_symbol(self, node: ast.ClassDef, lines: list[str]) -> SymbolInfo:
        """Convert a class AST node to SymbolInfo with method children."""
        children: list[SymbolInfo] = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                sym = SymbolInfo(
                    name=child.name,
                    kind="method",
                    line=child.lineno,
                    line_end=child.end_lineno or child.lineno,
                    signature=self._extract_signature(child, lines),
                    decorators=self._extract_decorators(child),
                    docstring=self._extract_docstring(child),
                )
                children.append(sym)

        return SymbolInfo(
            name=node.name,
            kind="class",
            line=node.lineno,
            line_end=node.end_lineno or node.lineno,
            signature=self._extract_signature(node, lines),
            children=children,
            decorators=self._extract_decorators(node),
            docstring=self._extract_docstring(node),
        )

    def _node_to_detail(
        self,
        node: ast.AST,
        lines: list[str],
        parent: str | None,
    ) -> SymbolDetail:
        """Convert an AST node to SymbolDetail with body."""
        start = node.lineno - 1  # 0-indexed
        end = (node.end_lineno or node.lineno)
        body = "\n".join(lines[start:end])
        kind = "method" if parent else ("class" if isinstance(node, ast.ClassDef) else "function")

        return SymbolDetail(
            name=node.name,
            kind=kind,
            line=node.lineno,
            line_end=end,
            signature=self._extract_signature(node, lines),
            decorators=self._extract_decorators(node),
            docstring=self._extract_docstring(node),
            body=body,
            parent=parent,
            char_count=len(body),
        )

    def _extract_signature(self, node: ast.AST, lines: list[str]) -> str:
        """Extract the signature line from source."""
        line = lines[node.lineno - 1].strip()
        return line

    def _extract_decorators(self, node: ast.AST) -> list[str]:
        """Extract decorator names."""
        decorators: list[str] = []
        for dec in getattr(node, "decorator_list", []):
            if isinstance(dec, ast.Name):
                decorators.append(f"@{dec.id}")
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"@{ast.unparse(dec)}")
            elif isinstance(dec, ast.Call):
                decorators.append(f"@{ast.unparse(dec.func)}")
            else:
                decorators.append(f"@{ast.unparse(dec)}")
        return decorators

    def _extract_docstring(self, node: ast.AST) -> str | None:
        """Extract first line of docstring if present."""
        doc = ast.get_docstring(node)
        if doc:
            return doc.split("\n")[0].strip()
        return None

    def _find_parent_class(self, tree: ast.Module, target: ast.AST) -> str | None:
        """Find the enclosing class name for a method."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in ast.iter_child_nodes(node):
                    if child is target:
                        return node.name
        return None
