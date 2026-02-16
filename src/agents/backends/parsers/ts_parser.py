"""JavaScript/TypeScript parser using tree-sitter."""

from __future__ import annotations

from pathlib import PurePosixPath

import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser

from ..models import SymbolInfo, SymbolDetail

JS_LANGUAGE = Language(tsjs.language())
TS_LANGUAGE = Language(tsts.language_typescript())
TSX_LANGUAGE = Language(tsts.language_tsx())

# Map file extensions to tree-sitter languages
_LANG_MAP: dict[str, Language] = {
    ".js": JS_LANGUAGE,
    ".jsx": JS_LANGUAGE,
    ".mjs": JS_LANGUAGE,
    ".ts": TS_LANGUAGE,
    ".tsx": TSX_LANGUAGE,
}


class TSParser:
    """Parse JS/TS source using tree-sitter."""

    def _get_language(self, file_path: str) -> Language:
        """Pick the right tree-sitter language from file extension."""
        ext = PurePosixPath(file_path).suffix.lower()
        return _LANG_MAP.get(ext, JS_LANGUAGE)

    def _parse(self, source: str, file_path: str):
        """Parse source and return tree."""
        lang = self._get_language(file_path)
        parser = Parser(lang)
        return parser.parse(source.encode())

    def get_symbols(self, source: str, file_path: str) -> list[SymbolInfo]:
        """Extract top-level symbols from JS/TS source."""
        tree = self._parse(source, file_path)
        lines = source.splitlines()
        symbols: list[SymbolInfo] = []

        for node in tree.root_node.children:
            sym = self._node_to_symbol(node, lines, source)
            if sym:
                symbols.append(sym)

        return symbols

    def get_imports(self, source: str) -> list[str]:
        """Extract import module names from JS/TS source."""
        # Use JS parser — import syntax is the same
        parser = Parser(JS_LANGUAGE)
        tree = parser.parse(source.encode())
        imports: list[str] = []

        for node in tree.root_node.children:
            if node.type == "import_statement":
                source_node = node.child_by_field_name("source")
                if source_node:
                    module = source_node.text.decode().strip("'\"")
                    imports.append(module)

        return imports

    def extract_symbol(self, source: str, symbol_name: str, file_path: str = "file.js") -> SymbolDetail | None:
        """Extract a specific symbol's full body."""
        tree = self._parse(source, file_path)
        lines = source.splitlines()

        return self._find_symbol_in_tree(tree.root_node, symbol_name, lines, source, parent=None)

    def _find_symbol_in_tree(self, root, symbol_name: str, lines: list[str], source: str, parent: str | None) -> SymbolDetail | None:
        """Recursively search for a named symbol."""
        for node in root.children:
            name = self._get_node_name(node)

            # Check class bodies for methods
            if node.type in ("class_declaration", "class"):
                if name == symbol_name:
                    return self._node_to_detail(node, lines, source, parent=None, kind="class")
                body = node.child_by_field_name("body")
                if body:
                    result = self._find_symbol_in_tree(body, symbol_name, lines, source, parent=name)
                    if result:
                        return result
            elif node.type == "export_statement":
                # Check exported declarations
                declaration = node.child_by_field_name("declaration")
                if declaration:
                    dname = self._get_node_name(declaration)
                    if dname == symbol_name:
                        return self._node_to_detail(declaration, lines, source, parent=parent, kind=self._get_kind(declaration))
                    # Check class bodies inside exports
                    if declaration.type in ("class_declaration", "class"):
                        body = declaration.child_by_field_name("body")
                        if body:
                            result = self._find_symbol_in_tree(body, symbol_name, lines, source, parent=dname)
                            if result:
                                return result
            elif name == symbol_name:
                return self._node_to_detail(node, lines, source, parent=parent, kind=self._get_kind(node))

        return None

    def _node_to_symbol(self, node, lines: list[str], source: str) -> SymbolInfo | None:
        """Convert a tree-sitter node to SymbolInfo."""
        # Handle export wrappers
        actual = node
        if node.type == "export_statement":
            declaration = node.child_by_field_name("declaration")
            if declaration:
                actual = declaration
            else:
                return None

        name = self._get_node_name(actual)
        if not name:
            return None

        kind = self._get_kind(actual)
        if not kind:
            return None

        start_line = actual.start_point[0] + 1  # 1-indexed
        end_line = actual.end_point[0] + 1

        signature = lines[start_line - 1].strip() if start_line <= len(lines) else name

        children: list[SymbolInfo] = []
        if kind in ("class", "interface"):
            body = actual.child_by_field_name("body")
            if body:
                for child in body.children:
                    child_sym = self._method_to_symbol(child, lines)
                    if child_sym:
                        children.append(child_sym)

        return SymbolInfo(
            name=name,
            kind=kind,
            line=start_line,
            line_end=end_line,
            signature=signature,
            children=children,
        )

    def _method_to_symbol(self, node, lines: list[str]) -> SymbolInfo | None:
        """Convert a class body member to SymbolInfo."""
        if node.type not in ("method_definition", "public_field_definition", "property_definition"):
            return None

        name = self._get_node_name(node)
        if not name:
            return None

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        signature = lines[start_line - 1].strip() if start_line <= len(lines) else name

        return SymbolInfo(
            name=name,
            kind="method",
            line=start_line,
            line_end=end_line,
            signature=signature,
        )

    def _node_to_detail(self, node, lines: list[str], source: str, parent: str | None, kind: str) -> SymbolDetail:
        """Convert a tree-sitter node to SymbolDetail."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        body = "\n".join(lines[start_line - 1 : end_line])
        signature = lines[start_line - 1].strip() if start_line <= len(lines) else ""

        return SymbolDetail(
            name=self._get_node_name(node) or "",
            kind="method" if parent else kind,
            line=start_line,
            line_end=end_line,
            signature=signature,
            body=body,
            parent=parent,
            char_count=len(body),
        )

    def _get_node_name(self, node) -> str | None:
        """Extract name from various node types."""
        # Direct name field
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode()

        # Lexical declaration: const FOO = ...
        if node.type in ("lexical_declaration", "variable_declaration"):
            for child in node.children:
                if child.type == "variable_declarator":
                    name_n = child.child_by_field_name("name")
                    if name_n:
                        return name_n.text.decode()

        return None

    def _get_kind(self, node) -> str | None:
        """Determine the symbol kind from node type."""
        kind_map = {
            "function_declaration": "function",
            "class_declaration": "class",
            "interface_declaration": "interface",
            "method_definition": "method",
            "lexical_declaration": "variable",
            "variable_declaration": "variable",
        }
        return kind_map.get(node.type)
