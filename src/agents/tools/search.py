"""Code search tools using the FileBackend abstraction.

Provides grep-like pattern search and code definition finding capabilities.
"""

import ast
import re
from pathlib import PurePosixPath
from langchain_core.tools import tool, BaseTool

from .backends import FileBackend, DEFAULT_IGNORED_DIRS, DEFAULT_SEARCHABLE_EXTENSIONS
from .schemas import (
    GrepOutput,
    GrepMatch,
    FindDefinitionsOutput,
    DefinitionMatch,
)


# =============================================================================
# Core Functions (Backend-aware)
# =============================================================================


def grep_pattern(
    backend: FileBackend,
    pattern: str,
    path: str | None = None,
    max_results: int = 50,
    context_lines: int = 2,
) -> GrepOutput:
    """Search for a regex pattern in files, returning matches with line numbers.

    Args:
        backend: File backend to use for file access
        pattern: Regex pattern to search for
        path: Optional specific file to search (searches all files if None)
        max_results: Maximum number of matches to return
        context_lines: Number of context lines before/after each match

    Returns:
        GrepOutput with matches and metadata.
        Note: total_matches counts all matches in files that were searched,
        but files are skipped once max_results is reached, so the count
        may underrepresent the true total across the entire repository.
    """
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return GrepOutput(
            matches=[],
            total_matches=0,
            pattern=pattern,
            files_searched=0,
            error=f"Invalid regex pattern: {e}",
        )

    matches: list[GrepMatch] = []
    files_searched = 0
    total_found = 0

    if path:
        # Search specific file
        files_to_search = [path]
    else:
        # Search all searchable files via backend abstraction
        files_to_search = [
            f for f in backend.walk_files(ignore_dirs=DEFAULT_IGNORED_DIRS)
            if PurePosixPath(f).suffix.lower() in DEFAULT_SEARCHABLE_EXTENSIONS
        ]

    for file_path in files_to_search:
        if len(matches) >= max_results:
            break

        content = backend.read_file(file_path)
        if content is None:
            continue

        files_searched += 1
        lines = content.splitlines()

        for i, line in enumerate(lines):
            if regex.search(line):
                total_found += 1
                if len(matches) >= max_results:
                    continue  # Keep counting but don't add more matches

                # Get context lines
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)

                context_before = lines[start:i]
                context_after = lines[i + 1 : end]

                matches.append(
                    GrepMatch(
                        path=file_path,
                        line_num=i + 1,  # 1-indexed
                        content=line.strip(),
                        context_before=context_before,
                        context_after=context_after,
                    )
                )

    return GrepOutput(
        matches=matches,
        total_matches=total_found,
        pattern=pattern,
        files_searched=files_searched,
        error=None,
    )


def find_definitions(
    backend: FileBackend,
    name: str,
    def_type: str | None = None,
) -> FindDefinitionsOutput:
    """Find code definitions by name across the repository.

    Args:
        backend: File backend to use
        name: Name to search for (partial match supported)
        def_type: Optional filter: "function", "class", "method", or None for all

    Returns:
        FindDefinitionsOutput with matching definitions
    """
    definitions: list[DefinitionMatch] = []
    files_searched = 0

    files_to_search = [
        f for f in backend.walk_files(ignore_dirs=DEFAULT_IGNORED_DIRS)
        if PurePosixPath(f).suffix.lower() in DEFAULT_SEARCHABLE_EXTENSIONS
    ]

    for file_path in files_to_search:
        content = backend.read_file(file_path)
        if content is None:
            continue

        files_searched += 1
        suffix = PurePosixPath(file_path).suffix.lower()

        if suffix == ".py":
            file_defs = _find_python_definitions(content, name, file_path, def_type)
        elif suffix in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
            file_defs = _find_js_definitions(content, name, file_path, def_type)
        else:
            continue

        definitions.extend(file_defs)

    return FindDefinitionsOutput(
        definitions=definitions,
        name_searched=name,
        files_searched=files_searched,
        error=None,
    )


def _find_python_definitions(
    content: str,
    name: str,
    file_path: str,
    def_type: str | None = None,
) -> list[DefinitionMatch]:
    """Find Python definitions using AST.

    Args:
        content: Python source code
        name: Name to search for (case-insensitive partial match)
        file_path: Path to file
        def_type: Optional filter for definition type

    Returns:
        List of matching definitions
    """
    matches: list[DefinitionMatch] = []
    name_lower = name.lower()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return matches

    lines = content.splitlines()
    method_ids = _build_method_set(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            if name_lower not in node.name.lower():
                continue

            # Check if it's a method (inside a class)
            node_type = "method" if id(node) in method_ids else "function"

            if def_type and def_type != node_type:
                continue

            signature = _extract_python_signature(node, lines)
            matches.append(
                DefinitionMatch(
                    name=node.name,
                    def_type=node_type,
                    path=file_path,
                    line_num=node.lineno,
                    line_end=node.end_lineno,
                    signature=signature,
                )
            )

        elif isinstance(node, ast.ClassDef):
            if name_lower not in node.name.lower():
                continue

            if def_type and def_type != "class":
                continue

            matches.append(
                DefinitionMatch(
                    name=node.name,
                    def_type="class",
                    path=file_path,
                    line_num=node.lineno,
                    line_end=node.end_lineno,
                    signature=f"class {node.name}",
                )
            )

    return matches


def _build_method_set(tree: ast.AST) -> set[int]:
    """Build set of node ids that are methods (direct children of classes).

    Walking the tree once and collecting ids is O(n) rather than
    O(n*m) when calling _is_method per function node.
    """
    method_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_ids.add(id(child))
    return method_ids


def _extract_python_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]
) -> str:
    """Extract function signature from AST node."""
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "

    # Build signature from AST
    args = []
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            try:
                arg_str += f": {ast.unparse(arg.annotation)}"
            except (AttributeError, ValueError):
                pass  # ast.unparse can fail on malformed AST nodes; omit annotation
        args.append(arg_str)

    signature = f"{prefix}{node.name}({', '.join(args)})"

    # Add return type if present
    if node.returns:
        try:
            signature += f" -> {ast.unparse(node.returns)}"
        except (AttributeError, ValueError):
            pass  # ast.unparse can fail on malformed AST nodes; omit return type

    return signature


def _find_js_definitions(
    content: str,
    name: str,
    file_path: str,
    def_type: str | None = None,
) -> list[DefinitionMatch]:
    """Find JavaScript/TypeScript definitions using regex.

    Args:
        content: JS/TS source code
        name: Name to search for (case-insensitive partial match)
        file_path: Path to file
        def_type: Optional filter for definition type

    Returns:
        List of matching definitions
    """
    matches: list[DefinitionMatch] = []
    name_pattern = re.escape(name)
    lines = content.splitlines()

    # Patterns for different definition types
    patterns = [
        # function declarations: function name(...) or async function name(...)
        (
            rf"^(\s*)(?:export\s+)?(?:async\s+)?function\s+(\w*{name_pattern}\w*)\s*\(",
            "function",
        ),
        # class declarations: class Name
        (rf"^(\s*)(?:export\s+)?class\s+(\w*{name_pattern}\w*)", "class"),
        # arrow functions: const name = (...) => or const name = async (...) =>
        (
            rf"^(\s*)(?:export\s+)?(?:const|let|var)\s+(\w*{name_pattern}\w*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
            "function",
        ),
        # arrow functions without parens: const name = x =>
        (
            rf"^(\s*)(?:export\s+)?(?:const|let|var)\s+(\w*{name_pattern}\w*)\s*=\s*(?:async\s*)?\w+\s*=>",
            "function",
        ),
        # object/variable: const name = {
        (
            rf"^(\s*)(?:export\s+)?(?:const|let|var)\s+(\w*{name_pattern}\w*)\s*=\s*\{{",
            "variable",
        ),
    ]

    for line_num, line in enumerate(lines, 1):
        for pattern, found_type in patterns:
            if def_type and def_type != found_type:
                continue

            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                found_name = match.group(2)

                # Extract signature (first line of definition)
                signature = line.strip()
                if len(signature) > 100:
                    signature = signature[:100] + "..."

                # Try to find end line (simple heuristic: matching braces)
                line_end = _find_js_definition_end(lines, line_num - 1)

                matches.append(
                    DefinitionMatch(
                        name=found_name,
                        def_type=found_type,
                        path=file_path,
                        line_num=line_num,
                        line_end=line_end,
                        signature=signature,
                    )
                )
                break  # Only match one pattern per line

    return matches


def _find_js_definition_end(lines: list[str], start_idx: int) -> int | None:
    """Find the end line of a JS definition by tracking braces.

    Simple heuristic that works for most formatted code.

    Known limitation: brace counting does not account for braces inside
    string literals, template literals, or comments. A full JS parser
    would be needed for correctness, but this heuristic is sufficient
    for typical formatted source files.
    """
    brace_count = 0
    started = False

    for i in range(start_idx, min(start_idx + 500, len(lines))):
        line = lines[i]

        for char in line:
            if char == "{":
                brace_count += 1
                started = True
            elif char == "}":
                brace_count -= 1

        # Definition ends when we've seen braces and they're balanced
        if started and brace_count == 0:
            return i + 1  # 1-indexed

    return None


# =============================================================================
# LangChain Tool Factory
# =============================================================================


def create_search_tools(backend: FileBackend) -> list[BaseTool]:
    """Create code search tools bound to a specific backend.

    Args:
        backend: File backend to bind tools to

    Returns:
        List of LangChain tools ready for agent use
    """

    @tool
    def grep_in_files(
        pattern: str, path: str | None = None, max_results: int = 50
    ) -> dict:
        """Search for a regex pattern in repository files.

        Returns matches with file paths, line numbers, and surrounding context.
        Use this to find specific code patterns, function calls, error messages,
        or any text across the codebase.

        Args:
            pattern: Regex pattern to search for (case-insensitive)
            path: Optional specific file to search (searches all files if None)
            max_results: Maximum matches to return (default: 50)

        Returns:
            Dictionary with:
            - matches: List of {path, line_num, content, context_before, context_after}
            - total_matches: Total matches found (may exceed max_results)
            - pattern: The pattern that was searched
            - files_searched: Number of files searched
            - error: Error message if search failed
        """
        result = grep_pattern(backend, pattern, path, max_results)
        return result.model_dump()

    @tool
    def find_code_definitions(name: str, def_type: str | None = None) -> dict:
        """Find function, class, or method definitions by name.

        Searches Python and JavaScript/TypeScript files for definitions
        matching the given name. Supports partial matching.

        Args:
            name: Name to search for (partial match, case-insensitive)
            def_type: Optional filter: "function", "class", "method", or None for all

        Returns:
            Dictionary with:
            - definitions: List of {name, def_type, path, line_num, line_end, signature}
            - name_searched: The name that was searched
            - files_searched: Number of files searched
            - error: Error message if search failed
        """
        result = find_definitions(backend, name, def_type)
        return result.model_dump()

    return [grep_in_files, find_code_definitions]
