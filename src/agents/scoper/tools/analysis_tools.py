"""Code analysis tools for scoped context generation."""

import ast
import re
from pathlib import PurePosixPath
from langchain_core.tools import tool, BaseTool

from ..backends import FileBackend
from .schemas import ExtractImportsOutput, ImportInfo


def _parse_python_imports(content: str, file_path: str) -> list[ImportInfo]:
    """Parse imports from Python source code using AST.

    Args:
        content: Python source code
        file_path: Path to file (for resolving relative imports)

    Returns:
        List of ImportInfo objects
    """
    imports: list[ImportInfo] = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ImportInfo(
                        module=alias.name,
                        names=[],
                        alias=alias.asname,
                        is_relative=False,
                        resolved_path=_resolve_python_import(alias.name, file_path, False),
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            is_relative = node.level > 0
            names = [alias.name for alias in node.names]

            # Build full module path for relative imports
            if is_relative:
                dots = "." * node.level
                full_module = f"{dots}{module}" if module else dots
            else:
                full_module = module

            imports.append(
                ImportInfo(
                    module=full_module,
                    names=names,
                    alias=None,  # from imports don't have module aliases
                    is_relative=is_relative,
                    resolved_path=_resolve_python_import(
                        module, file_path, is_relative, node.level
                    ),
                )
            )

    return imports


def _resolve_python_import(
    module: str,
    file_path: str,
    is_relative: bool,
    level: int = 0,
) -> str | None:
    """Try to resolve a Python import to a file path within the repo.

    Args:
        module: Module name (e.g., "src.utils" or "utils")
        file_path: Path of the importing file
        is_relative: Whether this is a relative import
        level: Number of dots for relative imports

    Returns:
        Resolved path like "src/utils.py" or None if can't resolve
    """
    if not module and not is_relative:
        return None

    if is_relative:
        # Relative import: go up `level` directories from current file
        current = PurePosixPath(file_path).parent
        for _ in range(level - 1):  # -1 because level=1 means current package
            current = current.parent

        if module:
            parts = module.split(".")
            resolved = current / "/".join(parts)
        else:
            resolved = current
    else:
        # Absolute import: convert dots to path
        parts = module.split(".")
        resolved = PurePosixPath("/".join(parts))

    # Return as .py file path (could also be __init__.py in a package)
    return f"{resolved}.py"


def _parse_js_imports(content: str) -> list[ImportInfo]:
    """Parse imports from JavaScript/TypeScript using regex.

    Handles common patterns:
    - import X from 'module'
    - import { X, Y } from 'module'
    - import * as X from 'module'
    - const X = require('module')

    Args:
        content: JavaScript/TypeScript source code

    Returns:
        List of ImportInfo objects
    """
    imports: list[ImportInfo] = []

    # ES6 imports: import X from 'module' or import { X } from 'module'
    es6_pattern = r"import\s+(?:(\w+)|(?:\{([^}]+)\})|(?:\*\s+as\s+(\w+)))\s+from\s+['\"]([^'\"]+)['\"]"
    for match in re.finditer(es6_pattern, content):
        default_import, named_imports, namespace_import, module = match.groups()

        names = []
        alias = None

        if default_import:
            alias = default_import
        elif named_imports:
            names = [n.strip().split(" as ")[0].strip() for n in named_imports.split(",")]
        elif namespace_import:
            alias = namespace_import

        is_relative = module.startswith(".")

        imports.append(
            ImportInfo(
                module=module,
                names=names,
                alias=alias,
                is_relative=is_relative,
                resolved_path=_resolve_js_import(module) if is_relative else None,
            )
        )

    # CommonJS requires: const X = require('module')
    require_pattern = r"(?:const|let|var)\s+(\w+)\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
    for match in re.finditer(require_pattern, content):
        var_name, module = match.groups()
        is_relative = module.startswith(".")

        imports.append(
            ImportInfo(
                module=module,
                names=[],
                alias=var_name,
                is_relative=is_relative,
                resolved_path=_resolve_js_import(module) if is_relative else None,
            )
        )

    return imports


def _resolve_js_import(module: str) -> str | None:
    """Try to resolve a JS/TS relative import to a file path.

    Args:
        module: Module path (e.g., "./utils" or "../lib/helper")

    Returns:
        Resolved path or None
    """
    if not module.startswith("."):
        return None

    # Remove leading ./ and add extension
    path = module.lstrip("./")

    # Could be .js, .ts, .tsx, .jsx, or index file
    # Return the base path, agent can try variations
    return f"{path}.ts"  # Default to .ts, could be .js


def _detect_language(file_path: str) -> str:
    """Detect language from file extension.

    Args:
        file_path: Path to file

    Returns:
        Language identifier: 'python', 'javascript', 'typescript', 'unknown'
    """
    suffix = PurePosixPath(file_path).suffix.lower()

    if suffix == ".py":
        return "python"
    elif suffix in (".js", ".jsx", ".mjs", ".cjs"):
        return "javascript"
    elif suffix in (".ts", ".tsx", ".mts", ".cts"):
        return "typescript"
    else:
        return "unknown"


def extract_imports(
    backend: FileBackend,
    file_path: str,
) -> ExtractImportsOutput:
    """Extract import statements from a source file.

    Args:
        backend: File backend to use
        file_path: Relative path to file

    Returns:
        ExtractImportsOutput with imports and metadata
    """
    content = backend.read_file(file_path)

    if content is None:
        return ExtractImportsOutput(
            imports=[],
            path=file_path,
            language="unknown",
            error=f"Could not read file: {file_path}",
        )

    language = _detect_language(file_path)

    if language == "python":
        imports = _parse_python_imports(content, file_path)
    elif language in ("javascript", "typescript"):
        imports = _parse_js_imports(content)
    else:
        return ExtractImportsOutput(
            imports=[],
            path=file_path,
            language=language,
            error=f"Import extraction not supported for language: {language}",
        )

    return ExtractImportsOutput(
        imports=imports,
        path=file_path,
        language=language,
        error=None,
    )


def create_analysis_tools(backend: FileBackend) -> list[BaseTool]:
    """Create analysis tools bound to a specific backend.

    Args:
        backend: File backend to bind tools to

    Returns:
        List of LangChain tools ready for agent use
    """

    @tool
    def extract_file_imports(file_path: str) -> dict:
        """Extract import statements from a Python or JavaScript/TypeScript file.

        Use this to identify dependencies and related files to examine.
        Supports Python (ast parsing) and JS/TS (regex patterns).

        Args:
            file_path: Relative path to file within the repository

        Returns:
            Dictionary with:
            - imports: List of imports with module, names, alias, is_relative, resolved_path
            - path: Path of the analyzed file
            - language: Detected language
            - error: Error message if extraction failed
        """
        result = extract_imports(backend, file_path)
        return result.model_dump()

    return [extract_file_imports]
