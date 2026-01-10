# Implementation Plan: Enhanced File Search with Line Number References

## Overview
Enhance the scoped context pipeline to find specific code locations with line numbers and include a **Code References** section in generated context output.

## Implementation Steps

### Step 1: Add New Schemas
**File:** `src/agents/scoper/tools/schemas.py`

Add three new Pydantic models:

```python
class LineMatch(BaseModel):
    """A single line match from grep/search."""
    line_num: int = Field(description="1-indexed line number")
    content: str = Field(description="The line content (stripped)")
    context_before: list[str] = Field(default_factory=list, description="Lines before match")
    context_after: list[str] = Field(default_factory=list, description="Lines after match")

class CodeReference(BaseModel):
    """A code reference for output."""
    path: str = Field(description="File path")
    line_start: int = Field(description="Starting line number")
    line_end: int | None = Field(default=None, description="Ending line (None for single line)")
    description: str = Field(description="Brief description of what this code does")

class GrepOutput(BaseModel):
    """Output schema for grep_pattern tool."""
    matches: list[dict] = Field(description="List of {path, line_num, content, context}")
    total_matches: int = Field(description="Total matches found")
    pattern: str = Field(description="Pattern that was searched")
    error: str | None = Field(default=None)

class DefinitionMatch(BaseModel):
    """A code definition match."""
    name: str = Field(description="Name of the definition")
    type: str = Field(description="Type: function, class, method, variable")
    path: str = Field(description="File path")
    line_num: int = Field(description="Line number where definition starts")
    line_end: int | None = Field(default=None, description="Line where definition ends")
    signature: str | None = Field(default=None, description="Function/method signature if applicable")

class FindDefinitionsOutput(BaseModel):
    """Output schema for find_definitions tool."""
    definitions: list[DefinitionMatch] = Field(description="Matching definitions")
    name_searched: str = Field(description="Name that was searched")
    error: str | None = Field(default=None)
```

Update `FileMatch` to include optional line matches:
```python
class FileMatch(BaseModel):
    path: str
    match_type: str
    score: int
    line_matches: list[LineMatch] = Field(default_factory=list, description="Specific line matches if available")
```

---

### Step 2: Implement grep_pattern Tool
**File:** `src/agents/scoper/tools/code_search_tools.py` (new file)

Core function:
```python
def grep_pattern(
    backend: FileBackend,
    pattern: str,
    path: str | None = None,
    max_results: int = 50,
    context_lines: int = 2,
) -> GrepOutput:
    """Search for pattern in files, returning matches with line numbers."""
```

Implementation approach:
- If `path` provided: search only that file
- If `path` is None: search all files in repo (respecting IGNORED_DIRS, SEARCHABLE_EXTENSIONS)
- Use `re.compile(pattern)` for regex matching
- Return line number, content, and context lines for each match
- Limit to `max_results` total matches

Create LangChain tool wrapper:
```python
@tool
def grep_in_files(pattern: str, path: str | None = None, max_results: int = 50) -> dict:
    """Search for a regex pattern in repository files.

    Returns matches with file paths, line numbers, and surrounding context.
    Use this to find specific code patterns, function calls, or text.

    Args:
        pattern: Regex pattern to search for
        path: Optional specific file to search (searches all files if None)
        max_results: Maximum matches to return (default: 50)
    """
```

---

### Step 3: Implement find_definitions Tool
**File:** `src/agents/scoper/tools/code_search_tools.py`

Core function:
```python
def find_definitions(
    backend: FileBackend,
    name: str,
    def_type: str | None = None,  # "function", "class", "method", or None for all
) -> FindDefinitionsOutput:
    """Find code definitions by name across the repository."""
```

**Python implementation** (using AST):
```python
def _find_python_definitions(content: str, name: str, file_path: str) -> list[DefinitionMatch]:
    """Use ast.parse to find function/class definitions."""
    tree = ast.parse(content)
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and name in node.name:
            matches.append(DefinitionMatch(
                name=node.name,
                type="function",
                path=file_path,
                line_num=node.lineno,
                line_end=node.end_lineno,
                signature=_extract_signature(node),
            ))
        elif isinstance(node, ast.ClassDef) and name in node.name:
            # Similar handling for classes
    return matches
```

**JavaScript/TypeScript implementation** (using regex):
```python
def _find_js_definitions(content: str, name: str, file_path: str) -> list[DefinitionMatch]:
    """Use regex patterns for JS/TS definitions."""
    patterns = [
        (r'function\s+(\w*{name}\w*)\s*\(', 'function'),
        (r'class\s+(\w*{name}\w*)', 'class'),
        (r'const\s+(\w*{name}\w*)\s*=\s*(?:async\s*)?\(', 'function'),  # arrow functions
        (r'const\s+(\w*{name}\w*)\s*=\s*\{', 'variable'),  # object definitions
    ]
    # Match and extract line numbers
```

Create LangChain tool wrapper:
```python
@tool
def find_code_definitions(name: str, def_type: str | None = None) -> dict:
    """Find function, class, or method definitions by name.

    Searches Python and JavaScript/TypeScript files for definitions
    matching the given name (partial matches supported).

    Args:
        name: Name to search for (partial match)
        def_type: Optional filter: "function", "class", "method", or None for all
    """
```

---

### Step 4: Enhance Discovery Phase
**File:** `src/agents/scoper/discovery.py`

Add new function:
```python
def search_with_line_numbers(
    repo_path: Path,
    keywords: List[str],
    max_results: int = 20,
    max_lines_per_file: int = 5,
) -> List[Dict]:
    """Search for files and return specific line matches.

    Returns:
        List of dicts with: path, match_type, score, line_matches
        where line_matches is list of {line_num, content}
    """
```

Modify content matching in `search_relevant_files` to optionally capture line numbers:
- When a keyword matches file content, record which lines contain the match
- Store first N line matches per file
- Include in return dict

---

### Step 5: Update Scoped Generator
**File:** `src/agents/scoper/scoped_generator.py`

Add CodeReference tracking:
```python
class ScopedGenerator:
    def generate(
        self,
        # ... existing params ...
        code_references: list[CodeReference] | None = None,  # NEW
    ) -> Path:
```

Add method to format code references section:
```python
def _format_code_references(self, references: list[CodeReference]) -> str:
    """Format code references as markdown section."""
    if not references:
        return ""

    lines = ["## Code References", ""]
    for ref in references:
        if ref.line_end and ref.line_end != ref.line_start:
            line_range = f"{ref.line_start}-{ref.line_end}"
        else:
            line_range = str(ref.line_start)
        lines.append(f"- `{ref.path}:{line_range}` - {ref.description}")

    return "\n".join(lines)
```

Update `_build_context_file` to include code references section after main content.

---

### Step 6: Update Prompts
**File:** `src/agents/llm/prompts.py`

Update `SCOPE_EXPLORATION_PROMPT` to request line tracking:
```python
SCOPE_EXPLORATION_PROMPT = """...existing content...

When examining files, note important line numbers/ranges for:
- Key function or class definitions
- Important logic or configuration
- Entry points relevant to the question

Include in your preliminary_insights any notable file:line references you discover.
"""
```

Update `SCOPE_GENERATION_PROMPT` to include Code References:
```python
SCOPE_GENERATION_PROMPT = """...existing content...

5. Code References - List of specific file:line-range references with brief descriptions
   Format each as: `path/to/file.py:start-end` - Description of what this code does

Example:
## Code References
- `src/auth/login.py:45-78` - Main authentication flow
- `src/models/user.py:12-35` - User model definition
"""
```

---

### Step 7: Wire Up New Tools
**File:** `src/agents/scoper/tools/__init__.py`

Export new tools:
```python
from .code_search_tools import create_code_search_tools, grep_pattern, find_definitions
```

**File:** `src/agents/scoper/agent.py`

Add code search tools to the agent's tool list alongside existing file_tools and analysis_tools.

---

### Step 8: Update ScopedAnalyzer
**File:** `src/agents/scoper/scoped_analyzer.py`

Update return type to include code references:
```python
def analyze(...) -> Dict[str, Any]:
    # ... existing logic ...
    return {
        "relevant_files": examined_files,
        "insights": "\n".join(all_insights),
        "code_references": collected_references,  # NEW
    }
```

Parse code references from LLM insights (extract file:line patterns from preliminary_insights).

---

## File Summary

| File | Action |
|------|--------|
| `src/agents/scoper/tools/schemas.py` | Add LineMatch, CodeReference, GrepOutput, DefinitionMatch, FindDefinitionsOutput |
| `src/agents/scoper/tools/code_search_tools.py` | **NEW** - grep_pattern, find_definitions implementations |
| `src/agents/scoper/discovery.py` | Add search_with_line_numbers function |
| `src/agents/scoper/scoped_generator.py` | Add code_references param, _format_code_references method |
| `src/agents/llm/prompts.py` | Update SCOPE_EXPLORATION_PROMPT, SCOPE_GENERATION_PROMPT |
| `src/agents/scoper/tools/__init__.py` | Export new tools |
| `src/agents/scoper/agent.py` | Wire up code search tools |
| `src/agents/scoper/scoped_analyzer.py` | Track and return code references |

## Testing Plan

1. Unit tests for `grep_pattern` with various regex patterns
2. Unit tests for `find_definitions` with Python and JS/TS files
3. Unit tests for `search_with_line_numbers`
4. Integration test: run scoped context generation and verify Code References section appears
5. Test with the research-example.md style output expectation

## Success Criteria

- [ ] `grep_in_files` tool returns matches with line numbers
- [ ] `find_code_definitions` tool finds Python and JS/TS definitions
- [ ] Discovery phase can return line-level matches
- [ ] Generated scoped context includes Code References section
- [ ] References follow format: `path:line-range` - Description
