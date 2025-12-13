# Scoped Context Security & Quality Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address security vulnerability, quality issues, and GitHub PR review comments for the scoped context generation feature.

**Architecture:** Fix path traversal vulnerability, improve context file repo detection, fix import paths, add documentation comments, improve filename uniqueness, and enhance test assertions.

**Tech Stack:** Python 3.12, Pydantic, pytest, pathlib, yaml

---

## Task 1: Fix Path Traversal Vulnerability in ScopedAnalyzer

**Files:**
- Modify: `src/agents/scoper/scoped_analyzer.py:143-152`
- Create: `tests/scoper/test_path_traversal.py`

**Step 1: Write the failing test for path traversal protection**

Create `tests/scoper/test_path_traversal.py`:

```python
"""Tests for path traversal protection in scoped analyzer."""

import pytest
from pathlib import Path
from src.agents.scoper.scoped_analyzer import ScopedAnalyzer
from src.agents.llm.provider import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        return LLMResponse(content="{}", model="mock")

    def generate_structured(self, prompt: str, system: str | None, schema):
        return schema(
            additional_files_needed=[],
            reasoning="Done",
            sufficient_context=True,
            preliminary_insights="Test",
        )


class TestPathTraversalProtection:
    """Test that path traversal attacks are prevented."""

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a sample repository."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "src").mkdir()
        (repo / "src" / "safe.py").write_text("safe content")
        return repo

    @pytest.fixture
    def sensitive_file(self, tmp_path):
        """Create a sensitive file outside repo."""
        secret = tmp_path / "secret.txt"
        secret.write_text("sensitive data")
        return secret

    def test_read_file_blocks_parent_traversal(self, sample_repo, sensitive_file):
        """Test that ../path traversal is blocked."""
        mock_llm = MockLLMProvider()
        analyzer = ScopedAnalyzer(mock_llm)

        # Attempt to read file outside repo via parent traversal
        result = analyzer._read_file(sample_repo, "../secret.txt")

        assert result is None, "Should block parent directory traversal"

    def test_read_file_blocks_absolute_path(self, sample_repo, sensitive_file):
        """Test that absolute paths outside repo are blocked."""
        mock_llm = MockLLMProvider()
        analyzer = ScopedAnalyzer(mock_llm)

        # Attempt to read via absolute path
        result = analyzer._read_file(sample_repo, str(sensitive_file))

        assert result is None, "Should block absolute paths outside repo"

    def test_read_file_allows_valid_paths(self, sample_repo):
        """Test that valid paths within repo still work."""
        mock_llm = MockLLMProvider()
        analyzer = ScopedAnalyzer(mock_llm)

        result = analyzer._read_file(sample_repo, "src/safe.py")

        assert result == "safe content", "Should allow valid repo paths"

    def test_read_file_blocks_symlink_escape(self, sample_repo, tmp_path):
        """Test that symlinks pointing outside repo are blocked."""
        # Create symlink pointing outside repo
        secret = tmp_path / "secret.txt"
        secret.write_text("sensitive via symlink")
        symlink = sample_repo / "src" / "sneaky_link"
        symlink.symlink_to(secret)

        mock_llm = MockLLMProvider()
        analyzer = ScopedAnalyzer(mock_llm)

        result = analyzer._read_file(sample_repo, "src/sneaky_link")

        assert result is None, "Should block symlinks escaping repo"
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/scoper/test_path_traversal.py -v`

Expected: FAIL - tests will fail because current implementation doesn't validate paths

**Step 3: Implement path validation in _read_file**

Modify `src/agents/scoper/scoped_analyzer.py`. Replace the `_read_file` method (lines 143-152):

```python
    def _read_file(self, repo_path: Path, file_path: str) -> str | None:
        """Read file content safely with path traversal protection.

        Args:
            repo_path: Root path of the repository
            file_path: Relative path to file within repo

        Returns:
            File content as string, or None if file cannot be read safely
        """
        try:
            # Resolve to absolute path and check it stays within repo
            full_path = (repo_path / file_path).resolve()
            repo_resolved = repo_path.resolve()

            # Verify the resolved path is within the repository
            try:
                full_path.relative_to(repo_resolved)
            except ValueError:
                # Path escapes repository boundary
                return None

            # Check it's a real file (not symlink to outside)
            if full_path.is_symlink():
                # Resolve symlink and verify target is also in repo
                real_path = full_path.resolve()
                try:
                    real_path.relative_to(repo_resolved)
                except ValueError:
                    return None

            if full_path.exists() and full_path.is_file():
                if full_path.stat().st_size < 500_000:  # 500KB limit
                    return full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            # File access errors (permissions, broken symlinks, etc.)
            pass
        return None
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/scoper/test_path_traversal.py -v`

Expected: PASS - all 4 tests should pass

**Step 5: Run all scoper tests to verify no regressions**

Run: `source .venv/bin/activate && pytest tests/scoper/ -v`

Expected: PASS - all 17 tests (13 existing + 4 new) should pass

**Step 6: Commit**

```bash
git add tests/scoper/test_path_traversal.py src/agents/scoper/scoped_analyzer.py
git commit -m "fix(security): add path traversal protection to ScopedAnalyzer

Prevents LLM-suggested paths from escaping repository boundary via:
- Parent directory traversal (../)
- Absolute paths
- Symlinks pointing outside repo

CWE-22 mitigation"
```

---

## Task 2: Fix Import Paths Across All Test Files

**Files:**
- Modify: `tests/scoper/test_discovery.py:4`
- Modify: `tests/scoper/test_scoped_analyzer.py:5-6`
- Modify: `tests/scoper/test_scoped_generator.py:4-5`
- Modify: `tests/test_models.py:3`
- Modify: `tests/test_scope_integration.py:6-7,141`

**Step 1: Fix imports in test_discovery.py**

In `tests/scoper/test_discovery.py`, change line 4 from:

```python
from agents.scoper.discovery import extract_keywords, search_relevant_files
```

To:

```python
from src.agents.scoper.discovery import extract_keywords, search_relevant_files
```

**Step 2: Fix imports in test_scoped_analyzer.py**

In `tests/scoper/test_scoped_analyzer.py`, change lines 5-6 from:

```python
from agents.scoper.scoped_analyzer import ScopedAnalyzer
from agents.llm.provider import LLMProvider, LLMResponse
```

To:

```python
from src.agents.scoper.scoped_analyzer import ScopedAnalyzer
from src.agents.llm.provider import LLMProvider, LLMResponse
```

**Step 3: Fix imports in test_scoped_generator.py**

In `tests/scoper/test_scoped_generator.py`, change lines 4-5 from:

```python
from agents.scoper.scoped_generator import ScopedGenerator
from agents.llm.provider import LLMProvider, LLMResponse
```

To:

```python
from src.agents.scoper.scoped_generator import ScopedGenerator
from src.agents.llm.provider import LLMProvider, LLMResponse
```

**Step 4: Fix imports in test_models.py**

In `tests/test_models.py`, change line 3 from:

```python
from agents.models import ScopedContextMetadata
```

To:

```python
from src.agents.models import ScopedContextMetadata
```

**Step 5: Fix imports in test_scope_integration.py**

In `tests/test_scope_integration.py`:

Change line 7 from:
```python
from agents.llm.provider import LLMResponse
```

To:
```python
from src.agents.llm.provider import LLMResponse
```

Change line 141 from:
```python
        from agents.scoper.discovery import extract_keywords, search_relevant_files
```

To:
```python
        from src.agents.scoper.discovery import extract_keywords, search_relevant_files
```

**Step 6: Run tests to verify imports work**

Run: `source .venv/bin/activate && pytest tests/scoper/ tests/test_models.py tests/test_scope_integration.py -v`

Expected: PASS - all tests should pass with corrected imports

**Step 7: Commit**

```bash
git add tests/scoper/test_discovery.py tests/scoper/test_scoped_analyzer.py tests/scoper/test_scoped_generator.py tests/test_models.py tests/test_scope_integration.py
git commit -m "fix: standardize import paths to use src.agents prefix

All test files now consistently use src.agents.* imports to match
the project's package structure."
```

---

## Task 3: Add Documentation Comments to Constants

**Files:**
- Modify: `src/agents/scoper/scoped_analyzer.py:9-16`
- Modify: `src/agents/scoper/scoped_generator.py:10`
- Modify: `src/agents/scoper/discovery.py:8-22`
- Modify: `.env.example:10`

**Step 1: Add comments to scoped_analyzer.py constants**

In `src/agents/scoper/scoped_analyzer.py`, replace lines 9-16:

From:
```python
MAX_EXPLORATION_ROUNDS = 3

MAX_FILE_CONTENT_CHARS = 15_000
```

To:
```python
# Maximum number of LLM-guided exploration rounds before forcing synthesis.
# This limits cost and latency by preventing excessive iterations.
MAX_EXPLORATION_ROUNDS = 3

# Maximum number of characters from each file to include in the LLM prompt.
# This helps avoid exceeding the LLM's context window and keeps prompts efficient.
# Value chosen based on typical LLM context limits and empirical prompt size testing.
MAX_FILE_CONTENT_CHARS = 15_000
```

**Step 2: Add comment to scoped_generator.py constant**

In `src/agents/scoper/scoped_generator.py`, replace line 10:

From:
```python
MAX_CONTENT_PER_FILE = 10_000
```

To:
```python
# Maximum number of characters from each file to include in the generation prompt.
# Files exceeding this limit are truncated to prevent LLM context overflow.
MAX_CONTENT_PER_FILE = 10_000
```

**Step 3: Add comment to discovery.py STOPWORDS**

In `src/agents/scoper/discovery.py`, add a comment before STOPWORDS (line 8):

From:
```python
# Common stopwords to filter out
STOPWORDS: Set[str] = {
```

To:
```python
# Common English stopwords plus domain-specific terms to filter from keyword extraction.
# Based on standard NLP stopword lists with additions for code-related queries
# (e.g., "functionality", "feature", "work" which appear in questions but aren't searchable).
STOPWORDS: Set[str] = {
```

**Step 4: Add unit comment to .env.example**

In `.env.example`, change line 10:

From:
```
MAX_FILE_SIZE=1000000
```

To:
```
# Maximum file size in bytes (default: 1MB)
MAX_FILE_SIZE=1000000
```

**Step 5: Run linter**

Run: `source .venv/bin/activate && ruff check src/agents/scoper/`

Expected: `All checks passed!`

**Step 6: Commit**

```bash
git add src/agents/scoper/scoped_analyzer.py src/agents/scoper/scoped_generator.py src/agents/scoper/discovery.py .env.example
git commit -m "docs: add explanatory comments to constants and configuration

Explains rationale for:
- MAX_EXPLORATION_ROUNDS (cost/latency control)
- MAX_FILE_CONTENT_CHARS (context window limits)
- MAX_CONTENT_PER_FILE (prompt size limits)
- STOPWORDS (NLP filtering rationale)
- MAX_FILE_SIZE (unit clarification)"
```

---

## Task 4: Add Comments to Empty Except Clauses

**Files:**
- Modify: `src/agents/scoper/discovery.py:130`
- Modify: `src/agents/scoper/scoped_analyzer.py:150` (already fixed in Task 1)

**Step 1: Add comment to discovery.py except clause**

In `src/agents/scoper/discovery.py`, change lines 130-131:

From:
```python
                except (OSError, UnicodeDecodeError):
                    pass
```

To:
```python
                except (OSError, UnicodeDecodeError):
                    # Skip files that can't be read (permissions, encoding issues)
                    # These are non-critical - we simply exclude them from search results
                    pass
```

**Step 2: Run linter**

Run: `source .venv/bin/activate && ruff check src/agents/scoper/discovery.py`

Expected: `All checks passed!`

**Step 3: Commit**

```bash
git add src/agents/scoper/discovery.py
git commit -m "docs: add explanatory comments to empty except clauses

Clarifies why exceptions are silently caught in file search operations."
```

---

## Task 5: Improve Context File Repo Detection

**Files:**
- Modify: `src/agents/main.py:365-420`
- Create: `tests/test_scope_context_file.py`

**Step 1: Write failing test for frontmatter repo extraction**

Create `tests/test_scope_context_file.py`:

```python
"""Tests for scoping from context files."""

import pytest
from pathlib import Path
from src.agents.main import _extract_repo_from_context


class TestExtractRepoFromContext:
    """Test extracting repo path from context file frontmatter."""

    def test_extract_repo_from_valid_context(self, tmp_path):
        """Test extracting source_repo from context file."""
        context_file = tmp_path / "context.md"
        context_file.write_text("""---
source_repo: /path/to/original/repo
scan_date: 2025-01-22T10:30:00Z
user_summary: "Test project"
model_used: claude-3-5-sonnet
---

# Repository Context
""")

        result = _extract_repo_from_context(context_file)

        assert result == "/path/to/original/repo"

    def test_extract_repo_returns_none_for_missing_field(self, tmp_path):
        """Test that missing source_repo returns None."""
        context_file = tmp_path / "context.md"
        context_file.write_text("""---
scan_date: 2025-01-22T10:30:00Z
---

# No source_repo field
""")

        result = _extract_repo_from_context(context_file)

        assert result is None

    def test_extract_repo_returns_none_for_invalid_yaml(self, tmp_path):
        """Test that invalid YAML returns None gracefully."""
        context_file = tmp_path / "context.md"
        context_file.write_text("# No frontmatter at all")

        result = _extract_repo_from_context(context_file)

        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_scope_context_file.py -v`

Expected: FAIL - `_extract_repo_from_context` doesn't exist

**Step 3: Add helper function to main.py**

Add to `src/agents/main.py` after the imports (around line 15):

```python
import yaml


def _extract_repo_from_context(context_path: Path) -> str | None:
    """Extract source_repo from context file frontmatter.

    Args:
        context_path: Path to context.md file

    Returns:
        Source repo path from frontmatter, or None if not found
    """
    try:
        content = context_path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return None

        # Find end of frontmatter
        end_marker = content.find("---", 3)
        if end_marker == -1:
            return None

        frontmatter_text = content[3:end_marker].strip()
        frontmatter = yaml.safe_load(frontmatter_text)

        if isinstance(frontmatter, dict):
            return frontmatter.get("source_repo")
    except Exception:
        # YAML parse errors, file read errors, etc.
        pass
    return None
```

**Step 4: Update _scope_pipeline_mode to use frontmatter and always scan**

In `src/agents/main.py`, update the `_scope_pipeline_mode` function (around line 365-420):

```python
def _scope_pipeline_mode(
    source_path: Path,
    question: str,
    config: Config,
    is_context_file: bool,
    output: str | None,
) -> int:
    """Generate scoped context using pipeline mode."""
    click.echo(f"🔍 Scoping: {question}")

    # Determine repo path
    if is_context_file:
        # Extract repo path from context file's frontmatter
        source_repo = _extract_repo_from_context(source_path)
        if source_repo and Path(source_repo).exists():
            repo_path = Path(source_repo)
            repo_name = repo_path.name
            click.echo(f"   Source: context file (repo: {repo_name})")
        else:
            # Fallback: infer from directory structure
            repo_name = source_path.parent.name
            repo_path = source_path.parent.parent.parent
            click.echo(f"   Source: context file for {repo_name} (inferred)")
            if not repo_path.exists():
                click.echo(f"   Warning: Could not locate repository at {repo_path}", err=True)
                click.echo(f"   Consider using --repo flag or ensure source_repo is set in context file", err=True)
        source_context = str(source_path)
    else:
        repo_path = source_path
        repo_name = source_path.name
        source_context = None
        click.echo(f"   Source: repository {repo_name}")

    # Phase 1: Discovery
    click.echo("\n📂 Phase 1: Discovery...")
    keywords = extract_keywords(question)
    click.echo(f"   Keywords: {', '.join(keywords)}")

    # Always scan repository structure for file discovery
    scanner = StructureScanner(config)
    structure = scanner.scan(repo_path)
    file_tree = structure["tree"]

    # Search for relevant files
    candidates = search_relevant_files(repo_path, keywords)
    click.echo(f"   Found {len(candidates)} candidate files")

    if not candidates:
        click.echo("   No matching files found. Using fallback search...")
        # Fallback: use all files from structure scan
        candidates = [
            {"path": f, "match_type": "fallback", "score": 1}
            for f in structure.get("all_files", [])[:20]
        ]
```

**Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_scope_context_file.py -v`

Expected: PASS - all 3 tests should pass

**Step 6: Run integration tests**

Run: `source .venv/bin/activate && pytest tests/test_scope_integration.py tests/test_scope_context_file.py -v`

Expected: PASS - all tests pass

**Step 7: Commit**

```bash
git add src/agents/main.py tests/test_scope_context_file.py
git commit -m "feat: improve context file repo detection and always scan for files

- Extract source_repo from context file YAML frontmatter
- Always scan repository even when scoping from context file
- Fix empty candidates issue when scoping from context
- Add warning when repo path cannot be determined"
```

---

## Task 6: Fix Type Annotation in _format_tree

**Files:**
- Modify: `src/agents/scoper/scoped_analyzer.py:154`

**Step 1: Update the type annotation**

In `src/agents/scoper/scoped_analyzer.py`, change the `_format_tree` method signature:

From:
```python
    def _format_tree(self, tree: Dict, indent: int = 0) -> str:
```

To:
```python
    def _format_tree(self, tree: Dict[str, Any], indent: int = 0) -> str:
```

**Step 2: Verify import exists**

Ensure `Any` is in the imports at top of file:
```python
from typing import Dict, List, Any
```

**Step 3: Run linter**

Run: `source .venv/bin/activate && ruff check src/agents/scoper/scoped_analyzer.py`

Expected: `All checks passed!`

**Step 4: Commit**

```bash
git add src/agents/scoper/scoped_analyzer.py
git commit -m "fix: add proper type annotation to _format_tree method"
```

---

## Task 7: Improve Filename Uniqueness in ScopedGenerator

**Files:**
- Modify: `src/agents/scoper/scoped_generator.py:98-103`
- Modify: `tests/scoper/test_scoped_generator.py`

**Step 1: Update _sanitize_filename to include timestamp**

In `src/agents/scoper/scoped_generator.py`, replace `_sanitize_filename` method:

From:
```python
    def _sanitize_filename(self, question: str) -> str:
        """Convert question to safe filename."""
        # Take first few words, remove special chars
        sanitized = re.sub(r"[^a-zA-Z0-9\s-]", "", question.lower())
        words = sanitized.split()[:4]
        return "-".join(words) if words else "context"
```

To:
```python
    def _sanitize_filename(self, question: str) -> str:
        """Convert question to safe, unique filename.

        Uses first 4 words of question plus timestamp suffix to ensure uniqueness
        even when multiple questions share the same prefix.
        """
        from datetime import datetime
        # Take first few words, remove special chars
        sanitized = re.sub(r"[^a-zA-Z0-9\s-]", "", question.lower())
        words = sanitized.split()[:4]
        base_name = "-".join(words) if words else "context"
        # Add timestamp suffix for uniqueness (HHMMSS format)
        timestamp = datetime.now().strftime("%H%M%S")
        return f"{base_name}-{timestamp}"
```

**Step 2: Update test to account for timestamp**

In `tests/scoper/test_scoped_generator.py`, update `test_generate_uses_sanitized_filename`:

From:
```python
    def test_generate_uses_sanitized_filename(self, output_dir):
        """Test that output filename is sanitized from question."""
        mock_llm = MockLLMProvider()
        generator = ScopedGenerator(mock_llm, output_dir)

        output_path = generator.generate(
            repo_name="test-repo",
            question="How does the auth/login flow work?",
            relevant_files={"src/auth.py": "def login(): pass"},
            insights="Auth flow",
            model_name="claude-3-5-sonnet",
        )

        # Filename should be sanitized (no special chars)
        assert "scope-" in output_path.name
        assert "/" not in output_path.name
        assert "?" not in output_path.name
```

To:
```python
    def test_generate_uses_sanitized_filename(self, output_dir):
        """Test that output filename is sanitized from question with timestamp."""
        mock_llm = MockLLMProvider()
        generator = ScopedGenerator(mock_llm, output_dir)

        output_path = generator.generate(
            repo_name="test-repo",
            question="How does the auth/login flow work?",
            relevant_files={"src/auth.py": "def login(): pass"},
            insights="Auth flow",
            model_name="claude-3-5-sonnet",
        )

        # Filename should be sanitized (no special chars) with timestamp
        assert "scope-" in output_path.name
        assert "/" not in output_path.name
        assert "?" not in output_path.name
        # Should have timestamp suffix (6 digits)
        import re
        assert re.search(r"-\d{6}\.md$", output_path.name), "Should have timestamp suffix"
```

**Step 3: Run tests**

Run: `source .venv/bin/activate && pytest tests/scoper/test_scoped_generator.py -v`

Expected: PASS

**Step 4: Commit**

```bash
git add src/agents/scoper/scoped_generator.py tests/scoper/test_scoped_generator.py
git commit -m "feat: add timestamp to scoped context filenames for uniqueness

Prevents filename collisions when multiple scope questions share
the same first 4 words. Format: scope-{words}-{HHMMSS}.md"
```

---

## Task 8: Improve Test Assertion in test_cli.py

**Files:**
- Modify: `tests/test_cli.py:531`

**Step 1: Find and update the test**

In `tests/test_cli.py`, find `test_scope_requires_question` and update the assertion:

From:
```python
    def test_scope_requires_question(self, runner, sample_repo):
        """Test that scope command requires --question flag."""
        result = runner.invoke(cli, ["scope", str(sample_repo)])

        assert result.exit_code != 0
        assert "error" in result.output.lower() or "missing" in result.output.lower()
```

To:
```python
    def test_scope_requires_question(self, runner, sample_repo):
        """Test that scope command requires --question flag."""
        result = runner.invoke(cli, ["scope", str(sample_repo)])

        assert result.exit_code != 0
        # Verify error specifically mentions the missing --question option
        assert "--question" in result.output or "-q" in result.output, \
            f"Error should mention missing --question flag, got: {result.output}"
```

**Step 2: Run test**

Run: `source .venv/bin/activate && pytest tests/test_cli.py::TestScopeCommand::test_scope_requires_question -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "test: improve assertion specificity for missing question flag

Now verifies the error message specifically mentions --question/-q
instead of just checking for generic error text."
```

---

## Task 9: Add Docstrings to Test Init Files

**Files:**
- Modify: `tests/test_scope_integration.py:31-32,53-54`

**Step 1: Add minimal docstrings to __init__.py creations in test fixture**

In `tests/test_scope_integration.py`, update the `sample_repo` fixture where it creates `__init__.py` files:

From:
```python
        (repo / "src" / "weather" / "__init__.py").write_text("")
```

To:
```python
        (repo / "src" / "weather" / "__init__.py").write_text('"""Weather module for forecast services."""\n')
```

From:
```python
        (repo / "src" / "auth" / "__init__.py").write_text("")
```

To:
```python
        (repo / "src" / "auth" / "__init__.py").write_text('"""Authentication module."""\n')
```

**Step 2: Run integration tests**

Run: `source .venv/bin/activate && pytest tests/test_scope_integration.py -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_scope_integration.py
git commit -m "test: add docstrings to __init__.py files in test fixtures

Makes test repository structure more realistic by including
minimal module docstrings."
```

---

## Task 10: Run Full Test Suite and Final Verification

**Files:**
- None (verification only)

**Step 1: Run all tests**

Run: `source .venv/bin/activate && pytest -v`

Expected: All tests pass

**Step 2: Run linter on all changed files**

Run: `source .venv/bin/activate && ruff check src/agents/scoper/ src/agents/main.py`

Expected: `All checks passed!`

**Step 3: Verify imports work**

Run: `source .venv/bin/activate && python -c "from src.agents.scoper import extract_keywords, search_relevant_files, ScopedAnalyzer, ScopedGenerator; from src.agents.main import _extract_repo_from_context; print('All imports OK')"`

Expected: `All imports OK`

**Step 4: Final commit (if any uncommitted changes)**

```bash
git status
# If clean, no action needed
```

---

## Summary of Changes

| Task | Files Changed | Issue Addressed |
|------|---------------|-----------------|
| 1 | `scoped_analyzer.py`, `test_path_traversal.py` | Path traversal security (CWE-22) |
| 2 | 5 test files | Import path consistency |
| 3 | 4 source files | Missing documentation comments |
| 4 | `discovery.py` | Empty except clause comments |
| 5 | `main.py`, `test_scope_context_file.py` | Context file repo detection + empty candidates |
| 6 | `scoped_analyzer.py` | Type annotation |
| 7 | `scoped_generator.py`, test | Filename uniqueness |
| 8 | `test_cli.py` | Test assertion specificity |
| 9 | `test_scope_integration.py` | Test realism |
| 10 | (verification) | Full test suite validation |

**Security Fixes:**
- CWE-22 Path Traversal - FIXED

**Bug Fixes:**
- Empty candidates when scoping from context file - FIXED
- Fragile repo path inference - FIXED
- Fallback logic not executing for context files - FIXED

**Quality Improvements:**
- Import paths standardized
- Constants documented
- Exception handling documented
- Filename collisions prevented
- Test assertions improved
- Test fixtures more realistic
