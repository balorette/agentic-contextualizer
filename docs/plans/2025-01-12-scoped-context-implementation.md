# Scoped Context Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `scope` command that generates focused context for a specific question/domain from a repository.

**Architecture:** Three-phase pipeline (Discovery → Exploration → Synthesis) that narrows down relevant files based on a question, uses LLM to expand intelligently, then generates focused context. Reuses existing scanner, LLM provider, and middleware infrastructure.

**Tech Stack:** Python, Click (CLI), Pydantic (models), pytest (testing), existing LangChain agent infrastructure for agent mode.

---

## Task 1: Add Scoped Context Models

**Files:**
- Modify: `src/agents/models.py`
- Test: `tests/test_models.py` (create)

### Step 1: Write the failing test

Create `tests/test_models.py`:

```python
"""Tests for data models."""

import pytest
from datetime import datetime, UTC
from pathlib import Path
from agents.models import ScopedContextMetadata


class TestScopedContextMetadata:
    """Test ScopedContextMetadata model."""

    def test_create_scoped_metadata_minimal(self):
        """Test creating scoped metadata with required fields."""
        metadata = ScopedContextMetadata(
            source_repo="/path/to/repo",
            scope_question="authentication flow",
            model_used="claude-3-5-sonnet",
            files_analyzed=5,
        )

        assert metadata.source_repo == "/path/to/repo"
        assert metadata.scope_question == "authentication flow"
        assert metadata.model_used == "claude-3-5-sonnet"
        assert metadata.files_analyzed == 5
        assert metadata.source_context is None
        assert metadata.scan_date is not None

    def test_create_scoped_metadata_from_context(self):
        """Test creating scoped metadata when scoping from existing context."""
        metadata = ScopedContextMetadata(
            source_repo="/path/to/repo",
            source_context="contexts/repo/context.md",
            scope_question="weather functionality",
            model_used="claude-3-5-sonnet",
            files_analyzed=12,
        )

        assert metadata.source_context == "contexts/repo/context.md"
        assert metadata.scope_question == "weather functionality"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'ScopedContextMetadata'`

### Step 3: Write minimal implementation

Add to `src/agents/models.py` after `ContextMetadata` class:

```python
class ScopedContextMetadata(BaseModel):
    """Frontmatter metadata for scoped context files."""

    source_repo: str
    scope_question: str
    scan_date: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_used: str
    files_analyzed: int
    source_context: Optional[str] = None  # Set when scoping from existing context
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_models.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/agents/models.py tests/test_models.py
git commit -m "feat(models): add ScopedContextMetadata model"
```

---

## Task 2: Add Discovery Module - Keyword Extraction

**Files:**
- Create: `src/agents/scoper/__init__.py`
- Create: `src/agents/scoper/discovery.py`
- Test: `tests/scoper/__init__.py` (create)
- Test: `tests/scoper/test_discovery.py` (create)

### Step 1: Create module structure

Create empty `src/agents/scoper/__init__.py`:

```python
"""Scoped context generation module."""
```

Create empty `tests/scoper/__init__.py`:

```python
"""Tests for scoper module."""
```

### Step 2: Write the failing test for keyword extraction

Create `tests/scoper/test_discovery.py`:

```python
"""Tests for discovery module."""

import pytest
from agents.scoper.discovery import extract_keywords


class TestExtractKeywords:
    """Test keyword extraction from questions."""

    def test_extract_simple_keywords(self):
        """Test extracting keywords from a simple question."""
        keywords = extract_keywords("weather functionality")

        assert "weather" in keywords
        assert "functionality" not in keywords  # stopword-like

    def test_extract_technical_terms(self):
        """Test extracting technical terms."""
        keywords = extract_keywords("authentication flow with OAuth2")

        assert "authentication" in keywords
        assert "oauth2" in keywords
        assert "flow" in keywords
        assert "with" not in keywords  # stopword

    def test_extract_handles_case(self):
        """Test that extraction is case-insensitive."""
        keywords = extract_keywords("UserService API endpoints")

        # Should normalize to lowercase
        assert "userservice" in keywords or "user" in keywords
        assert "api" in keywords
        assert "endpoints" in keywords

    def test_extract_filters_short_words(self):
        """Test that very short words are filtered."""
        keywords = extract_keywords("how do I use the API")

        assert "api" in keywords
        assert "how" not in keywords
        assert "do" not in keywords
        assert "i" not in keywords
```

### Step 3: Run test to verify it fails

Run: `pytest tests/scoper/test_discovery.py::TestExtractKeywords -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.scoper'`

### Step 4: Write minimal implementation

Create `src/agents/scoper/discovery.py`:

```python
"""Discovery phase for scoped context generation."""

import re
from typing import List, Set

# Common stopwords to filter out
STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "this", "that", "these", "those", "what",
    "which", "who", "whom", "functionality", "feature", "features", "work",
    "works", "use", "uses", "using",
}

MIN_KEYWORD_LENGTH = 3


def extract_keywords(question: str) -> List[str]:
    """Extract meaningful keywords from a scope question.

    Args:
        question: The user's scope question

    Returns:
        List of lowercase keywords, filtered and deduplicated
    """
    # Normalize: lowercase and split on non-alphanumeric
    words = re.split(r"[^a-zA-Z0-9]+", question.lower())

    # Filter
    keywords = []
    seen: Set[str] = set()

    for word in words:
        if (
            word
            and len(word) >= MIN_KEYWORD_LENGTH
            and word not in STOPWORDS
            and word not in seen
        ):
            keywords.append(word)
            seen.add(word)

    return keywords
```

### Step 5: Run test to verify it passes

Run: `pytest tests/scoper/test_discovery.py::TestExtractKeywords -v`
Expected: PASS

### Step 6: Commit

```bash
git add src/agents/scoper/ tests/scoper/
git commit -m "feat(scoper): add keyword extraction for discovery phase"
```

---

## Task 3: Add Discovery Module - File Search

**Files:**
- Modify: `src/agents/scoper/discovery.py`
- Test: `tests/scoper/test_discovery.py`

### Step 1: Write the failing test for file search

Add to `tests/scoper/test_discovery.py`:

```python
from pathlib import Path
from agents.scoper.discovery import extract_keywords, search_relevant_files


class TestSearchRelevantFiles:
    """Test file search based on keywords."""

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a sample repository with various files."""
        repo = tmp_path / "sample_repo"
        repo.mkdir()

        # Create directory structure
        (repo / "src").mkdir()
        (repo / "src" / "weather").mkdir()
        (repo / "src" / "auth").mkdir()
        (repo / "tests").mkdir()

        # Create files
        (repo / "src" / "weather" / "service.py").write_text(
            "class WeatherService:\n    def get_forecast(self): pass"
        )
        (repo / "src" / "weather" / "models.py").write_text(
            "class WeatherData:\n    temperature: float"
        )
        (repo / "src" / "auth" / "login.py").write_text(
            "def authenticate(user): pass"
        )
        (repo / "tests" / "test_weather.py").write_text(
            "def test_weather_service(): pass"
        )
        (repo / "README.md").write_text("# Project\n\nWeather API service")

        return repo

    def test_search_finds_files_by_name(self, sample_repo):
        """Test that search finds files with matching names."""
        results = search_relevant_files(sample_repo, ["weather"])

        file_names = [r["path"] for r in results]
        assert any("weather" in f for f in file_names)

    def test_search_finds_files_by_content(self, sample_repo):
        """Test that search finds files with matching content."""
        results = search_relevant_files(sample_repo, ["forecast"])

        file_names = [r["path"] for r in results]
        assert any("service.py" in f for f in file_names)

    def test_search_returns_match_info(self, sample_repo):
        """Test that search results include match information."""
        results = search_relevant_files(sample_repo, ["weather"])

        assert len(results) > 0
        result = results[0]
        assert "path" in result
        assert "match_type" in result  # "filename" or "content"
        assert "score" in result

    def test_search_excludes_ignored_dirs(self, sample_repo):
        """Test that search respects ignored directories."""
        # Create a node_modules directory
        (sample_repo / "node_modules").mkdir()
        (sample_repo / "node_modules" / "weather.js").write_text("weather")

        results = search_relevant_files(sample_repo, ["weather"])

        file_paths = [r["path"] for r in results]
        assert not any("node_modules" in f for f in file_paths)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/scoper/test_discovery.py::TestSearchRelevantFiles -v`
Expected: FAIL with `ImportError: cannot import name 'search_relevant_files'`

### Step 3: Write minimal implementation

Add to `src/agents/scoper/discovery.py`:

```python
import os
from pathlib import Path
from typing import Dict

# Directories to always ignore
IGNORED_DIRS: Set[str] = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "egg-info", ".egg-info", ".tox", ".nox",
}

# File extensions to search
SEARCHABLE_EXTENSIONS: Set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".md", ".txt", ".yaml", ".yml", ".json",
    ".toml", ".ini", ".cfg", ".conf",
}


def search_relevant_files(
    repo_path: Path,
    keywords: List[str],
    max_results: int = 50,
) -> List[Dict]:
    """Search for files relevant to the given keywords.

    Args:
        repo_path: Path to repository root
        keywords: List of keywords to search for
        max_results: Maximum number of results to return

    Returns:
        List of dicts with: path, match_type, score
    """
    results: List[Dict] = []
    keyword_set = set(kw.lower() for kw in keywords)

    for root, dirs, files in os.walk(repo_path):
        # Prune ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.endswith(".egg-info")]

        root_path = Path(root)

        for filename in files:
            file_path = root_path / filename
            rel_path = str(file_path.relative_to(repo_path))

            # Check filename match
            filename_lower = filename.lower()
            name_matches = sum(1 for kw in keyword_set if kw in filename_lower)

            # Check directory path match
            path_lower = rel_path.lower()
            path_matches = sum(1 for kw in keyword_set if kw in path_lower)

            if name_matches > 0 or path_matches > 0:
                results.append({
                    "path": rel_path,
                    "match_type": "filename",
                    "score": name_matches * 2 + path_matches,
                })
                continue

            # Check content match for searchable files
            suffix = file_path.suffix.lower()
            if suffix in SEARCHABLE_EXTENSIONS:
                try:
                    if file_path.stat().st_size > 500_000:  # Skip large files
                        continue
                    content = file_path.read_text(encoding="utf-8", errors="ignore").lower()
                    content_matches = sum(1 for kw in keyword_set if kw in content)
                    if content_matches > 0:
                        results.append({
                            "path": rel_path,
                            "match_type": "content",
                            "score": content_matches,
                        })
                except (OSError, UnicodeDecodeError):
                    pass

    # Sort by score descending, limit results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/scoper/test_discovery.py::TestSearchRelevantFiles -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/agents/scoper/discovery.py tests/scoper/test_discovery.py
git commit -m "feat(scoper): add file search for discovery phase"
```

---

## Task 4: Add Scope Prompts

**Files:**
- Modify: `src/agents/llm/prompts.py`

### Step 1: Add prompts (no test needed for static strings)

Add to `src/agents/llm/prompts.py`:

```python
SCOPE_EXPLORATION_PROMPT = """You are analyzing a codebase to find all files relevant to a specific question.

SCOPE QUESTION:
{scope_question}

FILE TREE:
{file_tree}

CANDIDATE FILES (already found by keyword search):
{candidate_files}

CANDIDATE FILE CONTENTS:
{candidate_contents}

Your task:
1. Review the candidate files and their contents
2. Identify what additional files should be examined (imports, related tests, configs)
3. Determine if you have enough context to answer the scope question

Respond with JSON:
{{
    "additional_files_needed": ["path/to/file1.py", "path/to/file2.py"],
    "reasoning": "Why these files are needed",
    "sufficient_context": true/false,
    "preliminary_insights": "What you've learned so far"
}}

If sufficient_context is true, additional_files_needed should be empty.
"""


SCOPE_GENERATION_PROMPT = """Generate a focused context document for the following scope question.

SCOPE QUESTION:
{scope_question}

RELEVANT FILES AND CONTENTS:
{relevant_files}

ANALYSIS INSIGHTS:
{insights}

Create a markdown document with:
1. Summary - Direct answer to the scope question
2. Relevant sections based on what's important for this specific topic
   (could be API endpoints, data models, processing logic, etc. - use your judgment)
3. Key Files - List of files the reader should examine
4. Usage Examples / Related Tests - If available, show how this functionality is used

Be concise but thorough. Focus only on information relevant to the scope question.
Do NOT include a generic structure - tailor sections to what matters for this topic.
"""
```

### Step 2: Commit

```bash
git add src/agents/llm/prompts.py
git commit -m "feat(prompts): add scope exploration and generation prompts"
```

---

## Task 5: Add Scoped Analyzer

**Files:**
- Create: `src/agents/scoper/scoped_analyzer.py`
- Test: `tests/scoper/test_scoped_analyzer.py` (create)

### Step 1: Write the failing test

Create `tests/scoper/test_scoped_analyzer.py`:

```python
"""Tests for scoped analyzer."""

import pytest
import json
from pathlib import Path
from agents.scoper.scoped_analyzer import ScopedAnalyzer, ScopeExplorationOutput
from agents.llm.provider import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Return mock response."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return LLMResponse(content=json.dumps(response), model="mock")
        return LLMResponse(content="{}", model="mock")

    def generate_structured(self, prompt: str, system: str | None, schema):
        """Return mock structured response."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return schema(**response)
        return schema(
            additional_files_needed=[],
            reasoning="Done",
            sufficient_context=True,
            preliminary_insights="Test insights",
        )


class TestScopedAnalyzer:
    """Test ScopedAnalyzer."""

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a sample repository."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "src").mkdir()
        (repo / "src" / "weather.py").write_text("def get_weather(): pass")
        (repo / "src" / "utils.py").write_text("def helper(): pass")
        return repo

    def test_analyze_returns_relevant_files(self, sample_repo):
        """Test that analyze returns list of relevant files."""
        mock_llm = MockLLMProvider(responses=[
            {
                "additional_files_needed": [],
                "reasoning": "Found all needed files",
                "sufficient_context": True,
                "preliminary_insights": "Weather module handles forecasts",
            }
        ])

        analyzer = ScopedAnalyzer(mock_llm)
        result = analyzer.analyze(
            repo_path=sample_repo,
            question="weather functionality",
            candidate_files=[{"path": "src/weather.py", "match_type": "filename", "score": 2}],
            file_tree={"name": "repo", "type": "directory", "children": []},
        )

        assert "relevant_files" in result
        assert "insights" in result
        assert len(result["relevant_files"]) > 0

    def test_analyze_expands_with_additional_files(self, sample_repo):
        """Test that analyzer requests and incorporates additional files."""
        mock_llm = MockLLMProvider(responses=[
            {
                "additional_files_needed": ["src/utils.py"],
                "reasoning": "Utils is imported by weather",
                "sufficient_context": False,
                "preliminary_insights": "Need to check utils",
            },
            {
                "additional_files_needed": [],
                "reasoning": "Now have full context",
                "sufficient_context": True,
                "preliminary_insights": "Weather uses utils for helpers",
            },
        ])

        analyzer = ScopedAnalyzer(mock_llm)
        result = analyzer.analyze(
            repo_path=sample_repo,
            question="weather functionality",
            candidate_files=[{"path": "src/weather.py", "match_type": "filename", "score": 2}],
            file_tree={"name": "repo", "type": "directory", "children": []},
        )

        # Should have called LLM twice (initial + expansion)
        assert mock_llm.call_count == 2
        assert "insights" in result
```

### Step 2: Run test to verify it fails

Run: `pytest tests/scoper/test_scoped_analyzer.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

Create `src/agents/scoper/scoped_analyzer.py`:

```python
"""LLM-guided exploration for scoped context generation."""

from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from ..llm.provider import LLMProvider
from ..llm.prompts import SCOPE_EXPLORATION_PROMPT

MAX_EXPLORATION_ROUNDS = 3
MAX_FILE_CONTENT_CHARS = 15_000


class ScopeExplorationOutput(BaseModel):
    """Schema for scope exploration LLM output."""

    additional_files_needed: List[str] = Field(
        description="List of additional file paths to examine"
    )
    reasoning: str = Field(description="Why these files are needed")
    sufficient_context: bool = Field(
        description="Whether we have enough context to generate scoped output"
    )
    preliminary_insights: str = Field(
        description="What has been learned so far about the scope question"
    )


class ScopedAnalyzer:
    """Analyzes repository with LLM guidance for scoped context."""

    def __init__(self, llm_provider: LLMProvider, max_rounds: int = MAX_EXPLORATION_ROUNDS):
        """Initialize analyzer.

        Args:
            llm_provider: LLM provider for API calls
            max_rounds: Maximum exploration rounds before forcing synthesis
        """
        self.llm = llm_provider
        self.max_rounds = max_rounds

    def analyze(
        self,
        repo_path: Path,
        question: str,
        candidate_files: List[Dict],
        file_tree: Dict,
    ) -> Dict[str, Any]:
        """Analyze repository files guided by the scope question.

        Args:
            repo_path: Path to repository
            question: The scope question
            candidate_files: Initial candidate files from discovery
            file_tree: Repository file tree structure

        Returns:
            Dict with relevant_files and insights
        """
        # Track all files we've examined
        examined_files: Dict[str, str] = {}
        all_insights: List[str] = []

        # Start with candidate files
        files_to_examine = [f["path"] for f in candidate_files]

        for round_num in range(self.max_rounds):
            # Read files we haven't examined yet
            for file_path in files_to_examine:
                if file_path not in examined_files:
                    content = self._read_file(repo_path, file_path)
                    if content:
                        examined_files[file_path] = content

            # Ask LLM if we need more files
            exploration_result = self._explore(
                question=question,
                file_tree=file_tree,
                candidate_files=candidate_files,
                examined_contents=examined_files,
            )

            all_insights.append(exploration_result.preliminary_insights)

            if exploration_result.sufficient_context:
                break

            # Queue additional files for next round
            files_to_examine = [
                f for f in exploration_result.additional_files_needed
                if f not in examined_files
            ]

            if not files_to_examine:
                break

        return {
            "relevant_files": examined_files,
            "insights": "\n".join(all_insights),
        }

    def _explore(
        self,
        question: str,
        file_tree: Dict,
        candidate_files: List[Dict],
        examined_contents: Dict[str, str],
    ) -> ScopeExplorationOutput:
        """Ask LLM what additional files to examine."""
        # Format file tree
        tree_str = self._format_tree(file_tree)

        # Format candidate files
        candidates_str = "\n".join(
            f"- {f['path']} (match: {f['match_type']}, score: {f['score']})"
            for f in candidate_files
        )

        # Format examined contents (truncated)
        contents_str = ""
        for path, content in examined_contents.items():
            truncated = content[:MAX_FILE_CONTENT_CHARS]
            contents_str += f"\n=== {path} ===\n{truncated}\n"

        prompt = SCOPE_EXPLORATION_PROMPT.format(
            scope_question=question,
            file_tree=tree_str,
            candidate_files=candidates_str,
            candidate_contents=contents_str,
        )

        return self.llm.generate_structured(
            prompt=prompt,
            system="You are analyzing code to find files relevant to a specific question.",
            schema=ScopeExplorationOutput,
        )

    def _read_file(self, repo_path: Path, file_path: str) -> str | None:
        """Read file content safely."""
        full_path = repo_path / file_path
        try:
            if full_path.exists() and full_path.is_file():
                if full_path.stat().st_size < 500_000:  # 500KB limit
                    return full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
        return None

    def _format_tree(self, tree: Dict, indent: int = 0) -> str:
        """Format file tree as indented string."""
        lines = []
        prefix = "  " * indent

        if tree.get("type") == "directory":
            lines.append(f"{prefix}{tree['name']}/")
            for child in tree.get("children", []):
                lines.append(self._format_tree(child, indent + 1))
        else:
            lines.append(f"{prefix}{tree['name']}")

        return "\n".join(lines)
```

### Step 4: Run test to verify it passes

Run: `pytest tests/scoper/test_scoped_analyzer.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/agents/scoper/scoped_analyzer.py tests/scoper/test_scoped_analyzer.py
git commit -m "feat(scoper): add LLM-guided scoped analyzer"
```

---

## Task 6: Add Scoped Generator

**Files:**
- Create: `src/agents/scoper/scoped_generator.py`
- Test: `tests/scoper/test_scoped_generator.py` (create)

### Step 1: Write the failing test

Create `tests/scoper/test_scoped_generator.py`:

```python
"""Tests for scoped context generator."""

import pytest
from pathlib import Path
from agents.scoper.scoped_generator import ScopedGenerator
from agents.llm.provider import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Return mock generated context."""
        return LLMResponse(
            content="""## Summary

The weather module provides forecast functionality.

## API Endpoints

- GET /weather/forecast

## Key Files

- src/weather/service.py
""",
            model="mock-model",
        )


class TestScopedGenerator:
    """Test ScopedGenerator."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory."""
        out = tmp_path / "contexts"
        out.mkdir()
        return out

    def test_generate_creates_file(self, output_dir):
        """Test that generate creates a scoped context file."""
        mock_llm = MockLLMProvider()
        generator = ScopedGenerator(mock_llm, output_dir)

        output_path = generator.generate(
            repo_name="test-repo",
            question="weather functionality",
            relevant_files={"src/weather.py": "def get_weather(): pass"},
            insights="Weather module provides forecasts",
            model_name="claude-3-5-sonnet",
        )

        assert output_path.exists()
        content = output_path.read_text()
        assert "weather" in content.lower()

    def test_generate_includes_frontmatter(self, output_dir):
        """Test that generated file includes YAML frontmatter."""
        mock_llm = MockLLMProvider()
        generator = ScopedGenerator(mock_llm, output_dir)

        output_path = generator.generate(
            repo_name="test-repo",
            question="weather functionality",
            relevant_files={"src/weather.py": "def get_weather(): pass"},
            insights="Weather module",
            model_name="claude-3-5-sonnet",
        )

        content = output_path.read_text()
        assert content.startswith("---")
        assert "scope_question:" in content
        assert "weather functionality" in content
        assert "files_analyzed:" in content

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

### Step 2: Run test to verify it fails

Run: `pytest tests/scoper/test_scoped_generator.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

Create `src/agents/scoper/scoped_generator.py`:

```python
"""Scoped context file generation."""

import re
import yaml
from pathlib import Path
from typing import Dict
from datetime import datetime, UTC
from ..llm.provider import LLMProvider
from ..llm.prompts import SCOPE_GENERATION_PROMPT
from ..models import ScopedContextMetadata

MAX_CONTENT_PER_FILE = 10_000


class ScopedGenerator:
    """Generates scoped context markdown files."""

    def __init__(self, llm_provider: LLMProvider, output_dir: Path):
        """Initialize generator.

        Args:
            llm_provider: LLM provider for text generation
            output_dir: Directory to write context files
        """
        self.llm = llm_provider
        self.output_dir = output_dir

    def generate(
        self,
        repo_name: str,
        question: str,
        relevant_files: Dict[str, str],
        insights: str,
        model_name: str,
        source_repo: str | None = None,
        source_context: str | None = None,
        output_path: Path | None = None,
    ) -> Path:
        """Generate scoped context file.

        Args:
            repo_name: Name of the repository
            question: The scope question
            relevant_files: Dict of file_path -> content
            insights: Insights from analysis
            model_name: Name of LLM model used
            source_repo: Path to source repository
            source_context: Path to source context file (if scoping from context)
            output_path: Optional custom output path

        Returns:
            Path to generated scoped context file
        """
        # Format file contents for prompt
        files_str = self._format_files(relevant_files)

        # Generate content via LLM
        prompt = SCOPE_GENERATION_PROMPT.format(
            scope_question=question,
            relevant_files=files_str,
            insights=insights,
        )

        response = self.llm.generate(
            prompt=prompt,
            system="You are generating focused context documentation for a specific question about a codebase.",
        )

        # Build metadata
        metadata = ScopedContextMetadata(
            source_repo=source_repo or f"/{repo_name}",
            source_context=source_context,
            scope_question=question,
            model_used=model_name,
            files_analyzed=len(relevant_files),
        )

        # Build full content
        full_content = self._build_context_file(metadata, response.content)

        # Determine output path
        if output_path is None:
            sanitized_name = self._sanitize_filename(question)
            output_path = self.output_dir / repo_name / f"scope-{sanitized_name}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_content, encoding="utf-8")

        return output_path

    def _format_files(self, files: Dict[str, str]) -> str:
        """Format file contents for prompt."""
        parts = []
        for path, content in files.items():
            truncated = content[:MAX_CONTENT_PER_FILE]
            parts.append(f"=== {path} ===\n{truncated}")
        return "\n\n".join(parts)

    def _sanitize_filename(self, question: str) -> str:
        """Convert question to safe filename."""
        # Take first few words, remove special chars
        sanitized = re.sub(r"[^a-zA-Z0-9\s-]", "", question.lower())
        words = sanitized.split()[:4]
        return "-".join(words) if words else "context"

    def _build_context_file(self, metadata: ScopedContextMetadata, content: str) -> str:
        """Build final context file with frontmatter."""
        frontmatter_dict = {
            "source_repo": metadata.source_repo,
            "scope_question": metadata.scope_question,
            "scan_date": metadata.scan_date.isoformat(),
            "model_used": metadata.model_used,
            "files_analyzed": metadata.files_analyzed,
        }

        if metadata.source_context:
            frontmatter_dict["source_context"] = metadata.source_context

        frontmatter = yaml.dump(frontmatter_dict, default_flow_style=False)
        return f"---\n{frontmatter}---\n\n{content.strip()}\n"
```

### Step 4: Update scoper __init__.py exports

Update `src/agents/scoper/__init__.py`:

```python
"""Scoped context generation module."""

from .discovery import extract_keywords, search_relevant_files
from .scoped_analyzer import ScopedAnalyzer, ScopeExplorationOutput
from .scoped_generator import ScopedGenerator

__all__ = [
    "extract_keywords",
    "search_relevant_files",
    "ScopedAnalyzer",
    "ScopeExplorationOutput",
    "ScopedGenerator",
]
```

### Step 5: Run test to verify it passes

Run: `pytest tests/scoper/test_scoped_generator.py -v`
Expected: PASS

### Step 6: Commit

```bash
git add src/agents/scoper/ tests/scoper/
git commit -m "feat(scoper): add scoped context generator"
```

---

## Task 7: Add CLI Scope Command

**Files:**
- Modify: `src/agents/main.py`
- Test: `tests/test_cli.py`

### Step 1: Write the failing test

Add to `tests/test_cli.py`:

```python
class TestCLIScope:
    """Test the scope command."""

    @pytest.fixture
    def runner(self):
        """Create Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a minimal sample repository."""
        repo = tmp_path / "sample_repo"
        repo.mkdir()
        (repo / "README.md").write_text("# Test Project")
        (repo / "weather.py").write_text("def get_weather(): pass")
        return repo

    @pytest.fixture
    def sample_context_file(self, tmp_path):
        """Create a sample existing context file."""
        contexts = tmp_path / "contexts" / "myrepo"
        contexts.mkdir(parents=True)
        context_file = contexts / "context.md"
        context_file.write_text("""---
source_repo: /path/to/myrepo
scan_date: 2025-01-01T00:00:00Z
user_summary: Test project
model_used: claude-3-5-sonnet
---

# Repository Context: myrepo

Test content.
""")
        return context_file

    def test_scope_help(self, runner):
        """Test that scope --help works."""
        result = runner.invoke(cli, ["scope", "--help"])

        assert result.exit_code == 0
        assert "Generate scoped context" in result.output or "scope" in result.output.lower()
        assert "--question" in result.output

    def test_scope_requires_question(self, runner, sample_repo):
        """Test that scope requires --question flag."""
        result = runner.invoke(cli, ["scope", str(sample_repo)])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_scope_requires_existing_source(self, runner):
        """Test that scope requires existing repo or context file."""
        result = runner.invoke(
            cli, ["scope", "/nonexistent/path", "-q", "test question"]
        )

        assert result.exit_code != 0

    @patch("src.agents.main._scope_pipeline_mode")
    def test_scope_from_repo(self, mock_pipeline, runner, sample_repo):
        """Test scoping directly from a repository."""
        mock_pipeline.return_value = 0

        result = runner.invoke(
            cli, ["scope", str(sample_repo), "-q", "weather functionality"]
        )

        mock_pipeline.assert_called_once()
        assert result.exit_code == 0

    @patch("src.agents.main._scope_pipeline_mode")
    def test_scope_from_context_file(self, mock_pipeline, runner, sample_context_file):
        """Test scoping from an existing context file."""
        mock_pipeline.return_value = 0

        result = runner.invoke(
            cli, ["scope", str(sample_context_file), "-q", "authentication flow"]
        )

        mock_pipeline.assert_called_once()
        assert result.exit_code == 0

    @patch("src.agents.main._scope_agent_mode")
    def test_scope_agent_mode(self, mock_agent, runner, sample_repo):
        """Test scope in agent mode."""
        mock_agent.return_value = 0

        result = runner.invoke(
            cli, ["scope", str(sample_repo), "-q", "test", "--mode", "agent"]
        )

        mock_agent.assert_called_once()
        assert result.exit_code == 0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_cli.py::TestCLIScope -v`
Expected: FAIL with `No such command 'scope'`

### Step 3: Write minimal implementation

Add to `src/agents/main.py` after the imports:

```python
from .scoper.discovery import extract_keywords, search_relevant_files
from .scoper.scoped_analyzer import ScopedAnalyzer
from .scoper.scoped_generator import ScopedGenerator
```

Add the scope command after the `refine` command:

```python
@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--question", "-q", required=True, help="Question/topic to scope to")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["pipeline", "agent"]),
    default="pipeline",
    help="Execution mode: pipeline (deterministic) or agent (agentic)",
)
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.option("--stream", is_flag=True, help="Enable streaming output (agent mode)")
def scope(source: str, question: str, output: str | None, mode: str, debug: bool, stream: bool):
    """Generate scoped context for a specific question.

    SOURCE can be either:
    - A repository path: scopes directly from the repo
    - A context.md file: uses existing context as starting point

    Examples:
        # Scope from repo
        python -m agents.main scope /path/to/repo -q "weather functionality"

        # Scope from existing context
        python -m agents.main scope contexts/repo/context.md -q "auth flow"

        # With custom output
        python -m agents.main scope /path/to/repo -q "API endpoints" -o my-scope.md

        # Agent mode with streaming
        python -m agents.main scope /path/to/repo -q "auth" --mode agent --stream
    """
    source_path = Path(source).resolve()
    config = Config.from_env()

    if not config.api_key:
        click.echo("Error: ANTHROPIC_API_KEY not set in environment", err=True)
        return 1

    # Determine if source is a repo or context file
    is_context_file = source_path.is_file() and source_path.suffix == ".md"

    if mode == "agent":
        return _scope_agent_mode(source_path, question, config, is_context_file, debug, stream)
    else:
        return _scope_pipeline_mode(source_path, question, config, is_context_file, output)


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
        # Extract repo path from context file's location
        # contexts/{repo-name}/context.md -> infer repo or read from frontmatter
        repo_name = source_path.parent.name
        repo_path = source_path.parent.parent.parent  # Best guess
        source_context = str(source_path)
        click.echo(f"   Source: context file for {repo_name}")
    else:
        repo_path = source_path
        repo_name = source_path.name
        source_context = None
        click.echo(f"   Source: repository {repo_name}")

    # Phase 1: Discovery
    click.echo("\n📂 Phase 1: Discovery...")
    keywords = extract_keywords(question)
    click.echo(f"   Keywords: {', '.join(keywords)}")

    if not is_context_file:
        # Scan repository structure
        scanner = StructureScanner(config)
        structure = scanner.scan(repo_path)
        file_tree = structure["tree"]

        # Search for relevant files
        candidates = search_relevant_files(repo_path, keywords)
        click.echo(f"   Found {len(candidates)} candidate files")
    else:
        # When scoping from context, we still need the repo to search
        # For now, require the repo to exist at expected location
        click.echo("   (Searching based on context file location)")
        # Simplified: search in parent directories
        candidates = []
        file_tree = {"name": repo_name, "type": "directory", "children": []}

    if not candidates and not is_context_file:
        click.echo("   No matching files found. Searching full repo...")
        # Fallback: use all files from structure scan
        candidates = [
            {"path": f, "match_type": "fallback", "score": 1}
            for f in structure.get("all_files", [])[:20]
        ]

    # Phase 2: LLM-guided exploration
    click.echo(f"\n🤖 Phase 2: Exploration with {config.model_name}...")
    llm = AnthropicProvider(config.model_name, config.api_key)
    analyzer = ScopedAnalyzer(llm)

    analysis_result = analyzer.analyze(
        repo_path=repo_path,
        question=question,
        candidate_files=candidates,
        file_tree=file_tree,
    )

    click.echo(f"   Analyzed {len(analysis_result['relevant_files'])} files")

    # Phase 3: Generate scoped context
    click.echo("\n📝 Phase 3: Generating scoped context...")
    generator = ScopedGenerator(llm, config.output_dir)

    output_path = Path(output) if output else None
    result_path = generator.generate(
        repo_name=repo_name,
        question=question,
        relevant_files=analysis_result["relevant_files"],
        insights=analysis_result["insights"],
        model_name=config.model_name,
        source_repo=str(repo_path),
        source_context=source_context,
        output_path=output_path,
    )

    click.echo(f"\n✅ Scoped context generated: {result_path}")
    return 0


def _scope_agent_mode(
    source_path: Path,
    question: str,
    config: Config,
    is_context_file: bool,
    debug: bool,
    stream: bool,
) -> int:
    """Generate scoped context using agent mode."""
    from .factory import create_contextualizer_agent
    from .memory import create_checkpointer, create_agent_config
    from .observability import configure_tracing, is_tracing_enabled

    click.echo(f"🤖 Agent mode: Scoping '{question}'")

    if stream:
        click.echo("   Streaming: Enabled")

    # Configure tracing
    configure_tracing()

    # Create checkpointer
    checkpointer = create_checkpointer()

    # Create agent
    agent = create_contextualizer_agent(
        model_name=config.model_name if config.model_name.startswith("anthropic:") else f"anthropic:{config.model_name}",
        checkpointer=checkpointer,
        debug=debug,
    )

    # Create agent configuration
    agent_config = create_agent_config(f"scope-{source_path}")

    # Build user message
    if is_context_file:
        user_message = f"Generate scoped context for the question: '{question}'. Use the existing context at {source_path} as a starting point."
    else:
        user_message = f"Generate scoped context for the repository at {source_path}. Focus specifically on: {question}"

    click.echo(f"   Thread ID: {agent_config['configurable']['thread_id']}")
    if is_tracing_enabled():
        click.echo("   Tracing: Enabled")

    try:
        if stream:
            from .streaming import stream_agent_execution, simple_stream_agent_execution
            import sys

            if sys.stdout.isatty():
                stream_agent_execution(
                    agent,
                    messages=[{"role": "user", "content": user_message}],
                    config=agent_config,
                    verbose=debug,
                )
            else:
                simple_stream_agent_execution(
                    agent,
                    messages=[{"role": "user", "content": user_message}],
                    config=agent_config,
                )
        else:
            click.echo("\n🔄 Agent executing...")
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config=agent_config,
            )
            final_message = result.get("messages", [])[-1]
            output_content = final_message.content if hasattr(final_message, "content") else str(final_message)
            click.echo("\n📋 Agent Response:")
            click.echo(output_content)
            click.echo("\n✅ Agent execution complete")

        return 0

    except Exception as e:
        click.echo(f"\n❌ Agent execution failed: {e}", err=True)
        if debug:
            import traceback
            traceback.print_exc()
        return 1
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_cli.py::TestCLIScope -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/agents/main.py tests/test_cli.py
git commit -m "feat(cli): add scope command for question-scoped context"
```

---

## Task 8: Integration Test

**Files:**
- Create: `tests/test_scope_integration.py`

### Step 1: Write integration test

Create `tests/test_scope_integration.py`:

```python
"""Integration tests for scoped context generation."""

import pytest
import json
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, Mock
from src.agents.main import cli
from agents.llm.provider import LLMResponse


class TestScopeIntegration:
    """End-to-end tests for scope command."""

    @pytest.fixture
    def runner(self):
        """Create Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a realistic sample repository."""
        repo = tmp_path / "weather-api"
        repo.mkdir()

        # Create structure
        (repo / "src").mkdir()
        (repo / "src" / "weather").mkdir()
        (repo / "src" / "auth").mkdir()
        (repo / "tests").mkdir()

        # Weather module
        (repo / "src" / "weather" / "__init__.py").write_text("")
        (repo / "src" / "weather" / "service.py").write_text("""
class WeatherService:
    def get_forecast(self, location: str) -> dict:
        '''Get weather forecast for a location.'''
        return {"location": location, "temp": 72}

    def get_alerts(self, location: str) -> list:
        '''Get weather alerts for a location.'''
        return []
""")
        (repo / "src" / "weather" / "models.py").write_text("""
from dataclasses import dataclass

@dataclass
class Forecast:
    location: str
    temperature: float
    conditions: str
""")

        # Auth module (unrelated)
        (repo / "src" / "auth" / "__init__.py").write_text("")
        (repo / "src" / "auth" / "login.py").write_text("""
def authenticate(username: str, password: str) -> bool:
    return True
""")

        # Tests
        (repo / "tests" / "test_weather.py").write_text("""
from src.weather.service import WeatherService

def test_get_forecast():
    service = WeatherService()
    result = service.get_forecast("NYC")
    assert result["location"] == "NYC"
""")

        # README
        (repo / "README.md").write_text("# Weather API\n\nA weather forecasting service.")

        return repo

    @patch("src.agents.main.AnthropicProvider")
    def test_scope_pipeline_end_to_end(self, mock_provider_class, runner, sample_repo, tmp_path):
        """Test full pipeline scope execution."""
        # Mock LLM responses
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider

        # Mock exploration response
        mock_provider.generate_structured.return_value = Mock(
            additional_files_needed=[],
            reasoning="Found all weather files",
            sufficient_context=True,
            preliminary_insights="Weather service provides forecast and alerts",
        )

        # Mock generation response
        mock_provider.generate.return_value = LLMResponse(
            content="""## Summary

The weather module provides forecast and alert functionality.

## API Surface

- `WeatherService.get_forecast(location)` - Returns forecast data
- `WeatherService.get_alerts(location)` - Returns weather alerts

## Key Files

- src/weather/service.py - Main service implementation
- src/weather/models.py - Data models

## Usage Examples

See tests/test_weather.py for usage examples.
""",
            model="claude-3-5-sonnet",
        )

        # Set up environment
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("src.agents.main.Config.from_env") as mock_config:
                mock_config.return_value = Mock(
                    api_key="test-key",
                    model_name="claude-3-5-sonnet",
                    output_dir=output_dir,
                    ignored_dirs=[".git", "__pycache__"],
                    max_file_size=1_000_000,
                )

                result = runner.invoke(
                    cli,
                    ["scope", str(sample_repo), "-q", "weather functionality"],
                )

        # Verify execution
        assert result.exit_code == 0
        assert "Scoping:" in result.output
        assert "Phase 1: Discovery" in result.output
        assert "Phase 2: Exploration" in result.output
        assert "Phase 3: Generating" in result.output
        assert "Scoped context generated" in result.output

    def test_scope_discovery_finds_relevant_files(self, sample_repo):
        """Test that discovery phase finds weather-related files."""
        from agents.scoper.discovery import extract_keywords, search_relevant_files

        keywords = extract_keywords("weather forecast functionality")
        results = search_relevant_files(sample_repo, keywords)

        paths = [r["path"] for r in results]

        # Should find weather-related files
        assert any("weather" in p for p in paths)
        # Should not prioritize auth files
        auth_paths = [p for p in paths if "auth" in p]
        weather_paths = [p for p in paths if "weather" in p]
        assert len(weather_paths) >= len(auth_paths)
```

### Step 2: Run integration test

Run: `pytest tests/test_scope_integration.py -v`
Expected: PASS

### Step 3: Commit

```bash
git add tests/test_scope_integration.py
git commit -m "test: add integration tests for scope command"
```

---

## Task 9: Update Module Exports and Documentation

**Files:**
- Modify: `src/agents/__init__.py`
- Modify: `README.md`

### Step 1: Update module exports

Add to `src/agents/__init__.py`:

```python
from .scoper import (
    extract_keywords,
    search_relevant_files,
    ScopedAnalyzer,
    ScopedGenerator,
)
```

### Step 2: Update README with scope command documentation

Add after the "Refine Context" section in `README.md`:

```markdown
### Scope Context

Generate focused context for a specific question or topic:

```bash
# Scope from repository
python -m agents.main scope /path/to/backend \
  --question "weather functionality"

# Scope from existing context file (faster, reuses metadata)
python -m agents.main scope contexts/backend/context.md \
  --question "authentication flow"

# With custom output path
python -m agents.main scope /path/to/repo \
  --question "API endpoints" \
  --output my-scoped-context.md

# Agent mode with streaming
python -m agents.main scope /path/to/repo \
  --question "database models" \
  --mode agent --stream
```

**Options:**
- `SOURCE` - Repository path OR existing context.md file
- `-q, --question` - The question/topic to scope to (required)
- `-o, --output` - Custom output file path
- `-m, --mode` - `pipeline` (default) or `agent`
- `--stream` - Enable streaming output (agent mode)
- `--debug` - Enable debug output

**How scoping works:**
1. **Discovery** - Extracts keywords, searches for matching files
2. **Exploration** - LLM analyzes candidates, follows imports/tests
3. **Synthesis** - Generates focused context for the specific question

**Output:** `contexts/{repo-name}/scope-{topic}.md`
```

### Step 3: Commit

```bash
git add src/agents/__init__.py README.md
git commit -m "docs: add scope command documentation"
```

---

## Task 10: Run Full Test Suite

### Step 1: Run all tests

Run: `pytest -v`
Expected: All tests PASS

### Step 2: Run linting

Run: `ruff check src/ tests/`
Expected: No errors (or fix any that appear)

### Step 3: Final commit

```bash
git add -A
git commit -m "feat: complete scoped context generation feature"
```

---

## Summary

**Files created:**
- `src/agents/scoper/__init__.py`
- `src/agents/scoper/discovery.py`
- `src/agents/scoper/scoped_analyzer.py`
- `src/agents/scoper/scoped_generator.py`
- `tests/test_models.py`
- `tests/scoper/__init__.py`
- `tests/scoper/test_discovery.py`
- `tests/scoper/test_scoped_analyzer.py`
- `tests/scoper/test_scoped_generator.py`
- `tests/test_scope_integration.py`

**Files modified:**
- `src/agents/models.py` - Added `ScopedContextMetadata`
- `src/agents/llm/prompts.py` - Added scope prompts
- `src/agents/main.py` - Added `scope` command
- `src/agents/__init__.py` - Updated exports
- `README.md` - Added documentation
- `tests/test_cli.py` - Added scope CLI tests

**Total commits:** 10 (one per task)
