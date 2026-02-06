"""Discovery phase for scoped context generation."""

import os
import re
from pathlib import Path
from typing import List, Set, Dict

# Common English stopwords plus domain-specific terms to filter from keyword extraction.
# Based on standard NLP stopword lists with additions for code-related queries
# (e.g., "functionality", "feature", "work" which appear in questions but aren't searchable).
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
    max_results: int = 20,
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
                    # Skip files that can't be read (permissions, encoding issues)
                    # These are non-critical - we simply exclude them from search results
                    pass

    # Sort by score descending, limit results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def search_with_line_numbers(
    repo_path: Path,
    keywords: List[str],
    max_results: int = 20,
    max_lines_per_file: int = 5,
) -> List[Dict]:
    """Search for files and return specific line matches with line numbers.

    Enhanced version of search_relevant_files that captures which lines
    contain keyword matches for content matches.

    Args:
        repo_path: Path to repository root
        keywords: List of keywords to search for
        max_results: Maximum number of file results to return
        max_lines_per_file: Maximum line matches to return per file

    Returns:
        List of dicts with: path, match_type, score, line_matches
        where line_matches is list of {line_num, content}
    """
    results: List[Dict] = []
    keyword_set = set(kw.lower() for kw in keywords)

    for root, dirs, files in os.walk(repo_path):
        # Prune ignored directories
        dirs[:] = [
            d for d in dirs if d not in IGNORED_DIRS and not d.endswith(".egg-info")
        ]

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
                results.append(
                    {
                        "path": rel_path,
                        "match_type": "filename",
                        "score": name_matches * 2 + path_matches,
                        "line_matches": [],  # No line matches for filename matches
                    }
                )
                continue

            # Check content match for searchable files
            suffix = file_path.suffix.lower()
            if suffix in SEARCHABLE_EXTENSIONS:
                try:
                    if file_path.stat().st_size > 500_000:  # Skip large files
                        continue
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    content_lower = content.lower()

                    content_matches = sum(1 for kw in keyword_set if kw in content_lower)
                    if content_matches > 0:
                        # Find specific line matches
                        line_matches = _find_line_matches(
                            content, keyword_set, max_lines_per_file
                        )
                        results.append(
                            {
                                "path": rel_path,
                                "match_type": "content",
                                "score": content_matches,
                                "line_matches": line_matches,
                            }
                        )
                except (OSError, UnicodeDecodeError):
                    # Skip files that can't be read (permissions, encoding);
                    # non-critical for search — we simply exclude them from results
                    pass

    # Sort by score descending, limit results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def _find_line_matches(
    content: str,
    keywords: Set[str],
    max_lines: int,
) -> List[Dict]:
    """Find lines containing keyword matches.

    Args:
        content: File content
        keywords: Set of lowercase keywords to find
        max_lines: Maximum number of line matches to return

    Returns:
        List of dicts with line_num and content
    """
    matches: List[Dict] = []
    lines = content.splitlines()

    for i, line in enumerate(lines):
        if len(matches) >= max_lines:
            break

        line_lower = line.lower()
        if any(kw in line_lower for kw in keywords):
            matches.append(
                {
                    "line_num": i + 1,  # 1-indexed
                    "content": line.strip()[:200],  # Limit line length
                }
            )

    return matches
