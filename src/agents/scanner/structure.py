"""Repository structure scanner."""

import os
from pathlib import Path
from typing import List, Set, Tuple
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPattern
from ..config import Config


MAX_TREE_DEPTH = 6
MAX_CHILDREN_PER_DIR = 200


class StructureScanner:
    """Scans repository structure and builds file tree."""

    def __init__(self, config: Config):
        """Initialize scanner with configuration.

        Args:
            config: Application configuration with ignored_dirs, max_file_size
        """
        self.config = config
        self.ignored_dirs: Set[str] = set(config.ignored_dirs)

    def scan(self, repo_path: Path) -> dict:
        """Scan repository and return structured file tree.

        Args:
            repo_path: Path to repository root

        Returns:
            Dictionary with:
            - 'tree': Nested dict representing directory structure
            - 'all_files': Flat list of all file paths (relative to repo)
            - 'total_files': Count of files
            - 'total_dirs': Count of directories
        """
        gitignore_spec = self._load_gitignore(repo_path)
        tree = self._build_tree(repo_path, repo_path, gitignore_spec)
        all_files, total_dirs = self._collect_all_files(repo_path, gitignore_spec)

        return {
            "tree": tree,
            "all_files": all_files,
            "total_files": len(all_files),
            "total_dirs": total_dirs,
        }

    def _build_tree(
        self, current_path: Path, repo_root: Path, gitignore_spec: PathSpec | None, depth: int = 0
    ) -> dict:
        """Recursively build directory tree.

        Args:
            current_path: Current directory being scanned
            repo_root: Repository root for relative path calculation

        Returns:
            Dictionary representing directory structure
        """
        tree = {"name": current_path.name, "type": "directory", "children": []}

        if depth >= MAX_TREE_DEPTH:
            tree["children"].append(
                {
                    "name": "...",
                    "type": "truncated",
                    "reason": "max depth reached",
                }
            )
            return tree

        try:
            entries = sorted(current_path.iterdir(), key=lambda p: (not p.is_file(), p.name.lower()))
        except PermissionError:
            return tree  # Skip directories we can't access

        child_count = 0
        for item in entries:
            if self._should_ignore(item, repo_root, gitignore_spec):
                continue

            if item.is_dir():
                tree["children"].append(
                    self._build_tree(item, repo_root, gitignore_spec, depth + 1)
                )
            else:
                try:
                    size = item.stat().st_size
                except OSError:
                    size = 0

                tree["children"].append(
                    {
                        "name": item.name,
                        "type": "file",
                        "size": size,
                        "path": str(item.relative_to(repo_root)),
                    }
                )

            child_count += 1
            if child_count >= MAX_CHILDREN_PER_DIR:
                tree["children"].append(
                    {
                        "name": "...",
                        "type": "truncated",
                        "reason": "max children reached",
                    }
                )
                break

        return tree

    def _collect_all_files(self, repo_path: Path, gitignore_spec: PathSpec | None) -> Tuple[List[str], int]:
        """Collect flat list of all file paths.

        Args:
            repo_path: Repository root
            gitignore_spec: Optional PathSpec for .gitignore patterns

        Returns:
            Tuple of (list of relative file paths, total directory count)
        """
        files: List[str] = []
        total_dirs = 0

        for root, dirs, filenames in os.walk(repo_path):
            root_path = Path(root)

            # Prune directories before descending further
            pruned_dirs = []
            for directory in dirs:
                candidate = root_path / directory
                if not self._should_ignore(candidate, repo_path, gitignore_spec):
                    pruned_dirs.append(directory)
            dirs[:] = pruned_dirs

            total_dirs += 1

            for filename in filenames:
                file_path = root_path / filename
                if self._should_ignore(file_path, repo_path, gitignore_spec):
                    continue
                files.append(str(file_path.relative_to(repo_path)))

        return sorted(files), total_dirs

    def _should_ignore(
        self, path: Path, repo_root: Path, gitignore_spec: PathSpec | None = None
    ) -> bool:
        """Check if path should be ignored.

        Args:
            path: Path to check
            repo_root: Repository root

        Returns:
            True if path should be ignored
        """
        # Check if any part of the path is in ignored_dirs
        rel_path = path.relative_to(repo_root)
        for part in rel_path.parts:
            if part in self.ignored_dirs:
                return True

        # Respect .gitignore patterns when available
        if gitignore_spec and gitignore_spec.match_file(rel_path.as_posix()):
            return True

        # Check file size limit for files
        if path.is_file():
            try:
                if path.stat().st_size > self.config.max_file_size:
                    return True
            except OSError:
                return True

        return False

    def _load_gitignore(self, repo_root: Path) -> PathSpec | None:
        """Load .gitignore patterns if present."""
        gitignore_path = repo_root / ".gitignore"
        if not gitignore_path.exists():
            return None

        try:
            patterns = gitignore_path.read_text().splitlines()
        except OSError:
            return None

        if not patterns:
            return None

        return PathSpec.from_lines(GitWildMatchPattern, patterns)
