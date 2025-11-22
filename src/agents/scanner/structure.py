"""Repository structure scanner."""

from pathlib import Path
from typing import List, Set
from ..config import Config


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
        tree = self._build_tree(repo_path, repo_path)
        all_files = self._collect_all_files(repo_path)

        return {
            'tree': tree,
            'all_files': all_files,
            'total_files': len(all_files),
            'total_dirs': len([f for f in repo_path.rglob('*') if f.is_dir() and not self._should_ignore(f, repo_path)])
        }

    def _build_tree(self, current_path: Path, repo_root: Path) -> dict:
        """Recursively build directory tree.

        Args:
            current_path: Current directory being scanned
            repo_root: Repository root for relative path calculation

        Returns:
            Dictionary representing directory structure
        """
        tree = {'name': current_path.name, 'type': 'directory', 'children': []}

        try:
            for item in sorted(current_path.iterdir()):
                if self._should_ignore(item, repo_root):
                    continue

                if item.is_dir():
                    tree['children'].append(self._build_tree(item, repo_root))
                else:
                    tree['children'].append({
                        'name': item.name,
                        'type': 'file',
                        'size': item.stat().st_size,
                        'path': str(item.relative_to(repo_root))
                    })
        except PermissionError:
            pass  # Skip directories we can't access

        return tree

    def _collect_all_files(self, repo_path: Path) -> List[str]:
        """Collect flat list of all file paths.

        Args:
            repo_path: Repository root

        Returns:
            List of relative file paths as strings
        """
        files = []
        for item in repo_path.rglob('*'):
            if item.is_file() and not self._should_ignore(item, repo_path):
                files.append(str(item.relative_to(repo_path)))
        return sorted(files)

    def _should_ignore(self, path: Path, repo_root: Path) -> bool:
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

        # Check file size limit for files
        if path.is_file():
            try:
                if path.stat().st_size > self.config.max_file_size:
                    return True
            except OSError:
                return True

        return False
