"""Project metadata extraction."""

import json
import tomllib
from pathlib import Path
from typing import Optional
from ..models import ProjectMetadata


class MetadataExtractor:
    """Extracts project metadata from configuration files."""

    def extract(self, repo_path: Path) -> ProjectMetadata:
        """Extract metadata from repository.

        Args:
            repo_path: Path to repository root

        Returns:
            ProjectMetadata with extracted information
        """
        name = repo_path.name
        project_type = self._detect_project_type(repo_path)
        dependencies = self._extract_dependencies(repo_path, project_type)
        entry_points = self._find_entry_points(repo_path, project_type)
        key_files = self._identify_key_files(repo_path)
        readme_content = self._read_readme(repo_path)

        return ProjectMetadata(
            name=name,
            path=repo_path,
            project_type=project_type,
            dependencies=dependencies,
            entry_points=entry_points,
            key_files=key_files,
            readme_content=readme_content,
        )

    def _detect_project_type(self, repo_path: Path) -> Optional[str]:
        """Detect project type from files present.

        Args:
            repo_path: Repository root

        Returns:
            Project type string or None
        """
        if (repo_path / "package.json").exists():
            return "node"
        elif (repo_path / "pyproject.toml").exists() or (repo_path / "setup.py").exists():
            return "python"
        elif (repo_path / "Cargo.toml").exists():
            return "rust"
        elif (repo_path / "go.mod").exists():
            return "go"
        elif (repo_path / "pom.xml").exists():
            return "java"
        return None

    def _extract_dependencies(self, repo_path: Path, project_type: Optional[str]) -> dict:
        """Extract dependencies based on project type.

        Args:
            repo_path: Repository root
            project_type: Detected project type

        Returns:
            Dictionary of dependency name -> version
        """
        deps = {}

        if project_type == "node":
            package_json = repo_path / "package.json"
            if package_json.exists():
                try:
                    data = json.loads(package_json.read_text())
                    deps.update(data.get("dependencies", {}))
                    deps.update(data.get("devDependencies", {}))
                except (json.JSONDecodeError, OSError):
                    pass

        elif project_type == "python":
            pyproject = repo_path / "pyproject.toml"
            if pyproject.exists():
                try:
                    data = tomllib.loads(pyproject.read_text())
                    project = data.get("project", {})
                    for dep in project.get("dependencies", []):
                        # Parse "package>=1.0.0" format
                        parts = dep.replace(">=", "==").replace("~=", "==").split("==")
                        deps[parts[0]] = parts[1] if len(parts) > 1 else "*"
                except (OSError, tomllib.TOMLDecodeError):
                    pass

        return deps

    def _find_entry_points(self, repo_path: Path, project_type: Optional[str]) -> list[str]:
        """Find likely entry points.

        Args:
            repo_path: Repository root
            project_type: Detected project type

        Returns:
            List of entry point file paths (relative to repo)
        """
        entry_points = []

        common_entries = [
            "main.py",
            "app.py",
            "__main__.py",
            "index.js",
            "server.js",
            "app.js",
            "main.go",
            "main.rs",
        ]

        for entry in common_entries:
            if (repo_path / entry).exists():
                entry_points.append(entry)

        # Check src/ directory
        src_dir = repo_path / "src"
        if src_dir.exists():
            for entry in common_entries:
                if (src_dir / entry).exists():
                    entry_points.append(f"src/{entry}")

        # Check package.json main field
        if project_type == "node":
            package_json = repo_path / "package.json"
            if package_json.exists():
                try:
                    data = json.loads(package_json.read_text())
                    if "main" in data:
                        entry_points.append(data["main"])
                except (json.JSONDecodeError, OSError):
                    pass

        return list(set(entry_points))  # Remove duplicates

    def _identify_key_files(self, repo_path: Path) -> list[str]:
        """Identify important configuration and documentation files.

        Args:
            repo_path: Repository root

        Returns:
            List of key file paths (relative to repo)
        """
        key_patterns = [
            "README.md",
            "README.rst",
            "README.txt",
            "package.json",
            "pyproject.toml",
            "setup.py",
            "Cargo.toml",
            "go.mod",
            "Dockerfile",
            "docker-compose.yml",
            "Makefile",
            ".env.example",
            "tsconfig.json",
            "jest.config.js",
            "pytest.ini",
        ]

        key_files = []
        for pattern in key_patterns:
            file_path = repo_path / pattern
            if file_path.exists():
                key_files.append(pattern)

        return key_files

    def _read_readme(self, repo_path: Path) -> Optional[str]:
        """Read README content if present.

        Args:
            repo_path: Repository root

        Returns:
            README content or None
        """
        for readme_name in ["README.md", "README.rst", "README.txt", "README"]:
            readme_path = repo_path / readme_name
            if readme_path.exists():
                try:
                    return readme_path.read_text(encoding="utf-8")
                except OSError:
                    pass
        return None
