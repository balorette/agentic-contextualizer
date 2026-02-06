# Git Pull Tool Design

**Date:** 2026-02-06
**Branch:** `feat/git-pull-tool`
**Status:** Approved

## Summary

Add support for GitHub URLs as input to the `generate` and `scope` CLI commands. When a user provides a GitHub URL instead of a local path, the tool shallow-clones the repo into a temp directory, runs the existing pipeline, then cleans up.

## Goals

- Users can run `python -m agents.main generate https://github.com/owner/repo` directly
- No changes to the core scanning/analysis/generation pipeline
- Cleanup is guaranteed even on errors
- Ambient git credentials are used (no auth management in this tool)

## Architecture

```
User input (URL or path)
        |
        v
   RepoResolver          <-- NEW: detects URL vs path, clones if needed
        |
        v
   Temp local path
        |
        v
   Existing pipeline      <-- UNCHANGED
        |
        v
   Context output
        |
        v
   Cleanup temp dir       <-- NEW: if cloned, remove temp dir
```

### New Components

- `src/agents/repo_resolver.py` - URL detection, shallow cloning, cleanup via context manager

### Modified Components

- `src/agents/main.py` - CLI commands wrap pipeline calls with `resolve_repo()`. The `repo_path` argument becomes `source` (no longer validated as `exists=True`).

### Unchanged

Scanner, analyzer, generator, scoper, backends, tools, config, models.

## Implementation Details

### `repo_resolver.py`

**URL Detection:**
- Regex pattern: `^https?://github\.com/[\w.\-]+/[\w.\-]+(\.git)?/?$`
- `is_github_url(source)` returns bool
- `extract_repo_name(url)` parses repo name for output path

**Clone:**
- `git clone --depth 1` (shallow, fast)
- `subprocess.run` with `capture_output=True`, `timeout=120`
- Clones into `tempfile.mkdtemp(prefix="ctx-")`

**Resolve (context manager):**
```python
@contextmanager
def resolve_repo(source: str):
    if is_github_url(source):
        tmp_dir = tempfile.mkdtemp(prefix="ctx-")
        try:
            clone_repo(source, tmp_dir)
            yield Path(tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        path = Path(source)
        if not path.exists():
            raise click.BadParameter(f"Path does not exist: {source}")
        yield path
```

### CLI Changes

**Before:**
```python
@click.argument("repo_path", type=click.Path(exists=True))
def generate(repo_path, ...):
```

**After:**
```python
@click.argument("source")
def generate(source, ...):
    with resolve_repo(source) as repo:
        # existing pipeline, unchanged
```

Same pattern for `scope` when source is a repo (not a context file).

**Error messages:**
- Clone failure: `"Failed to clone repository. Check the URL and your git credentials."`
- Invalid source: `"Source is not a valid local path or GitHub URL."`

**Output path:** `contexts/{repo-name}/context.md` using name extracted from URL.

### Authentication

No auth management. Relies on the user's existing git configuration (SSH keys, credential helpers, `gh auth`). Public repos work out of the box.

## Testing

**New file:** `tests/test_repo_resolver.py`

All clone tests mock `subprocess.run` - no network calls.

| Test | Validates |
|------|-----------|
| `test_is_github_url` | Recognizes valid GitHub URLs |
| `test_is_not_github_url` | Rejects non-GitHub URLs, local paths |
| `test_extract_repo_name` | Parses repo name from URL variants |
| `test_resolve_repo_local_path` | Passes through local paths, no temp dir |
| `test_resolve_repo_local_path_not_found` | Raises error for missing path |
| `test_resolve_repo_clones_url` | Calls `git clone --depth 1`, yields path, cleans up |
| `test_resolve_repo_cleanup_on_error` | Temp dir removed even if pipeline fails |
| `test_clone_timeout` | Timeout enforced on hanging clone |
| `test_clone_failure_bad_url` | Clear error on failed clone |

No changes to existing tests.

## Design Decisions

1. **GitHub-only (for now)** - Simplest case, covers most users. Expand to generic git URLs later if needed.
2. **Shallow clone** - `--depth 1` avoids downloading full history. The tool only needs current file state.
3. **Ephemeral temp dirs** - No caching, no disk bloat. Users who want repeated access should clone locally.
4. **Context manager pattern** - Guarantees cleanup via `finally` block.
5. **Ambient credentials** - Zero auth complexity. Leverages user's existing git setup.
