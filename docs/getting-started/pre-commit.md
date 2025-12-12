# Pre-commit Hooks

Pre-commit hooks help maintain code quality by automatically checking your code before each commit. This project uses pre-commit to enforce consistent formatting, catch common errors, and maintain security best practices.

## What are Pre-commit Hooks?

Pre-commit hooks are automated checks that run before you create a Git commit. They:

- **Format code automatically** (black, ruff)
- **Catch bugs and style issues** (mypy, ruff linting)
- **Detect security vulnerabilities** (bandit)
- **Prevent common mistakes** (large files, private keys, merge conflicts)
- **Ensure consistent code style** across the team

## Setup

Pre-commit is included in the dev dependencies and can be installed with:

```bash
# Install development dependencies (includes pre-commit)
uv pip install -e ".[dev]"

# Install the git hooks
pre-commit install
```

## What Hooks Are Enabled?

The project uses the following hooks (configured in `.pre-commit-config.yaml`):

### 1. General File Checks

- **Trailing whitespace removal** - Removes unnecessary whitespace at end of lines
- **End-of-file fixer** - Ensures files end with a newline
- **YAML/JSON/TOML validation** - Catches syntax errors in config files
- **Large file detection** - Prevents accidentally committing files >1MB
- **Private key detection** - Catches accidentally committed secrets
- **Merge conflict detection** - Ensures you don't commit merge markers

### 2. Python Code Quality

**Black** (code formatter)

- Automatically formats Python code to consistent style
- Line length: 100 characters
- Skips notebooks (`.ipynb` files)

**Ruff** (linter and import sorter)

- Fast Python linter (replaces flake8, isort, and more)
- Auto-fixes issues when possible
- Checks for code smells, unused imports, style violations
- Organizes imports alphabetically
- Also runs ruff-format for consistent code style

**docformatter** (docstring formatter)

- Automatically formats Python docstrings
- Enforces blank lines before lists in docstrings (fixes MkDocs rendering)
- Wraps long summary and description lines
- Ensures consistent Google/NumPy-style formatting

### 3. Documentation

**mdformat** (markdown formatter)

- Automatically formats markdown files for consistency
- Enforces blank lines around lists (prevents rendering issues)
- Supports GitHub Flavored Markdown (GFM)
- Handles frontmatter in documentation files
- Preserves line wrapping (--wrap no)

## Usage

### Automatic (Recommended)

Once installed, pre-commit runs automatically before each `git commit`:

```bash
# Make changes to files
vim src/data/make_dataset.py

# Stage changes
git add src/data/make_dataset.py

# Commit - pre-commit hooks run automatically
git commit -m "feat: improve data loading performance"
```

If any hooks fail, the commit is aborted. Fix the issues and try again:

```bash
# Fix issues (many are auto-fixed)
git add .  # Re-stage auto-fixed files

# Try commit again
git commit -m "feat: improve data loading performance"
```

### Manual Run

You can run pre-commit manually on staged files:

```bash
# Run all hooks on staged files
pre-commit run

# Run all hooks on all files in repo
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run ruff --all-files
pre-commit run mypy --all-files
```

### Skip Hooks (Not Recommended)

In rare cases, you can skip hooks with:

```bash
# Skip all hooks for this commit (use sparingly!)
git commit --no-verify -m "WIP: temporary commit"

# Better: Skip specific hooks with SKIP environment variable
SKIP=mypy git commit -m "feat: add new module (type hints TODO)"
```

**Warning:** Only skip hooks when absolutely necessary and plan to fix issues before merging.

## Common Workflows

### First-Time Setup

```bash
# 1. Install dev dependencies
uv pip install -e ".[dev]"

# 2. Install pre-commit hooks
pre-commit install

# 3. (Optional) Run on all files to fix existing issues
pre-commit run --all-files

# 4. Commit the fixes
git add .
git commit -m "chore: apply pre-commit auto-fixes"
```

### Daily Development

```bash
# 1. Make changes
vim src/models/train_advanced.py

# 2. Stage changes
git add src/models/train_advanced.py

# 3. Commit (hooks run automatically)
git commit -m "feat: add Optuna hyperparameter tuning"

# If hooks fail:
# - Review the error messages
# - Most issues are auto-fixed (black, ruff, trailing whitespace)
# - For auto-fixes: git add . and commit again
# - For other issues: manually fix and retry
```

### Working with Notebooks

Notebooks are handled specially:

- **nbstripout** cleans output cells before committing
- **Black and ruff** are skipped for `.ipynb` files
- This keeps notebooks clean in Git while preserving local outputs

```bash
# After working in notebooks
git add notebooks/01-eda-and-cleaning.ipynb

# Commit - nbstripout removes outputs automatically
git commit -m "docs: add EDA analysis"

# Your local notebook still has outputs, but committed version is clean
```

To preserve outputs in Git (not recommended):

```bash
# Skip nbstripout for this commit
SKIP=nbstripout git commit -m "docs: include notebook outputs"
```

## Troubleshooting

### Hook Fails with "command not found"

**Problem:** Pre-commit can't find a tool (e.g., black, ruff)

**Solution:**

```bash
# Reinstall dev dependencies
uv pip install -e ".[dev]"

# Update pre-commit hooks
pre-commit install --install-hooks
```

### Mypy Errors on Valid Code

**Problem:** Mypy reports errors for third-party libraries without type stubs

**Solution:** Add to `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = [
    "new_library.*",
]
ignore_missing_imports = true
```

### Hooks Take Too Long

**Problem:** Pre-commit is slow on large commits

**Solution:**

```bash
# Run specific hooks instead of all
pre-commit run black
pre-commit run ruff

# Or commit staged files incrementally
git add src/data/
git commit -m "refactor: improve data processing"
git add src/models/
git commit -m "feat: add new model"
```

### Need to Update Hooks

Pre-commit hooks are versioned. To update:

```bash
# Update all hooks to latest versions
pre-commit autoupdate

# Commit the updated .pre-commit-config.yaml
git add .pre-commit-config.yaml
git commit -m "chore: update pre-commit hooks"
```

## Configuration

Pre-commit behavior is configured in two files:

### `.pre-commit-config.yaml`

Defines which hooks run and their settings:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        exclude: ^notebooks/
```

### `pyproject.toml`

Tool-specific settings:

```toml
[tool.black]
line-length = 100
exclude = '''
/(
    \.git
  | \.venv
  | notebooks
)/
'''

[tool.ruff]
line-length = 100
exclude = ["notebooks"]

[tool.bandit]
exclude_dirs = ["tests", "notebooks"]
```

## Best Practices

1. **Install hooks early** - Set up pre-commit at the start of development
1. **Run on all files initially** - `pre-commit run --all-files` to fix existing issues
1. **Don't skip hooks** - If hooks fail, fix the issues instead of bypassing
1. **Keep hooks updated** - Run `pre-commit autoupdate` periodically
1. **Understand failures** - Read error messages to learn what's wrong
1. **Commit auto-fixes separately** - Let pre-commit auto-fix, then review and commit

## Related Documentation

- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
