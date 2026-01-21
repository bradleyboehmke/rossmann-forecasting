# GitHub Actions CI/CD

This project uses GitHub Actions for continuous integration and deployment automation.

______________________________________________________________________

## Overview

Two workflows automate key project tasks:

1. **Test Workflow** - Runs pytest on every push and pull request
1. **Documentation Workflow** - Builds and deploys MkDocs to GitHub Pages

______________________________________________________________________

## Test Workflow

**File**: [.github/workflows/test.yml](../../.github/workflows/test.yml)

### Triggers

The test workflow runs automatically when:

- **Push to main**: Any commit pushed to the main branch
- **Pull requests**: Any PR targeting the main branch
- **Manual trigger**: Via the Actions tab in GitHub ("Run workflow" button)

### What It Does

1. **Checkout code** - Clones the repository
1. **Set up Python 3.10** - Installs Python runtime
1. **Install uv** - Installs the uv package manager with caching
1. **Install dependencies** - Creates venv and installs project with dev dependencies
1. **Run pytest** - Executes all tests with verbose output
1. **Upload test results** - Stores coverage reports as artifacts (30-day retention)

### Example Output

```
Run pytest
============================= test session starts ==============================
platform linux -- Python 3.10.x, pytest-7.x.x
collected 45 items

tests/test_data_processing.py::test_load_raw_data PASSED                 [  2%]
tests/test_data_processing.py::test_merge_store_info PASSED              [  4%]
tests/test_features.py::test_add_calendar_features PASSED                [  6%]
...
============================== 45 passed in 12.34s ==============================
```

### Viewing Results

1. Navigate to the **Actions** tab in your GitHub repository
1. Click on the latest workflow run
1. View test results in the job logs
1. Download coverage artifacts if needed

### Manual Trigger

To manually run tests:

1. Go to **Actions** tab
1. Select **"Run Tests"** workflow
1. Click **"Run workflow"** dropdown
1. Select branch and click **"Run workflow"** button

______________________________________________________________________

## Documentation Workflow

**File**: [.github/workflows/docs.yml](../../.github/workflows/docs.yml)

### Triggers

The documentation workflow runs when:

- **Push to main**: Changes to `docs/`, `mkdocs.yml`, or the workflow file itself
- **Manual trigger**: Via the Actions tab

### What It Does

**Build Job**:

1. Checkout code
1. Set up Python and install dependencies
1. Build documentation with `mkdocs build --strict`
1. Upload built site as Pages artifact

**Deploy Job**:

1. Deploy artifact to GitHub Pages
1. Publish to `https://<username>.github.io/<repo-name>/`

### Setup Requirements

**First-time setup** (one-time configuration):

1. Go to **Settings** > **Pages** in your repository
1. Under "Build and deployment":
    - Source: **GitHub Actions**
1. Save changes

The workflow will then automatically deploy on the next push.

### Viewing Documentation

Once deployed, your documentation will be available at:

```
https://<your-username>.github.io/rossmann-forecasting/
```

Example: `https://bradleyboehmke.github.io/rossmann-forecasting/`

### Deployment Status

Check deployment status:

1. **Actions** tab shows build progress
1. **Environments** section shows "github-pages" deployments
1. Green checkmark indicates successful deployment

______________________________________________________________________

## Workflow Configuration

### Dependencies

Both workflows use:

- **actions/checkout@v4** - Latest checkout action
- **actions/setup-python@v5** - Python installation
- **astral-sh/setup-uv@v3** - uv package manager with caching

### Caching

The uv action automatically caches dependencies to speed up subsequent runs:

- First run: ~2-3 minutes (installs everything)
- Cached runs: ~30-60 seconds (uses cached packages)

### Python Version

Both workflows use **Python 3.10** to match the development environment specified in `pyproject.toml`.

______________________________________________________________________

## Best Practices

### Testing

**Before pushing to main**:

```bash
# Run tests locally first
pytest -v

# Check that all tests pass
# Fix any failures before committing
```

**Pull request workflow**:

1. Create feature branch: `git checkout -b feature/my-feature`
1. Make changes and commit
1. Push branch: `git push origin feature/my-feature`
1. Open pull request on GitHub
1. Tests run automatically - wait for green checkmark
1. Review results before merging

### Documentation

**Before pushing docs changes**:

```bash
# Build docs locally to check for errors
mkdocs build --strict

# Serve locally to preview
mkdocs serve
# Visit http://localhost:8000
```

**Deployment process**:

1. Make changes to `docs/` files
1. Test locally with `mkdocs serve`
1. Commit and push to main
1. Workflow automatically builds and deploys
1. Check deployment status in Actions tab
1. Visit GitHub Pages URL to verify

______________________________________________________________________

## Troubleshooting

### Test Workflow Failures

**Common issues**:

1. **Import errors**: Missing dependencies in `pyproject.toml`

    ```bash
    # Fix: Add missing package to pyproject.toml
    uv pip install <package>
    ```

1. **Test failures**: Broken tests or code changes

    ```bash
    # Fix locally first
    pytest -v
    ```

1. **Coverage issues**: Not all code paths tested

    ```bash
    # Check coverage locally
    pytest --cov=src --cov-report=html
    ```

### Documentation Workflow Failures

**Common issues**:

1. **Build errors**: Invalid mkdocs.yml or broken links

    ```bash
    # Test build locally
    mkdocs build --strict
    ```

1. **Missing files**: References to non-existent pages

    ```
    ERROR   -  Doc file 'path/to/file.md' not found
    ```

    Fix: Check file paths in `mkdocs.yml` navigation

1. **Permission errors**: GitHub Pages not enabled

    - Solution: Enable Pages in repository Settings

### Viewing Workflow Logs

To debug issues:

1. Go to **Actions** tab
1. Click failed workflow run
1. Click on failed job
1. Expand failed step to see error details
1. Copy error message and fix locally

______________________________________________________________________

## Workflow Status Badges

Add status badges to your README:

```markdown
![Tests](https://github.com/<username>/rossmann-forecasting/workflows/Run%20Tests/badge.svg)
![Docs](https://github.com/<username>/rossmann-forecasting/workflows/Deploy%20Documentation/badge.svg)
```

Replace `<username>` with your GitHub username.

______________________________________________________________________

## Advanced Configuration

### Running Subset of Tests

Modify `.github/workflows/test.yml` to run specific tests:

```yaml
- name: Run pytest
  run: |
    source .venv/bin/activate
    pytest tests/test_features.py -v  # Only feature tests
```

### Custom MkDocs Theme

The documentation workflow uses the configuration in `mkdocs.yml`. To customize:

1. Edit `mkdocs.yml` locally
1. Test with `mkdocs serve`
1. Commit and push - workflow uses your config

### Scheduled Runs

Add scheduled testing (e.g., nightly builds):

```yaml
on:
  schedule:
    - cron: '0 0 * * *'  # Run daily at midnight UTC
```

______________________________________________________________________

## Cost and Usage

**Good news**: GitHub Actions is free for public repositories!

- **Public repos**: Unlimited minutes
- **Private repos**: 2,000 free minutes/month

**Storage**:

- Artifacts retained for 30 days (configurable)
- Test results and coverage reports count against storage

______________________________________________________________________

## Related Documentation

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Testing with pytest](../../README.md#testing)
- [Building Documentation](../../README.md#documentation)
- [uv Package Manager](https://github.com/astral-sh/uv)
