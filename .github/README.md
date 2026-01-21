# GitHub Workflows

This directory contains GitHub Actions workflows for CI/CD automation.

## Workflows

### 1. Test Workflow ([workflows/test.yml](workflows/test.yml))

**Purpose**: Automated testing with pytest

**Triggers**:

- Push to main branch
- Pull requests to main
- Manual trigger via Actions tab

**What it does**:

1. Sets up Python 3.10
1. Installs dependencies with uv
1. Runs pytest with verbose output
1. Uploads test results and coverage

### 2. Documentation Workflow ([workflows/docs.yml](workflows/docs.yml))

**Purpose**: Build and deploy MkDocs to GitHub Pages

**Triggers**:

- Push to main (when docs/, mkdocs.yml, or workflow file changes)
- Manual trigger via Actions tab

**What it does**:

1. Builds documentation with `mkdocs build --strict`
1. Deploys to GitHub Pages
1. Available at: `https://<username>.github.io/rossmann-forecasting/`

## Setup

### Enable GitHub Pages

For the documentation workflow to work:

1. Go to **Settings** > **Pages**
1. Under "Build and deployment":
    - Source: **GitHub Actions**
1. Save

### Manual Triggers

Both workflows can be manually triggered:

1. Go to **Actions** tab
1. Select the workflow
1. Click **"Run workflow"**
1. Choose branch and click **"Run workflow"** button

## Documentation

Full documentation available at: [docs/cicd/github-actions.md](../docs/cicd/github-actions.md)

Or visit the deployed docs: `https://<username>.github.io/rossmann-forecasting/cicd/github-actions/`
