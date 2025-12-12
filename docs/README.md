# Documentation

This directory contains the MkDocs documentation for the Rossmann forecasting project.

## Local Development

### Install Dependencies

```bash
uv pip install -e ".[docs]"
```

### Serve Locally

```bash
mkdocs serve
```

Then visit: http://127.0.0.1:8000

The site will auto-reload when you save changes to any markdown files.

### Build Documentation

```bash
mkdocs build
```

This creates a `site/` directory with static HTML files.

## Deployment

### GitHub Pages

Deploy to GitHub Pages with one command:

```bash
mkdocs gh-deploy
```

This will:

1. Build the documentation
1. Push to `gh-pages` branch
1. Make it available at: `https://bradleyboehmke.github.io/rossmann-forecasting`

## Documentation Structure

```
docs/
├── index.md                    # Home page
├── getting-started/
│   ├── setup.md               # Installation guide
│   ├── quickstart.md          # Quick commands
│   └── structure.md           # Project structure
├── dataops/
│   ├── overview.md            # DataOps guide
│   ├── processing.md          # Data processing
│   ├── validation.md          # Data validation
│   ├── versioning.md          # DVC usage
│   └── pipeline.md            # DVC pipelines
├── modelops/
│   ├── overview.md            # ModelOps guide
│   ├── tracking.md            # MLflow tracking
│   ├── training.md            # Model training
│   ├── registry.md            # Model registry
│   └── tuning.md              # Hyperparameter tuning
├── deployment/
│   ├── overview.md            # Deployment guide
│   ├── fastapi.md             # FastAPI setup
│   ├── streamlit.md           # Streamlit dashboard
│   └── docker.md              # Docker deployment
├── monitoring/
│   ├── overview.md            # Monitoring guide
│   ├── drift.md               # Data drift
│   ├── performance.md         # Performance tracking
│   └── logging.md             # Prediction logging
├── testing/
│   ├── overview.md            # Testing guide
│   ├── data-tests.md          # Data tests
│   ├── model-tests.md         # Model tests
│   └── api-tests.md           # API tests
├── cicd/
│   ├── overview.md            # CI/CD overview
│   ├── github-actions.md      # GitHub Actions
│   └── workflows.md           # Workflow examples
├── api/
│   ├── data.md                # Data module reference
│   ├── features.md            # Features module reference
│   ├── models.md              # Models module reference
│   ├── evaluation.md          # Evaluation module reference
│   └── monitoring.md          # Monitoring module reference
└── contributing/
    ├── guidelines.md          # Contributing guidelines
    └── development.md         # Development setup
```

## Writing Documentation

### Markdown Features

MkDocs Material supports many useful features:

#### Admonitions

```markdown
!!! note "Optional Title"
    This is a note admonition

!!! tip
    Helpful tip here

!!! warning
    Important warning

!!! danger
    Critical information
```

#### Code Blocks with Syntax Highlighting

````markdown
​```python
def hello_world():
    print("Hello, World!")
​```
````

#### Tabs

````markdown
=== "Python"
    ​```python
    print("Hello")
    ​```

=== "Bash"
    ​```bash
    echo "Hello"
    ​```
````

#### Mermaid Diagrams

````markdown
​```mermaid
graph LR
    A[Raw Data] --> B[Processing]
    B --> C[Model Training]
    C --> D[Deployment]
​```
````

### API Documentation

Use `mkdocstrings` to auto-generate API docs from docstrings:

```markdown
::: src.data.make_dataset
    handler: python
    options:
      show_source: true
```

## Configuration

The `mkdocs.yml` file in the project root controls:

- Theme and styling
- Navigation structure
- Plugins
- Markdown extensions

## Tips

1. **Auto-reload**: `mkdocs serve` watches for changes
1. **Link checking**: Use relative links between pages
1. **Images**: Place in `docs/assets/` directory
1. **Code examples**: Keep them short and focused
1. **Navigation**: Update `mkdocs.yml` when adding new pages
