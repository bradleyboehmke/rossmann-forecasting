# Deployment

This directory contains deployment infrastructure for the Rossmann sales forecasting model, including a production-ready FastAPI backend and an interactive Streamlit dashboard.

## Directory Structure

```
deployment/
├── api/                    # FastAPI REST API
│   ├── __init__.py
│   ├── main.py             # API server with prediction endpoints
│   └── test_api.sh         # API testing script
├── streamlit/              # Streamlit dashboard
│   ├── __init__.py
│   ├── Home.py             # Main dashboard page
│   ├── run_app.sh          # Quick start script
│   ├── utils/              # Shared utilities
│   │   ├── api_client.py   # FastAPI client wrapper
│   │   └── validation.py   # Input validation helpers
│   └── pages/
│       ├── 1_📈_Predictions.py         # Prediction interface (2 tabs)
│       ├── 2_🔧_Model_Management.py    # Model lifecycle management
│       └── 3_📚_Documentation.py       # API docs and guides
└── README.md               # This file
```

## Components

### FastAPI Backend ([api/main.py](api/main.py))

Production REST API for model predictions with the following endpoints:

- **GET /** - API information and available endpoints
- **GET /health** - Health check with model status
- **POST /predict** - Generate sales predictions (single or batch)
- **GET /model/info** - Model metadata and version information
- **GET /model/versions** - List Production and Staging versions

#### Features

- **Pydantic validation**: Request/response schema validation
- **Model caching**: In-memory model caching for fast predictions
- **CORS enabled**: Cross-origin resource sharing for web clients
- **Automatic docs**: Interactive API docs at `/docs` (Swagger UI)
- **Error handling**: Comprehensive error responses with timestamps

### Streamlit Dashboard ([streamlit/](streamlit/))

Interactive multi-page web application for model monitoring and predictions:

**Home Page** ([Home.py](streamlit/Home.py))

- **API Health Check**: Real-time FastAPI server status
- **Model Registry**: Production/Staging version display
- **System Status**: Model cache and server availability
- **Quick Start Guide**: Step-by-step instructions
- **Feature Overview**: Dashboard capabilities

**Predictions Page** ([pages/1_📈_Predictions.py](streamlit/pages/1_%F0%9F%93%88_Predictions.py))

- **Tab 1: Single Prediction**
    - Real-time forecasts for one store/date
    - Interactive form with simplified inputs (6 fields: Store, Date, Open, Promo, StateHoliday, SchoolHoliday)
    - **Auto-calculates Day of Week** from selected date (no manual input needed!)
    - Client-side validation before API call
    - Instant prediction with interpretation guide
- **Tab 2: Batch Upload**
    - CSV upload with just **6 fields** (Store, Date, Open, Promo, StateHoliday, SchoolHoliday)
    - **Auto-calculates Day of Week** from Date column (no manual input needed!)
    - **Flexible date formats** accepted (YYYY-MM-DD, MM/DD/YY, MM/DD/YYYY, etc.)
    - Template CSV download with sample data
    - Automatic date normalization to YYYY-MM-DD format
    - Batch validation and processing via API
    - Results export with store-level summaries (includes DayOfWeek column)
    - **Key Feature**: API automatically handles store metadata merge, data cleaning, and feature engineering

**Model Management Page** ([pages/2_🔧_Model_Management.py](streamlit/pages/2_%F0%9F%94%A7_Model_Management.py))

- View all registered models and versions
- Model lifecycle status (None → Staging → Production → Archived)
- Promote models through lifecycle stages
- Archive old model versions

**Documentation Page** ([pages/3_📚_Documentation.py](streamlit/pages/3_%F0%9F%93%9A_Documentation.py))

- API reference with curl and Python examples
- Model architecture overview
- Feature engineering details
- Deployment guides

## Getting Started

### Prerequisites

1. **Install dependencies**:

    ```bash
    uv pip install -e ".[dev]"
    ```

1. **Ensure models are registered** in MLflow Model Registry with at least one version in Production or Staging stage.

    **Note**: The FastAPI server uses file-based MLflow tracking (`file://mlruns`) by default. If you prefer to use an MLflow tracking server:

    ```bash
    # Optional: Start MLflow UI for visualization
    mlflow ui --port 5000

    # Set environment variable to use HTTP tracking
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    ```

    The server will automatically detect and use the appropriate tracking URI.

### Running the FastAPI Server

**Option 1: Using launch script (recommended)**:

```bash
# From project root
bash scripts/launch_api.sh
```

**Option 2: Direct execution**:

```bash
cd deployment/api
python main.py
```

**Option 3: Using uvicorn**:

```bash
cd deployment/api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API server will be available at:

- **API**: http://localhost:8000
- **Interactive docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Running the Streamlit Dashboard

**Option 1: Using launch script (recommended)**:

```bash
# From project root
bash scripts/launch_streamlit.sh
```

**Option 2: Direct execution**:

```bash
cd deployment/streamlit
streamlit run Home.py
```

**Option 3: Quick start script**:

```bash
cd deployment/streamlit
./run_app.sh
```

The dashboard will be available at:

- **Dashboard**: http://localhost:8501

### Using the Streamlit Dashboard

#### 1. Check System Status

On the **Home** page, verify:

- ✅ FastAPI Server is "🟢 Online"
- ✅ Model Cache shows a loaded model
- ✅ Production and Staging versions are displayed

If the API is offline, start it first (see instructions above).

#### 2. Single Prediction Workflow

Navigate to **Predictions** → **Single Prediction** tab:

1. Fill in the required fields:

    - Store ID (1-1115)
    - Prediction Date (Day of Week is **auto-calculated** from the date)
    - Store Open? (Yes/No)
    - State Holiday type
    - Promotion Active? (Yes/No)
    - School Holiday? (Yes/No)

1. Click **🚀 Generate Prediction**

1. View results:

    - Predicted sales amount
    - Model version used
    - Input summary
    - Interpretation guide

#### 3. Batch Upload Workflow

Navigate to **Predictions** → **Batch Upload** tab:

1. **Download Template**:

    - Click "📥 Download Template" to get the CSV format
    - Template shows the exact **6 fields** required (DayOfWeek auto-calculated!)

1. **Prepare Your Data**:

    - Fill in your data matching the template format
    - Required columns: `Store, Date, Open, Promo, StateHoliday, SchoolHoliday`
    - **Date format is flexible**: YYYY-MM-DD, MM/DD/YY, MM/DD/YYYY all work!
    - **No need to calculate DayOfWeek** - it's done automatically

1. **Upload CSV**:

    - Click "Upload CSV file" and select your file
    - Streamlit will validate the CSV automatically
    - Dates are normalized to YYYY-MM-DD format
    - DayOfWeek column is added automatically
    - Preview the processed data (first 10 rows)

1. **Generate Predictions**:

    - Click **🚀 Generate N Predictions** button
    - Wait for API processing (progress spinner shown)
    - View prediction summary and results table

1. **Export Results**:

    - Click "📥 Download Predictions CSV" to save results
    - Results include: original fields + `DayOfWeek` + `Predicted_Sales`
    - Optionally view store-level summary (grouped statistics)

## API Usage Examples

### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "timestamp": "2025-01-19T10:30:00",
  "model_loaded": true,
  "model_version": "7"
}
```

### Single Prediction

**IMPORTANT**: The API accepts raw data in **train.csv format** and automatically handles:

1. Merging with store metadata
1. Data cleaning
1. Feature engineering
1. Model prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "Store": 1,
        "DayOfWeek": 5,
        "Date": "2015-08-01",
        "Open": 1,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 0
      }
    ],
    "model_stage": "Production"
  }'
```

Response:

```json
{
  "predictions": [11231.71],
  "model_version": "2",
  "count": 1,
  "timestamp": "2025-12-20T10:29:33.744334"
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "Store": 1,
        "DayOfWeek": 5,
        "Date": "2015-08-01",
        "Open": 1,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 0
      },
      {
        "Store": 2,
        "DayOfWeek": 5,
        "Date": "2015-08-01",
        "Open": 1,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 0
      }
    ],
    "model_stage": "Production"
  }'
```

Response:

```json
{
  "predictions": [11351.80, 11426.57],
  "model_version": "2",
  "count": 2,
  "timestamp": "2025-12-20T10:29:55.884611"
}
```

### Python Client Example

```python
import requests

API_URL = "http://localhost:8000"

# Prepare raw input data (train.csv format)
inputs = [
    {
        "Store": 1,
        "DayOfWeek": 5,
        "Date": "2015-08-01",
        "Open": 1,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 0
    }
]

# Make prediction
response = requests.post(
    f"{API_URL}/predict",
    json={"inputs": inputs, "model_stage": "Production"}
)

result = response.json()
prediction = result["predictions"][0]
print(f"Predicted Sales: ${prediction:,.2f}")
print(f"Model Version: {result['model_version']}")
```

### Test All Endpoints

Run the comprehensive test script:

```bash
bash deployment/api/test_api.sh
```

## Input Requirements

The API accepts data in **train.csv format**. You only need to provide 7 fields:

| Field         | Type | Range           | Description                      |
| ------------- | ---- | --------------- | -------------------------------- |
| Store         | int  | 1-1115          | Unique store identifier          |
| DayOfWeek     | int  | 1-7             | Day of week (1=Monday, 7=Sunday) |
| Date          | str  | YYYY-MM-DD      | Date in ISO format               |
| Open          | int  | 0-1             | Store open flag                  |
| Promo         | int  | 0-1             | Daily promotion active           |
| StateHoliday  | str  | "0","a","b","c" | State holiday type               |
| SchoolHoliday | int  | 0-1             | School holiday flag              |

**The API automatically handles**:

- Loading store metadata (StoreType, Assortment, CompetitionDistance, etc.)
- Feature engineering (calendar features, lags, rolling stats, etc.)
- Data cleaning and validation
- Model prediction

**No manual feature engineering required!**

## Production Deployment Considerations

### FastAPI

- **Multiple workers**: Use Gunicorn with Uvicorn workers
    ```bash
    gunicorn deployment.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
    ```
- **Reverse proxy**: Configure Nginx for load balancing
- **CORS**: Update allowed origins in production
- **HTTPS**: Enable TLS/SSL encryption
- **Monitoring**: Set up logging, metrics, and health checks

### Streamlit

- **Authentication**: Enable authentication (Streamlit Cloud or custom)
- **Caching**: Use `@st.cache_data` for model loading
- **Environment configs**: Separate dev/staging/prod configs
- **Resource limits**: Monitor memory and CPU usage

### Infrastructure

- **Docker containers**: See Phase 4 for containerization
- **Kubernetes**: See Phase 4 for orchestration
- **Auto-scaling**: Configure based on traffic patterns
- **Circuit breakers**: Implement fault tolerance

## Troubleshooting

**Model not loaded**:

- Verify models are registered in MLflow Model Registry (`mlruns/` directory should exist)
- Check that at least one version is in Production or Staging stage
- The API uses file-based tracking by default, so MLflow server is optional

**MLflow connection issues**:

- By default, API uses `file://mlruns` (local file-based tracking)
- If you want to use HTTP tracking server, set `MLFLOW_TRACKING_URI` environment variable
- Verify MLflow registry contains models: `ls -la mlruns/models/`
- Check file permissions on `mlruns/` directory

**Import errors**:

- Run from project root directory
- Verify virtual environment is activated
- Check that all dependencies are installed

## Next Steps

- **Phase 4**: Docker containerization and Kubernetes deployment
- **Monitoring**: Integrate Prometheus and Grafana
- **CI/CD**: GitHub Actions for automated deployment
- **A/B Testing**: Deploy multiple model versions simultaneously

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html)
- [Project Documentation](../docs/)
