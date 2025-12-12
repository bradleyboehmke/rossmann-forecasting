# Deployment

This directory contains deployment infrastructure for the Rossmann sales forecasting model.

## Components

### FastAPI Backend (`fastapi_app.py`)

REST API for model predictions with endpoints:

- `POST /predict` - Single store prediction
- `POST /predict_batch` - Batch predictions
- `GET /model_info` - Model metadata and version
- `GET /health` - Health check

### Streamlit Dashboard (`streamlit_app.py`)

Interactive web interface featuring:

- Single store forecasting
- Multi-store comparison
- What-if analysis
- Model performance metrics
- Data quality monitoring

### Supporting Modules

- `schemas.py` - Pydantic request/response models
- `model_loader.py` - MLflow model loading utilities
- `prediction_logger.py` - Prediction logging and tracking

## Usage

### Running FastAPI

```bash
uvicorn deployment.fastapi_app:app --reload
```

### Running Streamlit

```bash
streamlit run deployment/streamlit_app.py
```

### Running with Docker

```bash
docker-compose up
```

## API Documentation

Once running, visit:

- FastAPI docs: http://localhost:8000/docs
- Streamlit app: http://localhost:8501
