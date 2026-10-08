# Arabic Sentiment Analysis

This project builds and serves an Arabic sentiment analysis model for customer review classification. It uses an AraBERT-based transformer model trained on Arabic text and exposes a prediction API through a BentoML service.

## Features

- Arabic text preprocessing and normalization
- AraBERT-based sentiment classification
- Batch prediction API for review text
- MLflow tracking artifacts for experiments
- BentoML deployment-ready service
- Unit tests for prediction behavior

## Project structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── arabert_sentiment/
├── mlruns/
├── monitoring/
├── notebooks/
├── src/
│   └── arabic_sentiment_analysis/
│       ├── model.py
│       ├── predict.py
│       ├── preprocessing.py
│       ├── service.py
│       └── training.py
├── tests/
├── bentofile.yaml
├── Dockerfile
├── locustfile.py
├── pyproject.toml
├── pytest.ini
└── README.md
```

## Tech stack

- Python 3.12+
- PyTorch
- Hugging Face Transformers
- scikit-learn
- MLflow
- BentoML
- Pytest
- Ruff
- Locust

## Installation

Use `uv` to install dependencies and set up the environment:

```bash
uv sync --group dev
```

If you do not have `uv` installed yet:

```bash
pip install uv
```

## Running tests

```bash
uv run pytest
```

## Predicting sentiment

You can run a quick prediction using the package API:

```bash
uv run python -c "from arabic_sentiment_analysis.predict import predict_sentiment; print(predict_sentiment('الخدمة سيئة جدًا'))"
```

Expected output shape:

```python
{
    'sentiment': 'Negative',
    'confidence': 0.91
}
```

## BentoML service

The service is defined in `src/arabic_sentiment_analysis/service.py` and loads the model from the local `models/arabert_sentiment` directory.

To start the service locally:

```bash
bentoml serve src/arabic_sentiment_analysis/service.py
```

You can then send batch requests with a list of Arabic review strings.

## Model and artifacts

- Model directory: `models/arabert_sentiment`
- Training metadata and experiment tracking: `mlruns/`
- Evaluation report artifacts are stored under `mlruns/.../artifacts/`

## Notes

- The model expects Arabic text and preprocesses it before tokenization.
- Input text is truncated and padded to a maximum length of 128 tokens.
- Predictions return the label name and the confidence score for the most likely class.

## License

This project is for educational and research use within the repository context.
