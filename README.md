# Arabic Sentiment Analysis

This project builds and serves an Arabic sentiment analysis model for customer review classification. It uses an AraBERT-based transformer model trained on Arabic text and exposes a prediction API through a BentoML service.

## Features

- Arabic text preprocessing and normalization
- AraBERT-based sentiment classification
- Batch prediction API for review text
- MLflow tracking artifacts for experiments
- BentoML deployment-ready service
- Vocabulary drift monitoring with Evidently, Prometheus, and Grafana
- Unit tests for prediction behavior

## Project structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── arabert_sentiment/
│       ├── model.fp16.engine
│       ├── model.onnx
│       ├── model.safetensors
│       └── vocabulary_reference.json
├── mlruns/                         # MLflow experiments and artifacts
├── monitoring/
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   └── vocabulary-drift.json
│   │   ├── provisioning/
│   │   └── dashboard-screenshot.png
│   ├── nginx/
│   │   └── nginx.conf
│   ├── prometheus/
│   │   ├── alert_rules.yml
│   │   └── prometheus.yml
│   └── docker-compose.yml
├── notebooks/
│   └── arabert-sentiment.ipynb
├── reports/
├── scripts/
│   └── build_vocabulary_reference.py
├── src/
│   └── arabic_sentiment_analysis/
│       ├── export_onnx.py
│       ├── model.py
│       ├── predict.py
│       ├── preprocessing.py
│       ├── service.py
│       ├── service_tensorrt.py
│       ├── training.py
│       └── vocabulary_drift.py
├── tests/                           # Unit and API tests
├── bentofile-tensorrt.yaml
├── bentofile.yaml
├── Dockerfile
├── locustfile.py
├── pyproject.toml
├── pytest.ini
├── uv.lock
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

## Exporting to ONNX

Install the optional ONNX dependency and export the locally saved model:

```bash
uv sync --extra onnx
uv run arabic-sentiment-export-onnx
```

The exporter writes `models/arabert_sentiment/model.onnx` by default, checks the
exported graph, and supports variable batch and sequence lengths. To choose a
different destination:

```bash
uv run arabic-sentiment-export-onnx --output path/to/model.onnx
```

## Building and serving the TensorRT Bento

The TensorRT Bento uses the prebuilt FP16 engine at
`models/arabert_sentiment/model.fp16.engine`. Install the project and its
TensorRT dependencies, then build the Bento:

```bash
uv sync --extra tensorrt
uv run bentoml build -f bentofile-tensorrt.yaml --name arabic-sentiment-tensorrt --platform windows
```

Serve the built Bento on a machine with an NVIDIA GPU and a compatible NVIDIA
driver:

```bash
uv run bentoml serve arabic-sentiment-tensorrt:latest --host 0.0.0.0 --port 8000
```

The engine was built with TensorRT 11.4 for Windows and the local GTX 1650.
TensorRT engine plans are platform- and runtime-specific: rebuild the engine
on the target operating system, GPU, and TensorRT version before building a
production Bento for a different environment. In particular, a Windows engine
cannot be served from a Linux container. The Bento loads the serialized engine
directly, chunks requests larger than the engine's batch limit of 16, and does
not silently fall back to PyTorch.

## BentoML service

The service is defined in `src/arabic_sentiment_analysis/service.py` and loads
the model from the local `models/arabert_sentiment` directory.

To start the service locally:

```bash
bentoml serve src/arabic_sentiment_analysis/service.py
```

You can then send batch requests with a list of Arabic review strings.

## Model and artifacts

- Model directory: `models/arabert_sentiment`
- Training metadata and experiment tracking: `mlruns/`
- Evaluation report artifacts are stored under `mlruns/.../artifacts/`

## Vocabulary drift monitoring

The service compares tokenizer token frequencies from inference traffic with a
reference sample built from the review corpus. It exports the rolling
Evidently Population Stability Index as
`arabic_sentiment_vocabulary_psi` on BentoML's Prometheus `/metrics` endpoint.
The metric is updated after at least 1,000 non-special tokens have been observed
and refreshes for each additional 1,000 tokens, using the most recent 50,000.
`[UNK]` tokens are retained; padding and other special tokens are excluded.

Generate or refresh the packaged reference after updating the corpus or model
tokenizer:

```bash
uv run python scripts/build_vocabulary_reference.py
```

The generated `models/arabert_sentiment/vocabulary_reference.json` is packaged
with both Bento variants and the Docker image; the raw corpus is not included in
the serving image.

Start both API cohorts, the weighted Nginx router, and monitoring stack from
the `monitoring` directory:

```bash
docker compose up --build
```

The router is exposed at `http://localhost:8000` and sends 5% of requests to
`sentiment-canary` and 95% to `sentiment-stable`, as configured in
`monitoring/nginx/nginx.conf`. Prometheus scrapes each cohort directly, with
`rollout="stable"` and `rollout="canary"` labels, so their PSI can be compared.
For a real rollout, deploy the already-built stable and candidate images using
`SENTIMENT_STABLE_IMAGE` and `SENTIMENT_CANARY_IMAGE`; do not build both from
the same source revision and treat that as a canary. For example, create
`monitoring/.env` with the immutable image tags:

```dotenv
SENTIMENT_STABLE_IMAGE=registry.example/arabic-sentiment:stable
SENTIMENT_CANARY_IMAGE=registry.example/arabic-sentiment:candidate-2026-10-12
```

Then deploy those images without rebuilding them:

```bash
docker compose up -d --no-build
```

Grafana is available at `http://localhost:3000` (default credentials
`admin` / `admin`) with the provisioned **Arabic Sentiment Model Drift**
dashboard. Prometheus is at `http://localhost:9090`; it fires the
`ArabicSentimentVocabularyDrift` alert when PSI exceeds `0.25` for one minute.
The rule evaluates each rollout cohort, and the alert is visible at
`http://localhost:9090/alerts`. No PSI value is reported until enough inference
tokens have been observed.

The Prometheus rule raises an alert but does not send notifications by itself.
The SMTP settings in `monitoring/docker-compose.yml` configure Grafana's mail
server; they do not route Prometheus alerts to an inbox. To receive email
notifications, provide `SMTP_USER` and `SMTP_PASSWORD` in `monitoring/.env` and
configure an email contact point and notification policy in the alerting
system.

The dashboard compares stable and canary vocabulary PSI and shows P95 inference
latency alongside container monitoring metrics:

![Grafana dashboard showing P95 inference latency and stable/canary vocabulary PSI](monitoring/grafana/dashboard-screenshot.png)

### Canary rollout stages

Keep each new model build isolated in the canary cohort, and advance only after
checking prediction errors, latency, and the cohort's vocabulary PSI in Grafana.
The Prometheus `ArabicSentimentVocabularyDrift` alert triggers when PSI is
greater than `0.25` for one minute.

1. **Baseline:** Run the stable image at 100% and record its error rate,
   latency, and PSI before deploying a candidate.
2. **Initial canary (5/95):** Route 5% to the candidate and 95% to stable.
   Observe for at least 30 minutes and collect enough traffic for the PSI
   window to populate.
3. **Expand (25/75):** If errors and latency remain within the existing service
   SLOs and candidate PSI stays at or below `0.25`, set the Nginx weights to
   canary `25`, stable `75`; observe for at least one hour.
4. **Majority canary (50/50):** If the same checks pass, set weights to
   canary `50`, stable `50`; observe for at least two hours.
5. **Promote (100/0):** After the 50/50 checks pass, route all traffic to the
   canary and mark the stable upstream `down`. Keep the stable image available
   for immediate rollback.
6. **Rollback:** At any stage, if the candidate breaches the error/latency SLO
   or its PSI alert fires, set stable to `weight=100` and mark canary `down`,
   then investigate before resuming.

For stages 2–4, edit the two `weight` values in the Nginx upstream. To promote
or roll back, use Nginx's `down` parameter for the zero-traffic upstream rather
than `weight=0`. Apply each change with
`docker compose exec nginx nginx -s reload`.

## Notes

- The model expects Arabic text and preprocesses it before tokenization.
- Input text is truncated and padded to a maximum length of 128 tokens.
- Predictions return the label name and the confidence score for the most likely class.

## License

This project is for educational and research use within the repository context.
