FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "bentoml>=1.4.30" "transformers>=5.17.0"

COPY src/arabic_sentiment_analysis /app/src/arabic_sentiment_analysis
COPY models/arabert_sentiment /app/models/arabert_sentiment

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["bentoml", "serve", "src.arabic_sentiment_analysis.service:SentimentService", "--host", "0.0.0.0", "--port", "8000"]