FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-w8d1.txt .

RUN pip install --no-cache-dir -r requirements-w8d1.txt

COPY w8d1_dockerised_ml_api.py .
COPY output_evidence/w4d3/linear_regression_model.joblib ./output_evidence/w4d3/linear_regression_model.joblib

EXPOSE 8000

CMD ["uvicorn", "w8d1_dockerised_ml_api:app", "--host", "0.0.0.0", "--port", "8000"]
