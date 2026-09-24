# W8D1 Monitoring Strategy

## Health
Monitor /health and verify HTTP 200 with model_loaded=true.

## API
Monitor:
- Request count
- Request latency
- HTTP 4xx/5xx errors
- Prediction endpoint availability

## Container
Monitor:
- Container uptime
- Restart events
- CPU usage
- Memory usage
- Application logs

## CI/CD
The pipeline must complete:
1. Ruff linting
2. Pytest validation
3. Docker image build
4. Docker image push

A failed validation stage prevents the dependent stage from running.

## Model
Verify that the expected serialized model artifact is present
and loaded successfully during application startup.
