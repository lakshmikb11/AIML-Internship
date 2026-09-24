from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_w8d1_api_files_exist():
    """Verify the Dockerized ML API implementation and model are present."""
    api_file = REPO_ROOT / "w4d4_fastapi_model_serving.py"
    model_file = REPO_ROOT / "output_evidence" / "w4d3" / "linear_regression_model.joblib"

    assert api_file.exists(), "FastAPI model-serving file is missing."
    assert model_file.exists(), "Trained model file is missing."


def test_w8d2_ragas_evidence_exists():
    """Verify W8D2 Ragas evaluation evidence is present."""
    evidence_dir = REPO_ROOT / "output_evidence" / "w8d2"

    assert evidence_dir.exists(), "W8D2 evidence directory is missing."
    assert any(evidence_dir.iterdir()), "W8D2 evidence directory is empty."


def test_w8d3_retrieval_evidence_exists():
    """Verify W8D3 retrieval evaluation evidence is present."""
    evidence_dir = REPO_ROOT / "output_evidence" / "w8d3"

    assert evidence_dir.exists(), "W8D3 evidence directory is missing."
    assert any(evidence_dir.iterdir()), "W8D3 evidence directory is empty."


def test_dockerfile_exists():
    """Verify the W8D1 Docker configuration is present."""
    dockerfile = REPO_ROOT / "Dockerfile"

    assert dockerfile.exists(), "Dockerfile is missing."


def test_w8_branch_documentation_exists():
    """Verify that W8 documentation/self-review evidence is present."""
    evidence_root = REPO_ROOT / "output_evidence"

    assert evidence_root.exists(), "Output evidence directory is missing."