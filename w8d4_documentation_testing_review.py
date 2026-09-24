from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def check_path(path: Path, description: str) -> bool:
    """Check whether a required project file or directory exists."""
    exists = path.exists()
    status = "PASS" if exists else "FAIL"
    print(f"[{status}] {description}: {path.relative_to(REPO_ROOT)}")
    return exists


def run_project_review() -> None:
    """Run basic documentation and project-integrity checks for W8D4."""
    checks = [
        (
            REPO_ROOT / "w4d4_fastapi_model_serving.py",
            "W8D1 FastAPI model-serving implementation",
        ),
        (
            REPO_ROOT / "Dockerfile",
            "W8D1 Docker configuration",
        ),
        (
            REPO_ROOT / "w8d2_ragas_evaluation.py",
            "W8D2 Ragas evaluation implementation",
        ),
        (
            REPO_ROOT / "w8d3_haystack_api.py",
            "W8D3 Haystack retrieval implementation",
        ),
        (
            REPO_ROOT / "output_evidence" / "w8d2",
            "W8D2 evaluation evidence",
        ),
        (
            REPO_ROOT / "output_evidence" / "w8d3",
            "W8D3 retrieval evidence",
        ),
        (
            REPO_ROOT / "tests" / "test_w8d4.py",
            "W8D4 automated tests",
        ),
        (
            REPO_ROOT / "output_evidence" / "w8d4",
            "W8D4 evidence directory",
        ),
    ]

    print("=" * 70)
    print("W8D4 Documentation, Testing & Code Review")
    print("=" * 70)

    results = [
        check_path(path, description)
        for path, description in checks
    ]

    passed = sum(results)
    total = len(results)

    print("-" * 70)
    print(f"Project review result: {passed}/{total} checks passed")

    if passed == total:
        print("Overall status: PASS")
    else:
        print("Overall status: REVIEW REQUIRED")


if __name__ == "__main__":
    run_project_review()