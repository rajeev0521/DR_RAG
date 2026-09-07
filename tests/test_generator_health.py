"""Unit test verifying AnswerGenerator produces real, non-error text on spot-checked queries."""

import json
from pathlib import Path
import pytest
from stp_rag.generation.answer import AnswerGenerator


def test_generator_health_and_spot_check():
    generator = AnswerGenerator(model_name="phi3:mini")
    is_healthy = generator.health_check()
    if not is_healthy:
        pytest.skip("Ollama is not running or phi3:mini is not reachable; skipping live generation test.")

    dev_path = Path("data/splits/dev.json")
    assert dev_path.exists(), "dev.json must exist for spot checks."

    with open(dev_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # Collect 10 spot-checked queries
    spot_queries = []
    for doc in docs:
        for qa in doc.get("qa_pairs", []):
            spot_queries.append((qa["question"], [doc["text"][:1000]]))
            if len(spot_queries) >= 10:
                break
        if len(spot_queries) >= 10:
            break

    assert len(spot_queries) >= 10, f"Expected at least 10 queries, got {len(spot_queries)}"

    for idx, (q, ctxs) in enumerate(spot_queries):
        ans, prompt = generator.generate(q, ctxs)
        assert not ans.startswith("[Ollama Offline:"), f"Query {idx} failed with offline error: {ans}"
        assert not ans.startswith("[Ollama Error"), f"Query {idx} failed with error: {ans}"
        assert len(ans.strip()) > 0, f"Query {idx} generated empty answer string"
