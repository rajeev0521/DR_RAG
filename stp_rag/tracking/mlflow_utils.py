"""MLflow tracking utilities for reproducible experiment management."""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional
from ..eval.results_schema import ArmResult


class MLflowTracker:
    """Helper wrapper for MLflow experiment and run management."""

    def __init__(
        self,
        experiment_name: str = "STP-RAG-Ablations",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._mlflow = None
        self._active_run = None

    def _init_mlflow(self):
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow
                if self.tracking_uri:
                    self._mlflow.set_tracking_uri(self.tracking_uri)
                self._mlflow.set_experiment(self.experiment_name)
            except Exception:
                self._mlflow = None

    @contextlib.contextmanager
    def run(self, run_name: str, tags: Optional[Dict[str, Any]] = None):
        """Context manager for an MLflow run."""
        self._init_mlflow()
        if self._mlflow is None:
            yield "dummy_local_run"
            return

        with self._mlflow.start_run(run_name=run_name, tags=tags or {}) as active_run:
            self._active_run = active_run
            try:
                yield active_run.info.run_id
            finally:
                self._active_run = None

    def log_params(self, params: Dict[str, Any]):
        self._init_mlflow()
        if self._mlflow and self._active_run:
            try:
                self._mlflow.log_params(params)
            except Exception:
                pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self._init_mlflow()
        if self._mlflow and self._active_run:
            try:
                self._mlflow.log_metrics(metrics, step=step)
            except Exception:
                pass

    def log_arm_result(self, result: ArmResult):
        """Logs all fields from an ArmResult into the active MLflow run."""
        self._init_mlflow()
        if not (self._mlflow and self._active_run):
            return

        params = {
            "system": result.system,
            "seed": result.seed,
            "split": result.split,
            "embedding_model_version": result.embedding_model_version,
            "corpus_version": result.corpus_version,
        }
        metrics = {
            "recall_at_1": result.recall_at_1,
            "recall_at_5": result.recall_at_5,
            "recall_at_10": result.recall_at_10,
            "mrr": result.mrr,
            "em": result.em,
            "f1": result.f1,
            "chunks_per_doc": result.chunks_per_doc,
            "tokens_per_chunk": result.tokens_per_chunk,
            "build_time_sec": result.build_time_sec,
            "retrieval_latency_ms": result.retrieval_latency_ms,
            "ragas_faithfulness": result.ragas_faithfulness,
            "ragas_answer_relevancy": result.ragas_answer_relevancy,
            "ragas_context_precision": result.ragas_context_precision,
            "ragas_context_recall": result.ragas_context_recall,
        }
        self.log_params(params)
        self.log_metrics(metrics)
