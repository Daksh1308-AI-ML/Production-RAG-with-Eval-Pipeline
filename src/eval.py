"""RAGAS evaluation pipeline."""

import argparse
import json
import time
from typing import List, Dict, Optional
from pathlib import Path

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

from .config import config

_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


class RAGEvaluator:
    """Evaluate RAG system with RAGAS."""

    def __init__(self):
        # ponytail: ragas skips its tolerant parser for langchain LLMs and
        # calls model_validate_json directly, so qwen's prose/fenced output
        # crashed. If JUDGE_API_KEY is set, use an OpenAI-compatible hosted
        # judge (e.g. OpenRouter free models) with enforced JSON mode; else
        # fall back to local Ollama JSON mode.
        if config.eval.judge_api_key:
            self.judge_llm = ChatOpenAI(
                model=config.eval.judge_model,
                temperature=0.2,
                base_url=config.eval.judge_base_url,
                api_key=config.eval.judge_api_key,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        else:
            self.judge_llm = ChatOllama(
                model=config.eval.judge_model,
                temperature=0.2,
                base_url=config.ollama.base_url,
                format="json",
                num_ctx=config.ollama.num_ctx,
            )
        self.judge_embeddings = OllamaEmbeddings(
            model=config.ollama.embedding_model,
            base_url=config.ollama.base_url
        )

class LocalJudgeEvaluator(RAGEvaluator):
    """Local judge for metrics that don't need embeddings (faithfulness, context_recall).

    Uses Ollama qwen2.5:7b — no API quota, no embedding model co-resident
    (avoids the 4GB VRAM OOM).
    """

    def __init__(self):
        self.judge_llm = ChatOllama(
            model=config.eval.judge_model,
            temperature=0.2,
            base_url=config.ollama.base_url,
            format="json",
            num_ctx=config.ollama.num_ctx,
        )
        self.judge_embeddings = None  # never used — MetricWithLLM only


    def run_evaluation(
        self,
        rag_pipeline,
        eval_dataset: Optional[EvaluationDataset] = None,
        metrics: Optional[List[str]] = None,
        name: str = ""
    ) -> Dict:
        """Run full RAGAS evaluation."""
        if eval_dataset is None:
            eval_dataset = self.load_eval_dataset()

        tag = f"[{name}] " if name else ""
        t0 = time.time()
        print(f"{tag}generating responses ({len(eval_dataset.samples)} samples)...", flush=True)
        eval_ds = self._generate_responses(rag_pipeline, eval_dataset)
        print(f"{tag}responses done ({time.time() - t0:.0f}s)", flush=True)

        from ragas.run_config import RunConfig
        # ponytail: local Ollama judge can't serve ragas' default 16 concurrent
        # workers — requests queue and blow the 180s timeout. Keep it fully
        # serialized: >1 worker races the OpenRouter judge into returning None
        # responses ('NoneType' object is not iterable at 0/it). Free-tier
        # serial is ~53s/sample — acceptable for the nightly A/B budget.
        run_config = RunConfig(
            max_workers=1,
            timeout=600,
            max_retries=5,
        )

        metrics = metrics or list(_METRICS)
        print(f"{tag}running RAGAS metrics {metrics}...", flush=True)
        # Run RAGAS evaluation
        result = evaluate(
            dataset=eval_ds,
            metrics=[_METRICS[m] for m in metrics],
            llm=self.judge_llm,
            embeddings=self.judge_embeddings,
            run_config=run_config,
            raise_exceptions=True
        )
        print(f"{tag}metrics done ({time.time() - t0:.0f}s)", flush=True)

        result_df = result.to_pandas()
        mean_row = result_df.mean(numeric_only=True).to_dict()

        return {
            "faithfulness": mean_row.get("faithfulness", 0.0),
            "answer_relevancy": mean_row.get("answer_relevancy", 0.0),
            "context_precision": mean_row.get("context_precision", 0.0),
            "context_recall": mean_row.get("context_recall", 0.0),
            "details": result
        }

    def _generate_responses(
        self,
        rag_pipeline,
        eval_dataset: EvaluationDataset
    ) -> EvaluationDataset:
        """Run the pipeline over every sample and return filled samples."""
        generated = []
        for sample in eval_dataset.to_list():
            response = rag_pipeline.query(sample.get("user_input", ""))
            sources = getattr(response, "sources", None) or []
            generated.append(
                SingleTurnSample(
                    user_input=sample.get("user_input", ""),
                    reference=sample.get("reference", ""),
                    reference_contexts=sample.get("reference_contexts"),
                    response=getattr(response, "answer", ""),
                    retrieved_contexts=[
                        s.get("content")
                        if isinstance(s, dict)
                        else getattr(s, "page_content", str(s))
                        for s in sources
                    ],
                )
            )
        return EvaluationDataset(samples=generated)

    def compare_strategies(
        self,
        pipelines: Dict[str, object],
        eval_dataset: Optional[EvaluationDataset] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """Compare multiple RAG strategies."""
        results = {}
        for name, pipeline in pipelines.items():
            t0 = time.time()
            print(f"\n=== {name} ===", flush=True)
            try:
                results[name] = self.run_evaluation(pipeline, eval_dataset,
                                                    metrics=metrics, name=name)
            except Exception as e:
                # Judge failures shouldn't abort the whole batch: record NaN
                # and keep going so results are still written.
                print(f"{name} FAILED after {time.time() - t0:.0f}s: "
                      f"{type(e).__name__}: {e}", flush=True)
                results[name] = {
                    k: float("nan")
                    for k in ("faithfulness", "answer_relevancy",
                              "context_precision", "context_recall")
                }
            print(f"{name} total: {time.time() - t0:.0f}s", flush=True)
        return results


def _build_pipelines(chunks, strategies: List[str]) -> Dict[str, object]:
    """Build one RAGPipeline per strategy over shared chunks."""
    from .rag import RAGPipeline
    pipelines = {}
    for name in strategies:
        pipe = RAGPipeline(strategy=name)
        pipe.initialize(chunks)
        pipelines[name] = pipe
    return pipelines


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS evaluation of RAG strategies")
    parser.add_argument("--dataset", default=config.eval.eval_dataset_path,
                        help="path to eval dataset JSON")
    parser.add_argument("--output", default=str(config.evaluation_dir / "results"),
                        help="output directory for results JSON")
    parser.add_argument("--limit", type=int, default=0,
                        help="limit number of QA pairs (0 = all)")
    parser.add_argument("--strategies", default="baseline,hybrid,rerank,full",
                        help="comma-separated strategies to evaluate")
    parser.add_argument("--metrics", default="",
                        help="comma-separated metrics "
                             "(faithfulness,answer_relevancy,context_precision,"
                             "context_recall)")
    args = parser.parse_args()

    metrics = [m for m in args.metrics.split(",") if m] or list(_METRICS)
    unknown = [m for m in metrics if m not in _METRICS]
    if unknown:
        parser.error(f"unknown metrics: {unknown}")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        parser.error(f"dataset not found: {dataset_path}")

    t0 = time.time()
    print(f"loading dataset: {dataset_path}", flush=True)
    evaluator = RAGEvaluator()
    eval_dataset = evaluator.load_eval_dataset(dataset_path)
    if args.limit > 0:
        eval_dataset = EvaluationDataset(samples=eval_dataset.samples[:args.limit])
        print(f"limited to {args.limit} samples", flush=True)

    from .chunker import DocumentChunker
    print("loading filings + chunking...", flush=True)
    with open(config.processed_dir / "filings.json", encoding="utf-8") as f:
        filings = json.load(f)
    chunks = DocumentChunker().chunk_documents(filings)
    print(f"chunks: {len(chunks)} ({time.time() - t0:.0f}s)", flush=True)

    strategies = [s for s in args.strategies.split(",") if s]
    print(f"building strategy pipelines: {strategies}...", flush=True)
    pipelines = _build_pipelines(chunks, strategies)

    print("running evaluation on all strategies...", flush=True)
    results = evaluator.compare_strategies(pipelines, eval_dataset, metrics=metrics)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        strategy: {
            "faithfulness": r.get("faithfulness") if r.get("faithfulness") == r.get("faithfulness") else None,
            "answer_relevancy": r.get("answer_relevancy") if r.get("answer_relevancy") == r.get("answer_relevancy") else None,
            "context_precision": r.get("context_precision") if r.get("context_precision") == r.get("context_precision") else None,
            "context_recall": r.get("context_recall") if r.get("context_recall") == r.get("context_recall") else None,
        }
        for strategy, r in results.items()
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for strategy, r in results.items():
        details = r.get("details")
        if details is not None:
            details.to_pandas().to_json(
                out_dir / f"{strategy}_detailed.json",
                orient="records",
                indent=2,
                force_ascii=False,
            )
        else:
            (out_dir / f"{strategy}_detailed.json").write_text(
                '{"error": "judge failed to parse; metrics recorded as NaN"}',
                encoding="utf-8")

    print("\n=== RESULTS ===")
    def _fmt(v) -> str:
        return "----" if v is None else f"{v:.3f}"

    for strategy, s in summary.items():
        print(f"{strategy:9} "
              f"faith={_fmt(s['faithfulness'])} rel={_fmt(s['answer_relevancy'])} "
              f"prec={_fmt(s['context_precision'])} recall={_fmt(s['context_recall'])}")
    print(f"\nsaved -> {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
