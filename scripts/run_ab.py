"""Resilient A/B evaluation runner with per-sample checkpointing."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.rag import RAGPipeline
from src.chunker import DocumentChunker
from src.eval import RAGEvaluator, LocalJudgeEvaluator, _METRICS

LOCAL_METRICS = {"faithfulness", "context_recall"}


def load_dataset(dataset_path):
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    return data


def build_chunks():
    with open(config.processed_dir / "filings.json", encoding="utf-8") as f:
        filings = json.load(f)
    return DocumentChunker().chunk_documents(filings)


def build_pipelines(chunks, strategies):
    pipelines = {}
    for name in strategies:
        pipe = RAGPipeline(strategy=name)
        pipe.initialize(chunks)
        pipelines[name] = pipe
    return pipelines


def _resp_path(out_dir, strategy):
    return out_dir / f"responses_{strategy}.json"


def _score_path(out_dir, strategy, metric):
    return out_dir / f"scores_{strategy}_{metric}.json"


def _load_json(path, default):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def generate_responses(strategy, pipe, dataset, out_dir, retries=3):
    path = _resp_path(out_dir, strategy)
    cache = _load_json(path, [])
    cached_qs = {r["question"] for r in cache}
    missing = [item for item in dataset if item["question"] not in cached_qs]

    if not missing:
        print(f"  [{strategy}] all {len(cache)} responses cached", flush=True)
        return cache

    print(f"  [{strategy}] {len(cached_qs)} cached, {len(missing)} to generate", flush=True)

    for i, item in enumerate(missing):
        q = item["question"]
        ok = False
        for attempt in range(1, retries + 1):
            try:
                response = pipe.query(q)
                sources = getattr(response, "sources", None) or []
                contexts = [
                    s.get("content") if isinstance(s, dict)
                    else getattr(s, "page_content", str(s))
                    for s in sources
                ]
                cache.append({
                    "question": q,
                    "response": getattr(response, "answer", ""),
                    "retrieved_contexts": contexts,
                })
                _save_json(path, cache)
                cached_qs.add(q)
                print(f"  [{strategy}] {len(cached_qs)}/{len(dataset)} "
                      f"ok ({getattr(response, 'latency_ms', 0):.0f}ms)", flush=True)
                ok = True
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  [{strategy}] attempt {attempt}/{retries} failed "
                      f"for '{q[:50]}...': {type(e).__name__}: {e}", flush=True)
                if attempt < retries:
                    print(f"  [{strategy}] retrying in {wait}s...", flush=True)
                    time.sleep(wait)
        if not ok:
            print(f"  [{strategy}] SKIPPED '{q[:50]}...' after {retries} attempts", flush=True)
            cache.append({
                "question": q,
                "response": None,
                "retrieved_contexts": None,
                "_failed": True,
            })
            _save_json(path, cache)

    return cache


def score_metric(strategy, metric, evaluator, local_evaluator, dataset,
                 responses, out_dir, sleep_seconds=5, retries=3):
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.run_config import RunConfig
    from ragas import evaluate as ragas_evaluate

    path = _score_path(out_dir, strategy, metric)
    state = _load_json(path, {"samples": []})
    samples = state["samples"]
    scored_qs = {s["question"] for s in samples if s.get("score") is not None}

    resp_by_q = {r["question"]: r for r in responses}
    unscored = [
        item for item in dataset
        if item["question"] not in scored_qs
    ]

    if not unscored:
        print(f"  [{strategy}/{metric}] all {len(samples)} scored", flush=True)
        return samples

    print(f"  [{strategy}/{metric}] {len(scored_qs)} cached, "
          f"{len(unscored)} to score", flush=True)

    metric_obj = _METRICS[metric]
    run_config = RunConfig(max_workers=1, timeout=600, max_retries=5)

    for i, item in enumerate(unscored):
        q = item["question"]
        resp = resp_by_q.get(q, {})

        if resp.get("_failed") or resp.get("response") is None:
            samples.append({"question": q, "score": None})
            _save_json(path, {"samples": samples})
            n_done = len(samples)
            print(f"  [{strategy}/{metric}] {n_done}/{len(dataset)} "
                  f"skipped (no response)", flush=True)
            continue

        score = None
        for attempt in range(1, retries + 1):
            try:
                sample = SingleTurnSample(
                    user_input=q,
                    reference=item.get("ground_truth") or item.get("reference", ""),
                    reference_contexts=item.get("reference_contexts") or item.get("contexts"),
                    response=resp["response"],
                    retrieved_contexts=resp.get("retrieved_contexts", []),
                )
                ds = EvaluationDataset(samples=[sample])

                if metric in LOCAL_METRICS:
                    llm = local_evaluator.judge_llm
                    embeddings = None
                else:
                    llm = evaluator.judge_llm
                    embeddings = evaluator.judge_embeddings

                result = ragas_evaluate(
                    dataset=ds,
                    metrics=[metric_obj],
                    llm=llm,
                    embeddings=embeddings,
                    run_config=run_config,
                    raise_exceptions=True,
                )
                result_df = result.to_pandas()
                col = metric
                if col in result_df.columns and len(result_df) > 0:
                    val = result_df[col].iloc[0]
                    if val is not None:
                        score = float(val)
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  [{strategy}/{metric}] attempt {attempt}/{retries} "
                      f"failed for '{q[:40]}...': {type(e).__name__}: {e}", flush=True)
                if attempt < retries:
                    print(f"  [{strategy}/{metric}] retrying in {wait}s...", flush=True)
                    time.sleep(wait)

        samples.append({"question": q, "score": score})
        _save_json(path, {"samples": samples})

        n_scored = sum(1 for s in samples if s.get("score") is not None)
        n_null = sum(1 for s in samples if s.get("score") is None)
        print(f"  [{strategy}/{metric}] {n_scored + n_null}/{len(dataset)} "
              f"(scored={n_scored}, null={n_null})", flush=True)

        if i < len(unscored) - 1:
            time.sleep(sleep_seconds)

    return samples


def aggregate(out_dir, strategies, metrics):
    summary = {}
    for strategy in strategies:
        summary[strategy] = {}
        for metric in metrics:
            path = _score_path(out_dir, strategy, metric)
            state = _load_json(path, {"samples": []})
            scores = [s["score"] for s in state["samples"] if s.get("score") is not None]
            summary[strategy][metric] = round(sum(scores) / len(scores), 4) if scores else None

    _save_json(out_dir / "summary.json", summary)

    with open(out_dir / "per_strategy_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy"] + metrics)
        for strategy in strategies:
            row = [strategy]
            for metric in metrics:
                v = summary[strategy].get(metric)
                row.append("" if v is None else f"{v:.4f}")
            writer.writerow(row)

    return summary


def print_table(summary, strategies, metrics):
    print("\n=== RESULTS ===")
    header = f"{'strategy':9} " + " ".join(f"{m[:5]:>7}" for m in metrics)
    print(header)
    for strategy in strategies:
        vals = []
        for metric in metrics:
            v = summary.get(strategy, {}).get(metric)
            vals.append("----" if v is None else f"{v:.4f}")
        print(f"{strategy:9} " + " ".join(f"{v:>7}" for v in vals))


def main():
    parser = argparse.ArgumentParser(description="Resilient A/B RAG evaluation")
    parser.add_argument("--dataset", default=str(Path("data/evaluation/ab_stratified.json")))
    parser.add_argument("--strategies", default="baseline,hybrid,rerank,full")
    parser.add_argument("--metrics", default="faithfulness,answer_relevancy,context_precision,context_recall")
    parser.add_argument("--out", default="data/evaluation/results_ab")
    parser.add_argument("--sleep", type=int, default=5)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in _METRICS]
    if unknown:
        parser.error(f"unknown metrics: {unknown}")

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"loading dataset: {dataset_path}", flush=True)
    dataset = load_dataset(dataset_path)
    print(f"  {len(dataset)} samples", flush=True)

    print("building chunks...", flush=True)
    chunks = build_chunks()
    print(f"  {len(chunks)} chunks", flush=True)

    print(f"building pipelines: {strategies}", flush=True)
    pipelines = build_pipelines(chunks, strategies)

    evaluator = RAGEvaluator()
    local_evaluator = LocalJudgeEvaluator()

    print("\n--- Phase 1: Response generation ---", flush=True)
    all_responses = {}
    for strategy in strategies:
        print(f"\n[{strategy}]", flush=True)
        all_responses[strategy] = generate_responses(
            strategy, pipelines[strategy], dataset, out_dir
        )

    print(f"\n--- Phase 2: Metric scoring (sleep={args.sleep}s) ---", flush=True)
    for strategy in strategies:
        for metric in metrics:
            print(f"\n[{strategy}/{metric}]", flush=True)
            score_metric(
                strategy, metric, evaluator, local_evaluator, dataset,
                all_responses[strategy], out_dir,
                sleep_seconds=args.sleep,
            )

    print("\n--- Phase 3: Aggregation ---", flush=True)
    summary = aggregate(out_dir, strategies, metrics)
    print_table(summary, strategies, metrics)

    elapsed = time.time() - t0
    print(f"\ntotal: {elapsed:.0f}s", flush=True)
    print(f"saved -> {out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
