"""RAGAS evaluation pipeline."""

import json
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

from .config import config


class RAGEvaluator:
    """Evaluate RAG system with RAGAS."""

    def __init__(self):
        self.judge_llm = ChatOllama(
            model=config.eval.judge_model,
            temperature=0,
            base_url=config.ollama.base_url
        )
        self.judge_embeddings = OllamaEmbeddings(
            model=config.ollama.embedding_model,
            base_url=config.ollama.base_url
        )

    def load_eval_dataset(self) -> EvaluationDataset:
        """Load evaluation dataset from a JSON file."""
        dataset_path = Path(config.eval.eval_dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found at {dataset_path}. "
                "Generate it first."
            )

        with open(dataset_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and "items" in data:
            data = data["items"]

        samples = [
            SingleTurnSample(
                user_input=item.get("question")
                or item.get("user_input")
                or item.get("query", ""),
                reference=item.get("ground_truth")
                or item.get("reference"),
                reference_contexts=item.get("contexts")
                or item.get("reference_contexts"),
            )
            for item in data
        ]

        return EvaluationDataset.from_list(
            SingleTurnSample(
                user_input=s.user_input,
                reference=s.reference,
                reference_contexts=s.reference_contexts,
            )
            for s in samples
        )

    def run_evaluation(
        self,
        rag_pipeline,
        eval_dataset: Optional[EvaluationDataset] = None
    ) -> Dict:
        """Run full RAGAS evaluation."""
        if eval_dataset is None:
            eval_dataset = self.load_eval_dataset()

        eval_ds = self._generate_responses(rag_pipeline, eval_dataset)

        # Run RAGAS evaluation
        result = evaluate(
            dataset=eval_ds,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=self.judge_llm,
            embeddings=self.judge_embeddings,
            raise_exceptions=True
        )

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
        return EvaluationDataset.from_list(generated)

    def compare_strategies(
        self,
        pipelines: Dict[str, object],
        eval_dataset: Optional[EvaluationDataset] = None
    ) -> Dict:
        """Compare multiple RAG strategies."""
        results = {}
        for name, pipeline in pipelines.items():
            results[name] = self.run_evaluation(pipeline, eval_dataset)
        return results
