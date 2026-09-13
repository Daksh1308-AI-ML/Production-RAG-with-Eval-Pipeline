"""Generate data/evaluation/eval_dataset.json.

Ground truths are written from figures verbatim-extracted from the 10-K
filings in data/processed/filings.json (see git history / session notes).
reference_contexts are literal sentences lifted from the filings so RAGAS
context_precision/recall have a measurable source of truth.
"""

import json
from pathlib import Path

from src.config import config

HERE = Path(__file__).resolve().parent
OUT = config.evaluation_dir / "eval_dataset.json"

QA = [
    # ---------- AAPL ----------
    {
        "question": "What were Apple's total net sales in fiscal 2025?",
        "ground_truth": "Apple's total net sales in fiscal 2025 were $416,161 million.",
        "reference_contexts": ["Total net sales $ 416,161 % $ 391,035 % $ 383,285"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What were Apple's total net sales in fiscal 2024?",
        "ground_truth": "Apple's total net sales in fiscal 2024 were $391,035 million.",
        "reference_contexts": ["Total net sales $ 416,161 % $ 391,035 % $ 383,285"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "How much revenue did Apple's Services business generate in fiscal 2025?",
        "ground_truth": "Apple's Services net sales in fiscal 2025 were $109,158 million.",
        "reference_contexts": ["Services net sales 109,158 % 96,169 % 85,200 Total net sales $ 416,161 % $ 391,035 % $ 383,285"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What drove the increase in Apple's Services net sales in fiscal 2025?",
        "ground_truth": "Services net sales increased primarily due to higher net sales from advertising, the App Store, and cloud services.",
        "reference_contexts": ["Services Services net sales increased during 2025 compared to 2024 primarily due to higher net sales from advertising, the App Store and cloud services."],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "mdna", "difficulty": "medium", "question_type": "analytical"},
    },
    {
        "question": "What factors affect Apple's total net sales and global supply chain?",
        "ground_truth": "Apple's global supply chain is large and complex, and a majority of the company's supplier facilities are located in Asia, exposing net sales to competition and supply chain risks.",
        "reference_contexts": ["total net sales. In addition, the Company's global supply chain is large and complex and a majority of the Company's supplier facilities are"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "risk_factors", "difficulty": "hard", "question_type": "analytical"},
    },

    # ---------- MSFT ----------
    {
        "question": "How much total revenue did Microsoft generate in fiscal 2026?",
        "ground_truth": "Microsoft's total revenue in fiscal 2026 was $331,839 million.",
        "reference_contexts": ["Total revenue 331,839 281,724 245,122"],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "How much revenue did Microsoft cloud generate in fiscal 2026?",
        "ground_truth": "Microsoft's cloud revenue increased 27% to $214.4 billion in fiscal 2026.",
        "reference_contexts": ["Cloud revenue increased 27% to $214.4 billion."],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "mdna", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What operating segments does Microsoft report?",
        "ground_truth": "Microsoft reports three segments: Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.",
        "reference_contexts": ["Intelligent Cloud, and More Personal Computing. Our segments provide management with a comprehensive financial view of our key businesses."],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "business", "difficulty": "easy", "question_type": "summary"},
    },
    {
        "question": "How fast is Microsoft's commercial remaining performance obligation growing?",
        "ground_truth": "Microsoft's commercial remaining performance obligation increased 84% to $678 billion as of June 30, 2026.",
        "reference_contexts": ["Commercial remaining performance obligation increased 84% to $678 billion."],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "mdna", "difficulty": "medium", "question_type": "analytical"},
    },
    {
        "question": "How did Microsoft's net income perform in fiscal 2026 on an adjusted basis?",
        "ground_truth": "Microsoft's adjusted (non-GAAP) net income was $128,786 million in fiscal 2026, up 22% from $105,452 million in fiscal 2025.",
        "reference_contexts": ["net income (non-GAAP) 128,786 105,452 22% Adjusted diluted earnings per share (non-GAAP) 17.28 14.13 22%"],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "financial_statements", "difficulty": "hard", "question_type": "factual"},
    },

    # ---------- GOOGL ----------
    {
        "question": "What were Alphabet's total revenues in fiscal 2025?",
        "ground_truth": "Alphabet's total revenues in fiscal 2025 were $402,836 million.",
        "reference_contexts": ["Total revenues $ 307,394 % $ 350,018 % $ 402,836 %"],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "How did Google's advertising revenue change in fiscal 2025?",
        "ground_truth": "Google Services revenues increased $37.8 billion (12%) and Google Cloud revenues increased $15.5 billion (36%), while Google Search & other revenues increased $26.4 billion.",
        "reference_contexts": ["Google Services revenues of $37.8 billion, or 12%, and an increase in Google Cloud revenues of $15.5 billion, or 36%.",
                               "Google Search & other revenues increased $26.4 billion from 2024 to 2025."],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "mdna", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "What segments does Google use to report revenue?",
        "ground_truth": "Google reports Google Services and Google Cloud segments; Google Services consists of Google advertising as well as subscriptions, platforms, and devices, and includes Google Search & other.",
        "reference_contexts": ["Google Services Google Services revenues consist of Google advertising as well as Google subscriptions, platforms, and devices revenues."],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "business", "difficulty": "medium", "question_type": "summary"},
    },
    {
        "question": "How fast did Google Cloud revenue grow in fiscal 2025?",
        "ground_truth": "Google Cloud revenues increased $15.5 billion (36%) from 2024 to 2025, driven primarily by growth in Google Cloud Platform.",
        "reference_contexts": ["Google Cloud revenues increased $15.5 billion from 2024 to 2025, primarily driven by growth in Google Cloud Platform largely from infrastructure and platform services."],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "mdna", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What portion of Alphabet's international revenues is non-U.S.?",
        "ground_truth": "International revenues accounted for approximately 52% of Alphabet's consolidated revenues.",
        "reference_contexts": ["International revenues accounted for approximately 52% of consolidated revenues."],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "mdna", "difficulty": "medium", "question_type": "factual"},
    },

    # ---------- AMZN ----------
    {
        "question": "What were Amazon's total net sales in fiscal 2025?",
        "ground_truth": "Amazon's total net sales in fiscal 2025 were $716,924 million.",
        "reference_contexts": ["Total net sales $ 716,924 $ 637,959 $ 574,785"],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What operating segments does Amazon report?",
        "ground_truth": "Amazon reports three segments: North America, International, and AWS.",
        "reference_contexts": ["We have organized our operations into three segments: North America, International, and AWS."],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "business", "difficulty": "easy", "question_type": "summary"},
    },
    {
        "question": "How much net sales did Amazon Web Services generate in fiscal 2025?",
        "ground_truth": "AWS generated net sales of $128,725 million in fiscal 2025.",
        "reference_contexts": ["AWS Net sales $ 90,757 $ 107,556 $ 128,725"],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What was AWS operating income in fiscal 2025?",
        "ground_truth": "AWS operating income was $45,606 million in fiscal 2025.",
        "reference_contexts": ["AWS Net sales $ 90,757 $ 107,556 $ 128,725 Operating expenses 66,126 67,722 83,119 Operating income $ 24,631 $ 39,834 $ 45,606"],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "How did Amazon's North America segment perform in fiscal 2025?",
        "ground_truth": "North America net sales grew to $426,305 million in fiscal 2025 (from $387,497M in 2024) with operating income of $29,619 million, driven by increased unit sales and advertising sales.",
        "reference_contexts": ["North America Net sales $ 352,828 $ 387,497 $ 426,305 Operating expenses 337,951 362,530 396,686 Operating income $ 14,877 $ 24,967 $ 29,619"],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "financial_statements", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "How much did Amazon's International segment contribute to net sales in fiscal 2025?",
        "ground_truth": "Amazon's International segment generated net sales of $161,894 million in fiscal 2025, making up 23% of consolidated net sales.",
        "reference_contexts": ["International Net sales $ 131,200 $ 142,906 $ 161,894",
                               "net sales from our International segment accounted for 23% of our consolidated revenues."],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },

    # ---------- NVDA ----------
    {
        "question": "What was NVIDIA's total revenue in fiscal 2026?",
        "ground_truth": "NVIDIA's total revenue in fiscal 2026 was $215,938 million, up 65% from fiscal 2025.",
        "reference_contexts": ["Revenue $ 215,938 $ 130,497 Up 65% Gross margin 71.1 %"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What was NVIDIA's net income in fiscal 2026?",
        "ground_truth": "NVIDIA's net income in fiscal 2026 was $120,067 million, up 65% from $72,880 million in fiscal 2025.",
        "reference_contexts": ["Net income $ 120,067 $ 72,880 Up 65%"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "How much revenue did NVIDIA's Data Center (Compute & Networking) segment generate in fiscal 2026?",
        "ground_truth": "NVIDIA's Compute & Networking segment generated revenue of $193,479 million in fiscal 2026, up from $116,193 million in fiscal 2025.",
        "reference_contexts": ["Compute & Networking Total Graphics Revenue $ 193,479 $ 22,459 $ 215,938"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "How much revenue did NVIDIA's Graphics segment generate in fiscal 2026?",
        "ground_truth": "NVIDIA's Graphics segment generated revenue of $22,459 million in fiscal 2026.",
        "reference_contexts": ["Compute & Networking Total Graphics Revenue $ 193,479 $ 22,459 $ 215,938"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "How much did NVIDIA pay in cash dividends in fiscal 2026?",
        "ground_truth": "NVIDIA paid cash dividends to its shareholders of $974 million in fiscal 2026.",
        "reference_contexts": ["In fiscal year 2026, we paid cash dividends to our shareholders of $974 million."],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },

    # ---------- Comparative ----------
    {
        "question": "Compare the total revenue of Apple and Microsoft in their most recent fiscal years.",
        "ground_truth": "Apple's fiscal 2025 net sales were $416,161 million while Microsoft's fiscal 2026 revenue was $331,839 million, making Apple larger by revenue.",
        "reference_contexts": ["Total net sales $ 416,161 % $ 391,035 % $ 383,285", "Total revenue 331,839 281,724 245,122"],
        "metadata": {"company": "AAPL,MSFT", "filing_year": 2026, "section": "financial_statements", "difficulty": "medium", "question_type": "comparative"},
    },
    {
        "question": "Which company had higher operating income in its most recent fiscal year: Amazon or Microsoft?",
        "ground_truth": "Microsoft reported a higher gross margin base and cloud-driven growth, but Amazon's AWS segment operating income of $45,606 million in fiscal 2025 is a key driver; Microsoft does not disclose a directly comparable segment figure in the extracted data, so a side-by-side operating income comparison is not fully supported by the context.",
        "reference_contexts": ["AWS Net sales $ 90,757 $ 107,556 $ 128,725 Operating income $ 24,631 $ 39,834 $ 45,606"],
        "metadata": {"company": "AMZN,MSFT", "filing_year": 2026, "section": "financial_statements", "difficulty": "hard", "question_type": "comparative"},
    },
    {
        "question": "How do NVIDIA and Google differ in how they report segment revenue?",
        "ground_truth": "NVIDIA reports Compute & Networking and Graphics segments, while Google reports Google Services and Google Cloud segments built around advertising and cloud.",
        "reference_contexts": [
            "Compute & Networking Total Graphics Revenue $ 193,479 $ 22,459 $ 215,938",
            "Google Services Google Services revenues consist of Google advertising as well as Google subscriptions, platforms, and devices revenues.",
        ],
        "metadata": {"company": "NVDA,GOOGL", "filing_year": 2026, "section": "business", "difficulty": "hard", "question_type": "comparative"},
    },
    {
        "question": "Which company serves as the clearest benchmark for comparing cloud growth across Microsoft and Amazon?",
        "ground_truth": "Microsoft's cloud revenue grew 27% to $214.4 billion while Amazon's AWS segment grew net sales from $107,556 million to $128,725 million in fiscal 2025.",
        "reference_contexts": ["Cloud revenue increased 27% to $214.4 billion.",
                               "AWS Net sales $ 90,757 $ 107,556 $ 128,725"],
        "metadata": {"company": "MSFT,AMZN", "filing_year": 2026, "section": "mdna", "difficulty": "hard", "question_type": "comparative"},
    },
]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(QA, f, indent=2, ensure_ascii=False)

    # Self-check: every question has ground_truth + metadata
    assert all(item.get("question") and item.get("ground_truth") for item in QA), "missing q/gt"
    types = {}
    for item in QA:
        t = item["metadata"].get("question_type", "?")
        types[t] = types.get(t, 0) + 1
    print(f"wrote {len(QA)} QA pairs -> {OUT}")
    print("by type:", types)
    assert len(QA) >= 25, "not enough QA pairs"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())