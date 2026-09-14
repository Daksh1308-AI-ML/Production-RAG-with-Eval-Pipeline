"""Generate data/evaluation/eval_dataset.json.

Ground truths are written from figures verbatim-extracted from the 10-K
filings in data/processed/filings.json (see git history / session notes).
reference_contexts are literal sentences lifted from the filings so RAGAS
context_precision/recall have a measurable source of truth.
"""

import json
import re
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
        "reference_contexts": ["Services (1) 109,158 % 96,169 % 85,200 Total net sales $ 416,161 % $ 391,035 % $ 383,285"],
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
        "reference_contexts": ["total net sales. In addition, the Company\u2019s global supply chain is large and complex and a majority of the Company\u2019s supplier facilities, including manufacturing and assembly sites, are located outside the U.S."],
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
        "reference_contexts": ["International revenues accounted for approximately 52% of consolidated revenues in 2025."],
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
        "reference_contexts": ["Compute & Networking $ 193,479 $ 116,193 $ 77,286 %"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "How much revenue did NVIDIA's Graphics segment generate in fiscal 2026?",
        "ground_truth": "NVIDIA's Graphics segment generated revenue of $22,459 million in fiscal 2026.",
        "reference_contexts": ["Graphics 22,459 14,304 8,155 %"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "How much did NVIDIA pay in cash dividends in fiscal 2026?",
        "ground_truth": "NVIDIA paid cash dividends to its shareholders of $974 million in fiscal 2026.",
        "reference_contexts": ["In fiscal year 2026, we paid cash dividends to our shareholders of $974 million."],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },

    # ---------- AAPL (mined batch) ----------
    {
        "question": "What was Apple's total gross margin in fiscal 2025, in dollars and as a percentage of net sales?",
        "ground_truth": "Apple's total gross margin was $195,201 million in fiscal 2025, and its total gross margin percentage was 46.9% (compared to $180,683 million and 46.2% in fiscal 2024).",
        "reference_contexts": ["Total gross margin $ 195,201 $ 180,683 $ 169,148", "Total gross margin percentage 46.9 % 46.2 % 44.1 %"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What was Apple's operating income in fiscal 2025?",
        "ground_truth": "Apple's operating income was $133,050 million in fiscal 2025, up from $123,216 million in fiscal 2024 and $114,301 million in fiscal 2023.",
        "reference_contexts": ["Operating income 133,050 123,216 114,301"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What was Apple's net income in fiscal 2025?",
        "ground_truth": "Apple's net income was $112,010 million in fiscal 2025 (compared to $93,736 million in fiscal 2024 and $96,995 million in fiscal 2023).",
        "reference_contexts": ["Net income $ 112,010 $ 93,736 $ 96,995"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What was Apple's diluted earnings per share in fiscal 2025?",
        "ground_truth": "Apple's diluted earnings per share was $7.46 in fiscal 2025, compared to $6.08 in fiscal 2024 and $6.13 in fiscal 2023.",
        "reference_contexts": ["Earnings per share: Basic $ 7.49 $ 6.11 $ 6.16 Diluted $ 7.46 $ 6.08 $ 6.13"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "How much did Apple spend on research and development in fiscal 2025, and how did that compare with the prior year?",
        "ground_truth": "Apple's research and development expense was $34,550 million in fiscal 2025, up from $31,370 million in fiscal 2024 and $29,915 million in fiscal 2023.",
        "reference_contexts": ["Research and development 34,550 31,370 29,915"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "How much did Apple's cash, cash equivalents and marketable securities total as of September 27, 2025?",
        "ground_truth": "Apple's cash, cash equivalents and marketable securities totaled $132.4 billion as of September 27, 2025.",
        "reference_contexts": ["cash, cash equivalents and marketable securities, which totaled $132.4 billion as of September 27, 2025"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "mdna", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What was the cash generated by Apple's operating activities in fiscal 2025?",
        "ground_truth": "Cash generated by Apple's operating activities was $111,482 million in fiscal 2025, compared to $118,254 million in fiscal 2024 and $110,543 million in fiscal 2023.",
        "reference_contexts": ["Cash generated by operating activities 111,482 118,254 110,543"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What were Apple's total liabilities as of September 27, 2025?",
        "ground_truth": "Apple's total liabilities were $285,508 million as of September 27, 2025, compared to $308,030 million as of September 28, 2024.",
        "reference_contexts": ["Total liabilities 285,508 308,030"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "How did Apple's net sales in Greater China change between fiscal 2024 and fiscal 2025?",
        "ground_truth": "Apple's net sales in Greater China decreased from $66,952 million in fiscal 2024 to $64,377 million in fiscal 2025, a year-over-year decline of 4%; Greater China net sales had been $72,559 million in fiscal 2023.",
        "reference_contexts": ["Greater China 64,377 (4) % 66,952 (8) % 72,559"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "mdna", "difficulty": "medium", "question_type": "comparative"},
    },
    {
        "question": "What were Apple's iPhone net sales in fiscal 2025, and how did they compare with prior years?",
        "ground_truth": "Apple's iPhone net sales were $209,586 million in fiscal 2025, up from $201,183 million in fiscal 2024 and $200,583 million in fiscal 2023.",
        "reference_contexts": ["iPhone $ 209,586 $ 201,183 $ 200,583"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "mdna", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What were Apple's net sales in the Americas in fiscal 2025?",
        "ground_truth": "Apple's net sales in the Americas were $178,353 million in fiscal 2025, up from $167,045 million in fiscal 2024 and $162,560 million in fiscal 2023.",
        "reference_contexts": ["Americas $ 178,353 % $ 167,045 % $ 162,560"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "mdna", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What drove the increase in Apple's iPhone net sales during fiscal 2025?",
        "ground_truth": "Apple's iPhone net sales increased during fiscal 2025 compared to fiscal 2024 due to higher net sales of Pro models.",
        "reference_contexts": ["iPhone net sales increased during 2025 compared to 2024 due to higher net sales of Pro models"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "mdna", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "What drove the year-over-year decrease in Apple's Greater China net sales during fiscal 2025?",
        "ground_truth": "Greater China net sales decreased during fiscal 2025 compared to fiscal 2024 primarily due to lower net sales of iPhone, partially offset by higher net sales of Mac.",
        "reference_contexts": ["Greater China net sales decreased during 2025 compared to 2024 primarily due to lower net sales of iPhone, partially offset by higher net sales of Mac"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "mdna", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "What drove the growth in Apple's research and development expense during fiscal 2025?",
        "ground_truth": "The growth in Apple's research and development expense during fiscal 2025 compared to fiscal 2024 was primarily driven by increases in headcount-related expenses and infrastructure-related costs.",
        "reference_contexts": ["The growth in R&D expense during 2025 compared to 2024 was primarily driven by increases in headcount-related expenses and infrastructure-related costs"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "mdna", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "How does the Business section of Apple's fiscal 2025 Form 10-K characterize the competitive environment for its products and services?",
        "ground_truth": "Apple states that the markets for its products and services are highly competitive and are characterized by aggressive price competition, downward pressure on gross margins, continual improvement in product performance, and price sensitivity on the part of consumers and businesses.",
        "reference_contexts": ["aggressive price competition, downward pressure on gross margins, continual improvement in product performance, and price sensitivity on the part of consumers and businesses"],
        "metadata": {"company": "AAPL", "filing_year": 2025, "section": "business", "difficulty": "hard", "question_type": "factual"},
    },

    # ---------- MSFT (mined batch) ----------
    {
        "question": "What was Microsoft's operating income in fiscal 2026?",
        "ground_truth": "Microsoft's operating income for fiscal 2026 was $155,237 million, compared with $128,528 million in fiscal 2025 and $109,433 million in fiscal 2024.",
        "reference_contexts": ["Operating income 155,237 128,528 109,433"],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What was Microsoft's net income in fiscal 2026?",
        "ground_truth": "Microsoft's net income for fiscal 2026 was $133,749 million, an increase of 31% from $101,832 million in fiscal 2025 and up from $88,136 million in fiscal 2024.",
        "reference_contexts": ["Net income 133,749 101,832 31%", "Net income $ 133,749 $ 101,832 $ 88,136"],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What was Microsoft's diluted earnings per share in fiscal 2026?",
        "ground_truth": "Microsoft's diluted earnings per share for fiscal 2026 was $17.95, up 32% from $13.64 in fiscal 2025 and up from $11.80 in fiscal 2024.",
        "reference_contexts": ["Diluted earnings per share 17.95 13.64 32%", "Diluted $ 17.95 $ 13.64 $ 11.80"],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What was Microsoft's gross margin in fiscal 2026, and how did it change from fiscal 2025?",
        "ground_truth": "Microsoft's gross margin for fiscal 2026 was $225,465 million, an increase of $31.6 billion or 16% from $193,893 million in fiscal 2025, with growth across each of our segments.",
        "reference_contexts": ["Gross margin 225,465 193,893 16%", "Gross margin increased $31.6 billion or 16% with growth across each of our segments."],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "mdna", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What were Microsoft's research and development expenses in fiscal 2026, and how did they change from fiscal 2025?",
        "ground_truth": "In fiscal 2026, Microsoft's research and development expenses were $35,562 million, up $3.1 billion or 9% from $32,488 million in fiscal 2025, driven by continued investments in compute capacity, AI talent, and data to support product development.",
        "reference_contexts": ["Research and development $ 35,562 $ 32,488 9%", "Research and development expenses increased $3.1 billion or 9%"],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "mdna", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What were Microsoft's cash and cash equivalents and short-term investments as of June 30, 2026, compared with fiscal 2025 year-end?",
        "ground_truth": "As of June 30, 2026, Microsoft's cash and cash equivalents were $20,935 million and short-term investments were $55,908 million, totaling $76,843 million, down from $94,565 million as of June 30, 2025.",
        "reference_contexts": ["Cash and cash equivalents $ 20,935 $ 30,242 Short-term investments 55,908 64,323", "Total cash, cash equivalents, and short-term investments 76,843 94,565"],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What was Microsoft's net cash from operations in fiscal 2026?",
        "ground_truth": "Microsoft's net cash from operations for fiscal 2026 was $182,935 million, compared with $136,162 million in fiscal 2025 and $118,548 million in fiscal 2024.",
        "reference_contexts": ["Net cash from operations 182,935 136,162 118,548"],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What was the revenue of Microsoft's Intelligent Cloud segment in fiscal 2026, and how did it compare with fiscal 2025?",
        "ground_truth": "In fiscal 2026, Microsoft's Intelligent Cloud segment generated revenue of $137,791 million, up $31.5 billion or 30% from $106,265 million in fiscal 2025 and from $87,464 million in fiscal 2024.",
        "reference_contexts": ["Intelligent Cloud Revenue $ 137,791 $ 106,265 $ 87,464", "Revenue increased $31.5 billion or 30%."],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "In fiscal 2026, did Microsoft earn more revenue from the United States or from international (other countries) customers?",
        "ground_truth": "In fiscal 2026, Microsoft's United States revenue was $170,794 million, which exceeded international (other countries) revenue of $161,045 million.",
        "reference_contexts": ["United States (a) $ 170,794 $ 144,546 $ 124,704", "Other countries 161,045 137,178 120,418"],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "How did Microsoft's net income trend from fiscal 2024 to fiscal 2026?",
        "ground_truth": "Microsoft's net income trended up from $88,136 million in fiscal 2024 to $101,832 million in fiscal 2025, an increase of 16%, and then to $133,749 million in fiscal 2026, an increase of 31%.",
        "reference_contexts": ["Net income $ 133,749 $ 101,832 $ 88,136 31% 16%"],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "mdna", "difficulty": "medium", "question_type": "comparative"},
    },
    {
        "question": "What drove the 31% increase in Microsoft's net income in fiscal 2026?",
        "ground_truth": "Microsoft's net income rose 31% to $133,749 million in fiscal 2026 driven by revenue growth of $50.1 billion or 18% and a $26.7 billion or 21% increase in operating income; net income was also positively impacted by net gains from investments in OpenAI, which increased net income and diluted EPS by $5.0 billion and $0.67, respectively.",
        "reference_contexts": ["Revenue increased $50.1 billion or 18% driven by growth in Microsoft Cloud.",
                                "Operating income increased $26.7 billion or 21% driven by growth in Productivity and Business Processes and Intelligent Cloud.",
                                "Current year net income and diluted EPS were positively impacted by net gains from investments in OpenAI, which resulted in an increase in net income and diluted EPS of $5.0 billion and $0.67, respectively."],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "mdna", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "Why did Microsoft Cloud gross margin percentage decrease in fiscal 2026?",
        "ground_truth": "Microsoft Cloud gross margin percentage decreased to 66% in fiscal 2026, driven by continued investments in AI infrastructure and growing AI product usage, offset in part by efficiency gains in Azure and Microsoft 365 Commercial cloud.",
        "reference_contexts": ["Microsoft Cloud gross margin percentage decreased to 66% driven by continued investments in AI infrastructure and growing AI product usage, offset in part by efficiency gains in Azure and Microsoft 365 Commercial cloud.",
                                "Gross margin percentage decreased driven by the continued investments in AI infrastructure as well as sales mix shift to Azure, offset in part by efficiency gains in Azure."],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "mdna", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "Why did Microsoft's More Personal Computing revenue decline in fiscal 2026?",
        "ground_truth": "More Personal Computing revenue declined $597 million or 1% in fiscal 2026, driven by a $1.7 billion or 7% decrease in XBOX revenue stemming from declines in XBOX content and services and XBOX hardware, offset in part by a $1.3 billion or 9% increase in Search advertising revenue.",
        "reference_contexts": ["Revenue decreased $597 million or 1%.",
                                "XBOX revenue decreased $1.7 billion or 7% driven by declines in XBOX content and services and XBOX hardware.",
                                "Search advertising revenue increased $1.3 billion or 9%."],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "mdna", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "In Microsoft's fiscal 2026 Form 10-K, how much additional tax payment does the IRS seek in its Notices of Proposed Adjustment for tax years 2004 to 2013?",
        "ground_truth": "Microsoft discloses that it is under IRS audit and has received Notices of Proposed Adjustment (NOPAs) for tax years 2004 to 2013, in which the IRS is seeking an additional tax payment of $28.9 billion plus penalties and interest; the primary issues relate to intercompany transfer pricing, and Microsoft disagrees with the proposed adjustments and will contest them.",
        "reference_contexts": ["We are currently under IRS audit for prior tax years and have received Notices of Proposed Adjustment",
                                "In the NOPAs, the IRS is seeking an additional tax payment of $28.9 billion plus penalties and interest."],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "risk_factors", "difficulty": "hard", "question_type": "factual"},
    },
    {
        "question": "By how much did Microsoft's Azure and other cloud services revenue grow in fiscal 2026?",
        "ground_truth": "In fiscal 2026, Microsoft's Azure and other cloud services revenue grew 41% driven by demand for services across the platform with continued growth across all workloads, contributing to a $31.0 billion or 31% increase in Server products and cloud services revenue.",
        "reference_contexts": ["Azure and other cloud services revenue grew 41% driven by demand for services across the platform with continued growth across all workloads.",
                                "Server products and cloud services revenue increased $31.0 billion or 31% driven by Azure and other cloud services."],
        "metadata": {"company": "MSFT", "filing_year": 2026, "section": "mdna", "difficulty": "medium", "question_type": "factual"},
    },

    # ---------- GOOGL (mined batch) ----------
    {
        "question": "What was Alphabet (GOOGL)'s net income for fiscal year 2025?",
        "ground_truth": "Alphabet's net income for fiscal 2025 was $132,170 million, compared with $100,118 million in fiscal 2024 and $73,795 million in fiscal 2023.",
        "reference_contexts": ["Net income $ 73,795 $ 100,118 $ 132,170"],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What was Alphabet (GOOGL)'s diluted earnings per share for fiscal year 2025?",
        "ground_truth": "Alphabet's diluted net income per share for fiscal 2025 was $10.81, up from $8.04 in fiscal 2024 and $5.80 in fiscal 2023.",
        "reference_contexts": ["Diluted net income per share (Note 12) $ 5.80 $ 8.04 $ 10.81"],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What was Alphabet (GOOGL)'s total income from operations for fiscal year 2025?",
        "ground_truth": "Alphabet's total income from operations for fiscal 2025 was $129,039 million, up from $112,390 million in fiscal 2024 and $84,293 million in fiscal 2023.",
        "reference_contexts": ["Total income from operations $ 84,293 $ 112,390 $ 129,039"],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What was Google Cloud's segment revenue for Alphabet (GOOGL) in fiscal year 2025, and how did it change from fiscal 2024?",
        "ground_truth": "Google Cloud segment revenue for fiscal 2025 was $58,705 million, an increase of $15.5 billion (approximately 36%) from $43,229 million in fiscal 2024, primarily driven by growth in Google Cloud Platform largely from infrastructure and platform services.",
        "reference_contexts": ["Google Cloud 33,088 43,229 58,705", "Google Cloud revenues increased $15.5 billion from 2024 to 2025, primarily driven by growth in Google Cloud Platform largely from infrastructure and platform services."],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "In fiscal year 2025, which of Alphabet (GOOGL)'s two largest segments, Google Services or Google Cloud, generated more revenue, and by how much?",
        "ground_truth": "Google Services generated $342,721 million in fiscal 2025 revenue while Google Cloud generated $58,705 million, making Google Services larger by $284,016 million (about $284 billion) and roughly 5.8 times the size of Google Cloud.",
        "reference_contexts": ["Google Services total 304,930 342,721 Google Cloud 43,229 58,705"],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "comparative"},
    },
    {
        "question": "How much cash, cash equivalents, and marketable securities did Alphabet (GOOGL) hold as of December 31, 2025 (the end of fiscal year 2025)?",
        "ground_truth": "As of December 31, 2025, Alphabet held $126,843 million in total cash, cash equivalents, and marketable securities combined, up from $95,657 million at the end of fiscal 2024.",
        "reference_contexts": ["Total cash, cash equivalents, and marketable securities 95,657 126,843"],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What was Alphabet (GOOGL)'s net cash provided by operating activities for fiscal year 2025, and why did it move versus fiscal 2024?",
        "ground_truth": "Alphabet's net cash provided by operating activities for fiscal 2025 was $164,713 million, up from $125,299 million in fiscal 2024; the increase was due to higher cash received from customers, partially offset by higher cash payments for cost of revenues and operating expenses.",
        "reference_contexts": ["Net cash provided by operating activities $ 125,299 $ 164,713", "Net cash provided by operating activities increased from 2024 to 2025 due to an increase in cash received from customers, partially offset by an increase in cash payments for cost of revenues and operating expenses."],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What drove the growth in Alphabet (GOOGL)'s Google Search & other advertising revenues from fiscal 2024 to fiscal 2025?",
        "ground_truth": "Google Search & other revenues increased $26.4 billion from fiscal 2024 to fiscal 2025, driven by interrelated factors including increases in search queries from growth in user adoption and usage on mobile devices, growth in advertiser spending, and improvements in ad formats and delivery.",
        "reference_contexts": ["Google Search & other revenues increased $26.4 billion from 2024 to 2025.",
                                "The overall growth was driven by interrelated factors including increases in search queries resulting from growth in user adoption and usage on mobile devices; growth in advertiser spending; and improvements we have made in ad formats and delivery."],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "mdna", "difficulty": "medium", "question_type": "analytical"},
    },
    {
        "question": "What drove the growth in Alphabet (GOOGL)'s YouTube ads revenues from fiscal 2024 to fiscal 2025?",
        "ground_truth": "YouTube ads revenues increased $4.2 billion from fiscal 2024 to fiscal 2025, with growth driven by YouTube's direct response advertising products, followed by its brand advertising products, both of which benefited from increased spending by advertisers.",
        "reference_contexts": ["YouTube ads revenues increased $4.2 billion from 2024 to 2025.",
                                "The growth was driven by our direct response advertising products followed by our brand advertising products, both of which benefited from increased spending by our advertisers."],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "mdna", "difficulty": "medium", "question_type": "analytical"},
    },
    {
        "question": "How dependent is Alphabet (GOOGL) on online advertising, according to its fiscal year 2025 risk factors?",
        "ground_truth": "Alphabet generated more than 70% of its total revenues from online advertising in fiscal 2025, and its risk factors note that technologies that block ads online or affect its ability to personalize ads could harm its business.",
        "reference_contexts": ["We generated more than 70% of total revenues from online advertising in 2025."],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "risk_factors", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What were Alphabet (GOOGL)'s Google Search & other, YouTube ads, and Google Network advertising revenues for fiscal year 2025?",
        "ground_truth": "For fiscal 2025, Google Search & other advertising revenues were $224,532 million, YouTube ads revenues were $40,367 million, and Google Network revenues were $29,792 million, together making total Google advertising revenues of $294,691 million (compared with $264,590 million in fiscal 2024).",
        "reference_contexts": ["Google Search & other $ 198,084 $ 224,532", "YouTube ads 36,147 40,367 Google Network 30,359 29,792", "Google advertising 264,590 294,691"],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "financial_statements", "difficulty": "hard", "question_type": "factual"},
    },
    {
        "question": "Why did Alphabet (GOOGL)'s Google Cloud operating income grow so much between fiscal 2024 and fiscal 2025?",
        "ground_truth": "Google Cloud operating income increased $7.8 billion from fiscal 2024 to fiscal 2025, rising from $6,112 million to $13,910 million, primarily driven by an increase in revenues, partially offset by increases in usage costs for technical infrastructure and employee compensation expenses.",
        "reference_contexts": ["Google Cloud operating income increased $7.8 billion from 2024 to 2025.",
                                "The increase in operating income was primarily driven by an increase in revenues, partially offset by increases in usage costs for technical infrastructure and employee compensation expenses.",
                                "Google Cloud 1,716 6,112 13,910"],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "mdna", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "What were Alphabet (GOOGL)'s Other Bets revenue and operating loss for fiscal year 2025, and what notable charge affected the loss?",
        "ground_truth": "Other Bets generated $1,537 million of revenue in fiscal 2025, down from $1,648 million in fiscal 2024. Other Bets' operating loss widened by $3.1 billion to $7.5 billion for fiscal 2025, and that loss included a $2.1 billion employee compensation charge recognized in the fourth quarter for Waymo.",
        "reference_contexts": ["Google Cloud 43,229 58,705 Other Bets 1,648 1,537",
                                "Other Bets operating loss increased $3.1 billion from 2024 to 2025.",
                                "Other Bets operating loss of $7.5 billion for the year ended December 31, 2025 included a $2.1 billion employee compensation charge recognized in the fourth quarter for Waymo, primarily reflected in research and development expenses"],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "mdna", "difficulty": "hard", "question_type": "factual"},
    },
    {
        "question": "How did Alphabet (GOOGL)'s net income change from fiscal year 2024 to fiscal year 2025?",
        "ground_truth": "Alphabet's net income increased from $100,118 million in fiscal 2024 to $132,170 million in fiscal 2025, an increase of $32,052 million (about 32%), while income before income taxes grew from $119,815 million to $158,826 million over the same period.",
        "reference_contexts": ["Net income $ 73,795 $ 100,118 $ 132,170", "Income before income taxes 85,717 119,815 158,826"],
        "metadata": {"company": "GOOGL", "filing_year": 2025, "section": "financial_statements", "difficulty": "hard", "question_type": "comparative"},
    },

    # ---------- AMZN (mined batch) ----------
    {
        "question": "What was Amazon.com's net income for the fiscal year ended December 31, 2025?",
        "ground_truth": "For fiscal year 2025, Amazon.com reported net income of $77,670 million, compared with $59,248 million in fiscal 2024 and $30,425 million in fiscal 2023.",
        "reference_contexts": ["Net income $ 30,425 $ 59,248 $ 77,670", "Net income 30,425 59,248 77,670"],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What was Amazon.com's total consolidated operating income for fiscal year 2025?",
        "ground_truth": "Amazon.com's consolidated operating income for the fiscal year ended December 31, 2025 was $79,975 million ($80.0 billion), up from $68,593 million ($68.6 billion) in fiscal 2024.",
        "reference_contexts": ["Operating income 36,852 68,593 79,975", "Operating income was $68.6 billion and $80.0 billion for 2024 and 2025."],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What was Amazon.com's diluted earnings per share for fiscal year 2025?",
        "ground_truth": "Amazon.com's diluted earnings per share was $7.17 for fiscal 2025, up from $5.53 in fiscal 2024 and $2.90 in fiscal 2023.",
        "reference_contexts": ["Diluted earnings per share $ 2.90 $ 5.53 $ 7.17"],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "Did Amazon.com's gross margin improve or deteriorate from fiscal 2024 to fiscal 2025?",
        "ground_truth": "Amazon does not report a gross profit or gross margin line item, noting that operating income is a more meaningful measure due to the diversity of its product categories and services. Cost of sales was $326,288 million, or 51.1% of net sales, in fiscal 2024 and $356,414 million, or 49.7% of net sales, in fiscal 2025, which indicates the implied gross margin widened year over year.",
        "reference_contexts": ["Cost of sales $ 326,288 $ 356,414", "Cost of sales 51.1 % 49.7 %", "We believe that operating income is a more meaningful measure than gross profit and gross margin due to the diversity of our product categories and services."],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "mdna", "difficulty": "hard", "question_type": "comparative"},
    },
    {
        "question": "How much did Amazon.com spend on technology and infrastructure in fiscal year 2025, and how did it compare with prior years?",
        "ground_truth": "Amazon.com's Technology and infrastructure expense was $108,521 million in fiscal 2025, up 23% from $88,544 million in fiscal 2024 and compared with $85,622 million in fiscal 2023.",
        "reference_contexts": ["Technology and infrastructure 85,622 88,544 108,521", "The increase in technology and infrastructure costs in 2025, compared to the prior year, is primarily due to an increase in spending on infrastructure, including depreciation and amortization."],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What were Amazon.com's cash, cash equivalents, and marketable securities balances as of December 31, 2025?",
        "ground_truth": "As of December 31, 2025, Amazon.com held cash, cash equivalents, and marketable securities with a fair value of $123.0 billion, up from $101.2 billion as of December 31, 2024; this comprised cash and cash equivalents of $86,810 million and marketable securities of $36,219 million.",
        "reference_contexts": ["which, at fair value, were $101.2 billion and $123.0 billion as of December 31, 2024 and 2025.",
                                "Cash and cash equivalents $ 78,779 $ 86,810",
                                "Marketable securities 22,423 36,219"],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "mdna", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What was Amazon.com's net cash provided by operating activities for fiscal year 2025?",
        "ground_truth": "For fiscal 2025, Amazon.com's net cash provided by operating activities was $139,514 million, up from $115,877 million in fiscal 2024 and $84,946 million in fiscal 2023.",
        "reference_contexts": ["Net cash provided by (used in) operating activities 84,946 115,877 139,514", "Cash provided by (used in) operating activities was $115.9 billion and $139.5 billion in 2024 and 2025."],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "In fiscal 2025, how did Amazon.com's North America segment net sales compare with its International segment net sales, and how fast did each grow?",
        "ground_truth": "For fiscal 2025, Amazon.com's North America segment net sales were $426,305 million, up 10% from $387,497 million in fiscal 2024, while its International segment net sales were $161,894 million, up 13% from $142,906 million in fiscal 2024.",
        "reference_contexts": ["North America $ 387,497 $ 426,305", "International 142,906 161,894", "North America sales increased 10% in 2025, compared to the prior year.", "International sales increased 13% in 2025, compared to the prior year."],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "mdna", "difficulty": "medium", "question_type": "comparative"},
    },
    {
        "question": "How did Amazon.com's free cash flow change from fiscal 2024 to fiscal 2025, and what drove the change?",
        "ground_truth": "Amazon.com's free cash flow declined from $38,219 million in fiscal 2024 to $11,194 million in fiscal 2025, because cash capital expenditures grew from $77.7 billion to $128.3 billion, primarily reflecting investments in technology infrastructure (the majority of which supports AWS business growth) and additional capacity to support the fulfillment network.",
        "reference_contexts": ["Free cash flow $ 38,219 $ 11,194",
                                "Cash capital expenditures were $77.7 billion, and $128.3 billion in 2024 and 2025, which primarily reflect investments in technology infrastructure (the majority of which is to support AWS business growth) and in additional capacity to support our fulfillment network, both of which we expect to increase in 2026."],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "mdna", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "What drove the increase in Amazon.com's North America segment operating income in fiscal 2025?",
        "ground_truth": "The increase in Amazon.com's North America operating income in fiscal 2025, from $24,967 million in 2024 to $29,619 million in 2025, was primarily due to increased unit sales and increased advertising sales, partially offset by increased fulfillment, technology and infrastructure, shipping, and other operating costs.",
        "reference_contexts": ["North America $ 24,967 $ 29,619",
                                "The increase in North America operating income in 2025, compared to the prior year, is primarily due to increased unit sales and increased advertising sales, partially offset by increased fulfillment, technology and infrastructure, shipping, and other operating costs."],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "mdna", "difficulty": "medium", "question_type": "analytical"},
    },
    {
        "question": "How many employees did Amazon.com have as of December 31, 2025?",
        "ground_truth": "As of December 31, 2025, Amazon.com employed approximately 1,576,000 full-time and part-time employees, supplemented by independent contractors and temporary personnel.",
        "reference_contexts": ["As of December 31, 2025, we employed approximately 1,576,000 full-time and part-time employees."],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "business", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What caused Amazon.com's other income (expense), net to swing from a loss in fiscal 2024 to a large gain in fiscal 2025?",
        "ground_truth": "Amazon.com's other income (expense), net swung from a net loss of $(2.3) billion in fiscal 2024, primarily from the marketable securities loss on its equity investment in Rivian Automotive, Inc., to a net gain of $15.2 billion in fiscal 2025, primarily from an upward adjustment for observable changes in price relating to its nonvoting preferred stock in Anthropic and the reclassification gains on Anthropic convertible notes that were converted to nonvoting preferred stock during 2025.",
        "reference_contexts": ["Other income (expense), net ( 2,250 ) 15,229",
                                "The net loss of $(2.3) billion in 2024 is primarily from the marketable securities loss from our equity investment in Rivian Automotive, Inc.",
                                "The net gain of $15.2 billion in 2025 is primarily from an upward adjustment for observable changes in price relating to our nonvoting Table of Contents preferred stock in Anthropic, and the reclassification adjustments for the gains on available-for-sale debt securities from the portions of our convertible notes investments in Anthropic that were converted to nonvoting preferred stock during 2025."],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "mdna", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "What significant one-time charges did Amazon.com record during fiscal 2025?",
        "ground_truth": "During Q3 2025, Amazon.com recorded $2.5 billion of expense related to the settlement of a lawsuit with the FTC, and during Q4 2025 it recorded $2.4 billion of expense related to settlements of a lawsuit and tax disputes, severance costs, and asset impairments. For all of fiscal 2025, it recorded approximately $2.7 billion of estimated severance costs related to planned role eliminations (of which $1.8 billion was recorded in Q3 2025 and $730 million in the fourth quarter) and approximately $1.3 billion of asset impairments (including $610 million in the fourth quarter).",
        "reference_contexts": ["During Q3 2025, we recorded $2.5 billion of expense related to the settlement of a lawsuit with the FTC.",
                                "During Q4 2025, we recorded $2.4 billion of expense related to settlements of a lawsuit and tax disputes, severance costs, and asset impairments.",
                                "For the year ended December 31, 2025, we recorded approximately $2.7 billion of estimated severance costs primarily related to planned role eliminations, of which $1.8 billion was recorded in Q3 2025 and $730 million was recorded in the fourth quarter.",
                                "For the year ended December 31, 2025, we recorded approximately $1.3 billion of asset impairments, of which $610 million was recorded in the fourth quarter, primarily consisting of property and equipment and operating leases related to physical stores."],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "financial_statements", "difficulty": "hard", "question_type": "factual"},
    },
    {
        "question": "How much revenue did Amazon.com generate from advertising services in fiscal year 2025?",
        "ground_truth": "Amazon.com's advertising services net sales were $68,635 million in fiscal 2025, up from $56,214 million in fiscal 2024 and $46,906 million in fiscal 2023.",
        "reference_contexts": ["Advertising services (4) 46,906 56,214 68,635", "Revenue is recognized as ads are delivered based on the number of clicks or impressions."],
        "metadata": {"company": "AMZN", "filing_year": 2025, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },

    # ---------- NVDA (mined batch) ----------
    {
        "question": "What was NVIDIA's gross margin percentage in fiscal 2026, and how did it change from fiscal 2025?",
        "ground_truth": "NVIDIA's gross margin was 71.1% in fiscal 2026, down 3.9 percentage points from 75.0% in fiscal 2025. Gross profit was $153,463 million on revenue of $215,938 million.",
        "reference_contexts": ["Revenue $ 215,938 $ 130,497 Up 65% Gross margin 71.1 % 75.0 % -3.9 pts",
                                "Gross margins decreased to 71.1% in fiscal year 2026 from 75.0% in fiscal year 2025"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "How did NVIDIA's diluted earnings per share change from fiscal 2025 to fiscal 2026?",
        "ground_truth": "NVIDIA's net income per diluted share rose from $2.94 in fiscal 2025 to $4.90 in fiscal 2026, an increase of 67%. Basic EPS was $4.93 in fiscal 2026.",
        "reference_contexts": ["Net income per diluted share $ 4.90 $ 2.94 Up 67%", "Net income per share: Basic $ 4.93 $ 2.97 $ 1.21 Diluted $ 4.90 $ 2.94 $ 1.19"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "easy", "question_type": "comparative"},
    },
    {
        "question": "How much net cash did NVIDIA generate from operating activities in fiscal 2026?",
        "ground_truth": "NVIDIA's net cash provided by operating activities was $102,718 million ($102.7 billion) in fiscal 2026, up from $64,089 million in fiscal 2025.",
        "reference_contexts": ["Net cash provided by operating activities $ 102,718 $ 64,089", "Cash provided by operating activities increased in fiscal year 2026 compared to fiscal year 2025, due to higher revenue."],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "easy", "question_type": "factual"},
    },
    {
        "question": "What was NVIDIA's operating income in fiscal 2026, and by how much did it grow versus fiscal 2025?",
        "ground_truth": "NVIDIA's operating income was $130,387 million in fiscal 2026, up 60% from $81,453 million in fiscal 2025.",
        "reference_contexts": ["Operating income $ 130,387 $ 81,453 Up 60%", "Operating income 130,387 81,453 32,972"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "How much did NVIDIA spend on research and development in fiscal 2026?",
        "ground_truth": "NVIDIA's research and development expense was $18,497 million in fiscal 2026, up $5,583 million from $12,914 million in fiscal 2025.",
        "reference_contexts": ["Research and development $ 18,497 $ 12,914 $ 5,583 %"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What were NVIDIA's cash balances and marketable securities as of January 25, 2026?",
        "ground_truth": "As of January 25, 2026, NVIDIA had cash, cash equivalents, and marketable securities of $62,556 million ($62.6 billion), comprising $10,605 million of cash and cash equivalents and $51,951 million of marketable securities, versus $43,210 million a year earlier.",
        "reference_contexts": ["Cash and cash equivalents $ 10,605 $ 8,589 Marketable securities 51,951 34,621 Cash, cash equivalents, and marketable securities $ 62,556 $ 43,210",
                                "As of January 25, 2026, we had $62.6 billion in cash, cash equivalents, and marketable securities."],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "Within NVIDIA's fiscal 2026 Data Center revenue, how much came from Compute versus Networking?",
        "ground_truth": "NVIDIA's Data Center revenue was $193,737 million in fiscal 2026, of which Compute revenue was $162,361 million and Networking revenue was $31,376 million. In fiscal 2025, the split was $102,196 million of Compute and $12,990 million of Networking on Data Center revenue of $115,186 million.",
        "reference_contexts": ["Data Center $ 193,737 $ 115,186 $ 47,525 Compute 162,361 102,196 38,950 Networking 31,376 12,990 8,575"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "What drove the sharp growth in NVIDIA's Data Center networking revenue in fiscal 2026?",
        "ground_truth": "NVIDIA's Data Center networking revenue grew 142% in fiscal 2026, driven by the introduction and continued ramp of NVLink compute fabric for GB200 and GB300 systems and the growth of Ethernet and InfiniBand platforms, while Data Center computing revenue grew 59% on demand for the Blackwell computing platform.",
        "reference_contexts": ["Revenue from Data Center networking grew 142% driven by the introduction and continued ramp of NVLink compute fabric for GB200 and GB300 systems and the growth of Ethernet and InfiniBand platforms.",
                                "Revenue from Data Center computing grew 59% driven by demand for our Blackwell computing platform."],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "mdna", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "Why did NVIDIA's gross margin decline in fiscal 2026 despite record revenue?",
        "ground_truth": "NVIDIA's gross margin fell to 71.1% in fiscal 2026 from 75.0% in fiscal 2025 because its business model transitioned from offering Hopper HGX systems to Blackwell full-scale datacenter solutions and it recorded a $4.5 billion charge associated with H20 excess inventory and purchase obligations. Provisions for inventory and excess inventory purchase obligations totaled $7.2 billion in fiscal 2026.",
        "reference_contexts": ["Gross margins decreased to 71.1% in fiscal year 2026 from 75.0% in fiscal year 2025 as our business model transitioned from offering Hopper HGX systems to Blackwell full-scale datacenter solutions",
                                "Provisions for inventory and excess inventory purchase obligations totaled $7.2 billion and $3.7 billion for fiscal years 2026 and 2025",
                                "including $4.5 billion associated with H20 excess inventory and purchase obligations for the first quarter of fiscal year 2026"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "mdna", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "What drove NVIDIA's strong revenue growth in fiscal 2026?",
        "ground_truth": "NVIDIA's revenue grew 65% to $215.9 billion in fiscal 2026, driven by the major platform shifts of accelerated computing and AI. Data Center revenue was up 68% from a year ago.",
        "reference_contexts": ["Data Center revenue for fiscal year 2026 was up 68% from a year ago. The strong year-on-year growth was driven by the major platform shifts \u2013 accelerated computing and AI.",
                                "Revenue for fiscal year 2026 was $215.9 billion, up 65% from a year ago."],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "mdna", "difficulty": "medium", "question_type": "analytical"},
    },
    {
        "question": "What was NVIDIA's automotive revenue in fiscal 2026 and what drove its growth?",
        "ground_truth": "NVIDIA's automotive revenue was $2,349 million in fiscal 2026, up 39% from a year ago (when it was $1,694 million), driven by continued adoption of its self-driving platforms.",
        "reference_contexts": ["Automotive revenue for fiscal year 2026 was up 39% from a year ago, driven by continued adoption of our self-driving platforms.",
                                "Gaming 16,042 11,350 10,447 Professional Visualization 3,191 1,878 1,553 Automotive 2,349 1,694 1,091"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "mdna", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "How many shares of its common stock did NVIDIA repurchase in fiscal 2026 and at what cost?",
        "ground_truth": "In fiscal 2026, NVIDIA repurchased 282 million shares of its common stock for $40.4 billion. On August 26, 2025, the Board of Directors approved an additional $60.0 billion in share repurchase authorization, and as of January 25, 2026, NVIDIA was authorized to repurchase up to $58.5 billion of common stock.",
        "reference_contexts": ["In fiscal year 2026, we repurchased 282 million shares of our common stock for $40.4 billion.",
                                "On August 26, 2025, our Board of Directors approved an additional $60.0 billion in share repurchase authorization, without expiration.",
                                "As of January 25, 2026, we were authorized, subject to certain specifications, to repurchase up to $58.5 billion of our common stock."],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "medium", "question_type": "factual"},
    },
    {
        "question": "How did NVIDIA's net income trend from fiscal 2024 through fiscal 2026?",
        "ground_truth": "NVIDIA's net income was $29,760 million in fiscal 2024, $72,880 million in fiscal 2025, and $120,067 million in fiscal 2026, more than quadrupling between fiscal 2024 and fiscal 2026.",
        "reference_contexts": ["Net income $ 120,067 $ 72,880 $ 29,760", "Income before income tax 141,450 84,026 33,818 Income tax expense 21,383 11,146 4,058"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "financial_statements", "difficulty": "hard", "question_type": "comparative"},
    },
    {
        "question": "Why did NVIDIA's other income, net increase so sharply in fiscal 2026?",
        "ground_truth": "NVIDIA's other income, net jumped from $1,034 million in fiscal 2025 to $9,022 million in fiscal 2026, and was primarily driven by unrealized gains in non-marketable and publicly-held equity securities, including gains from its previously announced investment in Intel's common stock.",
        "reference_contexts": ["Other income, net 9,022 1,034 7,988 Total other income, net $ 11,063 $ 2,573 $ 8,490",
                                "The change in Other income, net, compared to fiscal year 2025, was primarily driven by unrealized gains in non-marketable and publicly-held equity securities, including gains from our previously announced investment in Intel\u2019s common stock."],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "mdna", "difficulty": "hard", "question_type": "analytical"},
    },
    {
        "question": "What U.S. export restrictions affected NVIDIA, and which of its chips were specifically impacted?",
        "ground_truth": "In August 2022, the U.S. government announced export restrictions and export licensing requirements targeting China's semiconductor and supercomputing industries. The restrictions specifically impact NVIDIA's A100 and H100 integrated circuits and systems or boards that incorporate them.",
        "reference_contexts": ["In August 2022, the U.S. government, or USG, announced export restrictions and export licensing requirements targeting China",
                                "to China (including Hong Kong and Macau) and Russia, and specifically impact our A100 and H100 integrated circuits"],
        "metadata": {"company": "NVDA", "filing_year": 2026, "section": "risk_factors", "difficulty": "medium", "question_type": "factual"},
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
        "reference_contexts": ["AWS Net sales $ 90,757 $ 107,556 $ 128,725 Operating expenses 66,126 67,722 83,119 Operating income $ 24,631 $ 39,834 $ 45,606"],
        "metadata": {"company": "AMZN,MSFT", "filing_year": 2026, "section": "financial_statements", "difficulty": "hard", "question_type": "comparative"},
    },
    {
        "question": "How do NVIDIA and Google differ in how they report segment revenue?",
        "ground_truth": "NVIDIA reports Compute & Networking and Graphics segments, while Google reports Google Services and Google Cloud segments built around advertising and cloud.",
        "reference_contexts": [
            "Compute & Networking $ 193,479 $ 116,193 $ 77,286 %",
            "Graphics 22,459 14,304 8,155 %",
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
    # Self-check: every question has ground_truth + metadata
    assert all(item.get("question") and item.get("ground_truth") for item in QA), "missing q/gt"
    questions = [item["question"] for item in QA]
    assert len(questions) == len(set(questions)), "duplicate questions"
    for key in ("company", "difficulty", "question_type"):
        counts = {}
        for item in QA:
            v = item["metadata"].get(key, "?")
            counts[v] = counts.get(v, 0) + 1
        print(f"by {key}:", counts)

    # Every reference_context must appear verbatim (whitespace-collapsed)
    # somewhere in the source filings.
    with open(config.processed_dir / "filings.json", encoding="utf-8") as f:
        blobs = [re.sub(r"\s+", " ", doc["content"]) for doc in json.load(f)]
    missing = []
    for i, item in enumerate(QA):
        for rc in item["reference_contexts"]:
            probe = re.sub(r"\s+", " ", rc)
            if not any(probe in blob for blob in blobs):
                missing.append((i, rc[:80]))
    if missing:
        print("reference_contexts NOT found verbatim in filings:")
        for i, rc in missing:
            print(f"  [{i}] {rc}")
    assert not missing, f"{len(missing)} reference_contexts not found in filings"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(QA, f, indent=2, ensure_ascii=False)

    n_contexts = sum(len(item["reference_contexts"]) for item in QA)
    print(f"wrote {len(QA)} QA pairs ({n_contexts} reference contexts) -> {OUT}")
    assert len(QA) >= 100, "not enough QA pairs (target 100+)"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())