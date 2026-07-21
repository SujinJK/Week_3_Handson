"""Compares retrieval using the raw user question against retrieval using a
HyDE-rewritten hypothetical passage (see rag.py's hyde_rewrite()), to check
whether bridging the gap between casual phrasing and formal document style
actually helps retrieval.

Uses deliberately colloquial phrasings, far from the corpus's own wording --
eval_set.json's questions are already phrased close enough to the source
text that this technique has nothing to fix there.

Costs two extra Claude calls per question (the HyDE rewrite, plus whatever
the eval set's questions already cost elsewhere) -- not free.

Run:
    python -m eval.query_transformation_comparison
"""
import pathlib
import sys

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from rag import get_collection, hybrid_retrieve, hyde_rewrite  # noqa: E402

load_dotenv()

CASES = [
    {
        "question": "can they just boot me early on without much of a heads up",
        "expected_source": "employee_handbook.md",
    },
    {
        "question": "who do i tell if my company laptop goes missing",
        "expected_source": "security_policy.md",
    },
    {
        "question": "will my stuff get wiped right away if i switch to a smaller plan and have too much saved",
        "expected_source": "product_faq.md",
    },
]


def run_comparison(k: int = 3) -> None:
    client = anthropic.Anthropic()
    collection = get_collection()

    raw_hits_count = 0
    hyde_hits_count = 0
    for case in CASES:
        question = case["question"]

        raw_hits = hybrid_retrieve(collection, question, k=k)
        raw_sources = [h["source"] for h in raw_hits]
        raw_hit = case["expected_source"] in raw_sources
        raw_hits_count += raw_hit

        hypothetical = hyde_rewrite(client, question)
        hyde_hits = hybrid_retrieve(collection, hypothetical, k=k)
        hyde_sources = [h["source"] for h in hyde_hits]
        hyde_hit = case["expected_source"] in hyde_sources
        hyde_hits_count += hyde_hit

        if hyde_hit and not raw_hit:
            status = "IMPROVED"
        elif raw_hit and hyde_hit:
            status = "PASS"
        else:
            status = "FAIL"

        print(f"[{status}] {question!r}")
        print(f"       expected:            {case['expected_source']}")
        print(f"       raw question top-{k}:   {raw_sources}")
        print(f"       HyDE passage:        {hypothetical!r}")
        print(f"       HyDE-rewritten top-{k}: {hyde_sources}")
        print()

    total = len(CASES)
    print(f"Hit@{k} (raw question):         {raw_hits_count}/{total}")
    print(f"Hit@{k} (HyDE-rewritten query): {hyde_hits_count}/{total}")


if __name__ == "__main__":
    run_comparison()
