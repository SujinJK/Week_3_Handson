"""Measures generation quality against eval_set.json using an LLM-as-judge:
does each answer stick strictly to its retrieved context (faithfulness), and
does every citation actually support the claim it's attached to (citation
accuracy)?

Unlike eval/retrieval_eval.py, this calls Claude twice per question (once to
generate the answer, once to judge it), so it costs a small amount of API
credit and is not free to re-run at will.

Run:
    python -m eval.generation_eval
"""
import json
import pathlib
import sys

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from rag import MODEL, build_context_block, generate_answer, get_collection, retrieve  # noqa: E402

load_dotenv()

EVAL_SET_PATH = pathlib.Path(__file__).parent / "eval_set.json"

JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader checking whether an AI assistant's answer is "
    "properly grounded in the context it was given. You will see numbered "
    "context snippets, a question, and the assistant's answer. Check two "
    "things independently:\n"
    "1. faithful: does the answer use ONLY claims that are actually "
    "supported by the context snippets, with nothing added from outside "
    "knowledge? An honest 'I don't have enough information' answer is "
    "always faithful.\n"
    "2. citations_valid: for every [N] citation in the answer, does snippet "
    "N actually support the specific claim it's attached to? An answer with "
    "no citations because it correctly declined to answer counts as valid.\n"
    "Be strict — a claim that is directionally right but not actually "
    "stated in the cited snippet should fail citations_valid."
)

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithful": {"type": "boolean"},
        "citations_valid": {"type": "boolean"},
        "explanation": {"type": "string", "description": "One sentence justifying both verdicts"},
    },
    "required": ["faithful", "citations_valid", "explanation"],
    "additionalProperties": False,
}


def judge_answer(client: anthropic.Anthropic, context: str, question: str, answer: str) -> dict:
    """Ask Claude to grade another Claude response for faithfulness and citation accuracy."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAssistant's answer:\n{answer}",
        }],
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def run_eval(k: int = 3) -> None:
    eval_cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    client = anthropic.Anthropic()
    collection = get_collection()

    faithful_count = 0
    citations_valid_count = 0
    for case in eval_cases:
        question = case["question"]
        hits = retrieve(collection, question, k=k, where={"status": "current"})
        context = build_context_block(hits)
        answer = generate_answer(client, hits, question)
        verdict = judge_answer(client, context, question, answer)

        faithful_count += verdict["faithful"]
        citations_valid_count += verdict["citations_valid"]

        status = "PASS" if verdict["faithful"] and verdict["citations_valid"] else "FAIL"
        print(f"[{status}] {question}")
        print(f"       answer: {answer[:150]}{'...' if len(answer) > 150 else ''}")
        print(
            f"       faithful={verdict['faithful']}  citations_valid={verdict['citations_valid']}"
            f"  — {verdict['explanation']}"
        )

    total = len(eval_cases)
    print(f"\nFaithful: {faithful_count}/{total} ({faithful_count / total:.0%})")
    print(f"Citations valid: {citations_valid_count}/{total} ({citations_valid_count / total:.0%})")


if __name__ == "__main__":
    run_eval()
