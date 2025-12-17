from __future__ import annotations
import os
from typing import Dict, Any

from langsmith import Client
from langsmith.schemas import Example, Run
from langsmith.evaluation import EvaluationResult

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.vector_db import get_retriever
from app.services.rag_chain import build_rag_chain

os.environ.setdefault("LANGSMITH_TRACING", "true")

client = Client()

K = 4
retriever = get_retriever(k=K)
rag_chain = build_rag_chain(model_name=settings.chat_model, k=K)

def answer_fn(question: str, history=None) -> Dict[str, Any]:
    """
    Simple function that takes a question and returns a dict
    with the model's answer and the retrieved documents.
    This is the 'model under test' for evaluation.
    """
    docs = retriever.invoke(question)
    ai_msg = rag_chain.invoke(question)
    return {
        "answer": ai_msg.content,
        "documents": docs,
    }

def ensure_dataset() -> str:
    """
    Create a small evaluation dataset if it doesn't exist yet.
    Returns the dataset name.
    """
    dataset_name = "KB-QA Eval (simple)"
    existing = [d for d in client.list_datasets(dataset_name=dataset_name)]
    if existing:
        return dataset_name

    examples = [
        {
            "inputs": {"question": "What is the dress code policy in the company?"},
            "outputs": {
                "answer": "Short, reference description of dress code from the handbook."
            },
        },
        {
            "inputs": {
                "question": "Summarize the main responsibilities described in the operations guide."
            },
            "outputs": {
                "answer": "Short, reference summary of responsibilities."
            },
        },
        {
            "inputs": {"question": "What does the employee handbook say about PTO?"},
            "outputs": {
                "answer": "Short, reference description of PTO sections and types of leave."
            },
        },
    ]

    ds = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(dataset_id=ds.id, examples=examples)
    return dataset_name

grader_llm = ChatOpenAI(model="gpt-4o", temperature=0)

RELEVANCE_PROMPT = """
You are grading how well an answer addresses a question.

QUESTION:
{question}

ANSWER:
{answer}

Give:
- score = 1.0 if the answer clearly and directly addresses the question.
- score = 0.0 if it does not.
Also explain your reasoning briefly.
"""


def relevance(example: Example, run: Run) -> EvaluationResult:
    """
    Simple LLM-as-judge evaluator:

    - example.inputs['question']  -> the question
    - run.outputs['answer']       -> model answer (from answer_fn)
    Returns an EvaluationResult that LangSmith understands.
    """
    question = example.inputs["question"]
    answer = run.outputs["answer"]

    prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)

    judge_msg = grader_llm.invoke(prompt)
    explanation = judge_msg.content.strip()

    score = 1.0
    if "not relevant" in explanation.lower():
        score = 0.0

    return EvaluationResult(
        key="relevance",   
        score=score,       
        comment=explanation,  
    )

def run_experiment() -> None:
    print("Running evaluation experiment...\n")
    dataset_name = ensure_dataset()
    def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return answer_fn(inputs["question"])

    results = client.evaluate(
        target,                       
        data=dataset_name,            
        evaluators=[relevance],       
        experiment_prefix="kb-rag-eval-simple",
    )

    url = getattr(results, "experiment_url", None) or getattr(results, "url", None)
    print("\nEvaluation finished.")

    if url:
        print(f"LangSmith URL:\n{url}\n")
    else:
        exp_name = getattr(results, "experiment_name", None)
        exp_id = getattr(results, "experiment_id", None)
        print(f"Experiment name: {exp_name}")
        if exp_id:
            print(f"Experiment id: {exp_id}")
        print("\n(Your terminal already printed the compare URL above.)\n")


    try:
        df = results.to_pandas()
        print("\nFirst few rows:\n")
        print(df.head())
    except Exception:
        pass


if __name__ == "__main__":
    run_experiment()
