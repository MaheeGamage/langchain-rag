"""
Testset generation for the RAG system using Ragas single-hop query synthesis.

Loads documents from knowledge_ingestion/content/v1, builds a knowledge graph,
applies transforms (headline extraction, splitting, keyphrase extraction), then
generates a synthetic Q&A testset using SingleHopSpecificQuerySynthesizer.

Providers are driven by the same JUDGE_PROVIDER / JUDGE_EMBEDDING_PROVIDER env
vars used by the evaluation scripts — no extra config needed.

Usage:
    # From repo root
    source .venv/bin/activate
    python -m experimentation.testset-generation.generate_testset

    # Override number of samples
    TESTSET_SIZE=20 python -m experimentation.testset-generation.generate_testset

    # Save output path
    OUTPUT_PATH=my_testset.json python -m experimentation.testset-generation.generate_testset
"""

from __future__ import annotations

import os
import sys
import json
import pathlib

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import (
    apply_transforms,
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
)
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.llms import LangchainLLMWrapper

from evaluation.ragas.ragas_factory import (
    get_ragas_judge_llm,
    get_ragas_judge_embeddings,
)
from app.config import (
    JUDGE_PROVIDER,
    JUDGE_EMBEDDING_PROVIDER,
    JUDGE_LLM_MODEL,
    JUDGE_EMBEDDING_MODEL,
)

# ── Config ────────────────────────────────────────────────────────────────────
# CONTENT_ROOT = REPO_ROOT / "knowledge_ingestion" / "content" / "v1"
CONTENT_ROOT = REPO_ROOT / "knowledge_ingestion" / "content" / "test"
TESTSET_SIZE = int(os.getenv("TESTSET_SIZE", "10"))
OUTPUT_PATH = pathlib.Path(
    os.getenv("OUTPUT_PATH", str(pathlib.Path(__file__).parent / "testset.json"))
)

# File extensions to load as plain-text documents
TEXT_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}


# ── Document loading ──────────────────────────────────────────────────────────

def load_text_documents(root: pathlib.Path) -> list[dict]:
    """Recursively load all text-based documents under *root*.

    Returns a list of dicts with keys ``page_content`` and ``document_metadata``.
    PDFs and notebooks are skipped here — the ingest pipeline handles those
    separately; for testset generation we rely on the markdown/rst sources.
    """
    docs = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as exc:
            print(f"  [warn] Could not read {path.relative_to(REPO_ROOT)}: {exc}")
            continue
        if not content:
            continue
        docs.append({
            "page_content": content,
            "document_metadata": {
                "source": str(path.relative_to(REPO_ROOT)),
                "file_name": path.name,
            },
        })
    return docs


# ── Personas ──────────────────────────────────────────────────────────────────

PERSONAS = [
    Persona(
        name="Quantum Software Researcher",
        role_description=(
            "A researcher working on quantum software experiments who needs to "
            "understand how to track and reproduce quantum computing experiments "
            "using provenance metadata and MLflow."
        ),
    ),
    Persona(
        name="MLflow Practitioner",
        role_description=(
            "A data scientist or ML engineer familiar with MLflow who wants to "
            "apply experiment tracking best practices to quantum computing workflows."
        ),
    ),
    Persona(
        name="Quantum Computing Student",
        role_description=(
            "A graduate student learning about quantum software development who "
            "needs clear explanations of QProv fields, Qiskit APIs, and how to "
            "log quantum experiment metadata."
        ),
    ),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Judge LLM provider   : {JUDGE_PROVIDER}")
    print(f"Judge embed provider : {JUDGE_EMBEDDING_PROVIDER}")
    print(f"Content root         : {CONTENT_ROOT}")
    print(f"Testset size         : {TESTSET_SIZE}")
    print(f"Output path          : {OUTPUT_PATH}")
    print()

    # 1. Load documents
    print("Loading documents...")
    raw_docs = load_text_documents(CONTENT_ROOT)
    print(f"  Loaded {len(raw_docs)} documents")
    if not raw_docs:
        raise RuntimeError(f"No text documents found under {CONTENT_ROOT}")

    # 2. Build knowledge graph
    print("Building knowledge graph...")
    kg = KnowledgeGraph()
    for doc in raw_docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc["page_content"],
                    "document_metadata": doc["document_metadata"],
                },
            )
        )
    print(f"  {kg}")

    # 3. LLM / embeddings
    print("Initialising judge LLM and embeddings...")
    ragas_llm = get_ragas_judge_llm()
    ragas_embeddings = get_ragas_judge_embeddings()

    # 4. Apply transforms
    print("Applying knowledge graph transforms...")
    transforms = [
        HeadlinesExtractor(llm=ragas_llm, max_num=20),
        HeadlineSplitter(max_tokens=1500),
        KeyphrasesExtractor(llm=ragas_llm),
    ]
    apply_transforms(kg, transforms=transforms)
    print(f"  After transforms: {kg}")

    # 5. Query distribution
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=ragas_llm, property_name="headlines"), 0.5),
        (SingleHopSpecificQuerySynthesizer(llm=ragas_llm, property_name="keyphrases"), 0.5),
    ]

    # 6. Generate testset
    print(f"\nGenerating {TESTSET_SIZE} test samples...")
    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
        knowledge_graph=kg,
        persona_list=PERSONAS,
    )
    testset = generator.generate(
        testset_size=TESTSET_SIZE,
        query_distribution=query_distribution,
    )

    # 7. Save output
    df = testset.to_pandas()
    print(f"\nGenerated {len(df)} samples")
    print(df[["user_input", "reference", "synthesizer_name"]].to_string(index=False))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON in the same shape as evaluation/eval_dataset.json
    records = []
    for _, row in df.iterrows():
        records.append({
            "inputs": {"question": row["user_input"]},
            "expectations": {"expected_response": row["reference"]},
            "metadata": {
                "synthesizer": row.get("synthesizer_name", ""),
                "reference_contexts": (
                    row["reference_contexts"]
                    if isinstance(row.get("reference_contexts"), list)
                    else []
                ),
            },
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nTestset saved to {OUTPUT_PATH} ✓")


if __name__ == "__main__":
    main()
