from string import Template

# V1 — original prompt; mentions both domains explicitly which can bias rewrites
# toward injecting quantum terms into unrelated MLflow queries.
REWRITE_PROMPT_V1 = Template(
    "You are a search query optimizer. Rewrite the user's question into a concise, "
    "keyword-rich query optimized for semantic and keyword search over a knowledge base "
    "about MLflow experiment tracking and quantum software development.\n\n"
    "Rules:\n"
    "- Return ONLY the rewritten query, nothing else.\n"
    "- Keep it short (one sentence or phrase).\n"
    "- Preserve domain-specific terms.\n\n"
    "User question: $question"
)

# V2 — strips context-framing clauses ("in the context of X", "for X domain") which
# caused V1 to produce cross-domain embeddings that missed single-domain documents.
REWRITE_PROMPT_V2 = Template(
    "You are a search query optimizer. Extract the core information need from the "
    "user's question and rewrite it as a concise, keyword-rich search query.\n\n"
    "Rules:\n"
    "- Return ONLY the rewritten query, nothing else.\n"
    "- Keep it short (one sentence or phrase).\n"
    "- Preserve technical terms and API names.\n"
    "- Strip context-framing clauses like 'in the context of X', 'for X application', "
    "'in X domain' — focus on what information is being sought, not where it will be used.\n\n"
    "User question: $question"
)

# V3 — extends V2 with retrieval-profile selection.  The helper LLM now emits
# both a rewritten query AND a named profile (default/acronym/conceptual/
# overview/reasoning) so the graph can pick a retriever tuned to the question.
REWRITE_PROMPT_V3 = Template(
    "You are a search query optimizer. Rewrite the user's question into a concise, "
    "keyword-rich search query AND select the retrieval strategy that best fits it.\n\n"
    "Rules:\n"
    "- Preserve technical terms and API names (e.g. mlflow.log_param, NISQ, VQE).\n"
    "- Strip context-framing clauses like 'in the context of X', 'for X application' "
    "— focus on what information is being sought, not where it will be used.\n"
    "- Return EXACTLY two lines, nothing else:\n"
    "  QUERY: <rewritten query>\n"
    "  PROFILE: <one of: default, acronym, conceptual, overview, reasoning, reranked>\n\n"
    "Profile menu:\n"
    "- default:    balanced 50/50 hybrid; use when unsure.\n"
    "- acronym:    BM25-heavy; for acronyms (NISQ, VQE, QAOA) or exact API names "
    "(mlflow.log_param, start_run).\n"
    "- conceptual: semantic-heavy; for definitions, explanations, 'what is X' questions.\n"
    "- overview:   large k; for summary / listing / taxonomy questions needing broad context.\n"
    "- reasoning:  large k; for multi-hop 'why' / 'how does X affect Y' questions needing "
    "several supporting facts.\n"
    "- reranked:   wide retrieve + cross-encoder rerank; pick for hard / ambiguous queries "
    "where precision matters more than latency.\n\n"
    "User question: $question"
)

# Active version — switch by reassigning to REWRITE_PROMPT_V1, V2, or V3.
REWRITE_PROMPT = REWRITE_PROMPT_V3
