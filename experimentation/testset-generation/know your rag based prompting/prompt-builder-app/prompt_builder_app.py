from __future__ import annotations

from datetime import datetime
from pathlib import Path
import logging
import re
import sys

import json

import streamlit as st


STEP_1_TEMPLATE = """In a few words, extract the main theme behind the following passage:
[[{context}]]"""

STEP_2_TEMPLATE = """Extract at most five factual statements based on the following passage and its theme. You need to strictly comply with the following guidelines:

* Each statement must be written in the style of an answer to a factual question.
* Each statement must be understandable without the aid of any other source of information.
* Each statement must include contextual information derived from the passage theme.
* Each statement must only contain information that exists in the original passage and theme.
* Each statement must be independent from the other statements.

Generate the statements as a bullet list with the following format:
> Statement
> Statement
etc

Theme: [[{theme}]]
Passage: [[{context}]]"""

STEP_3A_TEMPLATE = """Merge the following sentences into three summary statements.

* Each summary statement must summarise information contained in more than one sentence.
* Each summary statement must be independent and non-overlapping.
* Each summary statement should be a complete sentence.
* Each summary statement can include contextual information contained in the theme below.
* Each summary statement must be understandable without the aid of any other source of information.

Generate the statements as a bullet list with the following format:
> Summary statement
> Summary statement
> Summary statement

Theme: [[{theme}]]
Sentences: [[{statements}]]"""

STEP_3B_TEMPLATE = """A reasoning conclusion is an inferred piece of information obtained from critically analysing a group of multiple statements. Reasoning conclusions do not contain information directly contained on any statements.

Generate three reasoning conclusions that can be drawn from the following statements.

* Each conclusion must be independent and non-overlapping.
* Each conclusion should be a complete sentence.
* Each conclusion must be understandable without the aid of any other source of information.
* Each conclusion can include contextual information contained in the theme below.

Generate the conclusions as a bullet list with the following format:
> conclusion
> conclusion
> conclusion
etc

Theme: [[{theme}]]
Statements: [[{statements}]]"""

STEP_4_TEMPLATE = """I have a paragraph with the following theme:
[[{theme}]]

From this paragraph, I extracted the following statement:
[[{statement}]]

Generate one question which is answered only by the statement above.

In order to avoid generic questions, use contextual information from the theme to formulate the question.

The question should be concise and in the style of a user asking questions to a search engine.

Generate the question as a bullet list with the following format:
> Question

Do not output anything else other than the question."""


AUTOSAVE_PATH = Path(__file__).parent / "autosave_session.json"

SESSION_KEYS = {
    "context": "",
    "model_used": "",
    "step1_prompt": "",
    "theme_output": "",
    "step2_prompt": "",
    "factual_output": "",
    "step3a_prompt": "",
    "step3a_output": "",
    "step3b_prompt": "",
    "step3b_output": "",
    "selected_statement": "",
    "step4_prompt": "",
    "final_question_output": "",
    "step4_items": [],
    "report_markdown": "",
    "last_saved_report_path": "",
}


def _save_session() -> None:
    data = {key: st.session_state.get(key, default) for key, default in SESSION_KEYS.items()}
    AUTOSAVE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_session() -> None:
    if not AUTOSAVE_PATH.exists():
        return
    try:
        data = json.loads(AUTOSAVE_PATH.read_text(encoding="utf-8"))
        for key, default in SESSION_KEYS.items():
            st.session_state[key] = data.get(key, default)
    except Exception:
        pass


def init_state() -> None:
    if "_initialized" not in st.session_state:
        _load_session()
        st.session_state["_initialized"] = True
        return
    for key, default in SESSION_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def fill_from_previous_step() -> None:
    if not st.session_state["selected_statement"]:
        st.session_state["selected_statement"] = (
            st.session_state["step3a_output"]
            or st.session_state["step3b_output"]
            or st.session_state["factual_output"]
        ).strip()


def _clean_statement(value: str) -> str:
    return re.sub(r"^\s*(?:[-*>]|\d+[.)])\s*", "", value).strip()


def _extract_statements(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    blocks = [_clean_statement(part) for part in re.split(r"\n\s*\n", cleaned) if _clean_statement(part)]
    if len(blocks) > 1:
        return blocks

    lines = [_clean_statement(line) for line in cleaned.splitlines() if _clean_statement(line)]
    return lines


def _build_step4_prompt(theme: str, statement: str) -> str:
    return STEP_4_TEMPLATE.format(theme=theme.strip(), statement=statement.strip())


def _ensure_item_schema(item: dict) -> dict:
    return {
        "source": item.get("source", "manual"),
        "statement": item.get("statement", ""),
        "prompt": item.get("prompt", ""),
        "question": item.get("question", ""),
    }


def _all_question_records() -> list[dict]:
    records = [_ensure_item_schema(item) for item in st.session_state.get("step4_items", [])]

    single_statement = st.session_state.get("selected_statement", "").strip()
    single_prompt = st.session_state.get("step4_prompt", "").strip()
    single_question = st.session_state.get("final_question_output", "").strip()
    if single_statement and single_prompt:
        records.append(
            {
                "source": "single_step4",
                "statement": single_statement,
                "prompt": single_prompt,
                "question": single_question,
            }
        )

    return records


def _markdown_cell(value: str) -> str:
    compact = " ".join(value.split())
    return compact.replace("|", "\\|")


def build_report_markdown() -> str:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    step3_variants: list[str] = []
    if st.session_state["step3a_prompt"] or st.session_state["step3a_output"]:
        step3_variants.append("Summary (Step 3A)")
    if st.session_state["step3b_prompt"] or st.session_state["step3b_output"]:
        step3_variants.append("Reasoning (Step 3B)")
    variants_label = ", ".join(step3_variants) if step3_variants else "Not generated"

    question_records = _all_question_records()
    saved_questions = [record for record in question_records if record["question"].strip()]

    table_lines = ["## Questions Table"]
    table_lines.append("| # | Source | Theme | Statement | Question |")
    table_lines.append("|---|---|---|---|---|")
    if not question_records:
        table_lines.append("| 1 | Not available | Not available | Not available | Not available |")
    else:
        theme_value = st.session_state["theme_output"] or "Not provided"
        for idx, record in enumerate(question_records, start=1):
            table_lines.append(
                "| "
                + str(idx)
                + " | "
                + _markdown_cell(record["source"] or "Not available")
                + " | "
                + _markdown_cell(theme_value)
                + " | "
                + _markdown_cell(record["statement"] or "Not available")
                + " | "
                + _markdown_cell(record["question"] or "Not provided")
                + " |"
            )

    questions_table = "\n".join(table_lines)

    question_section_lines = ["## Step 4 Question Records"]
    question_section_lines.append(f"- Total Step 4 records: {len(question_records)}")
    question_section_lines.append(f"- Records with pasted questions: {len(saved_questions)}")

    if not question_records:
        question_section_lines.append("- No Step 4 records captured yet.")
    else:
        for idx, record in enumerate(question_records, start=1):
            question_section_lines.append("")
            question_section_lines.append(f"### Record {idx} ({record['source']})")
            question_section_lines.append("Statement:")
            question_section_lines.append(record["statement"] or "Not provided")
            question_section_lines.append("")
            question_section_lines.append("Step 4 Prompt:")
            question_section_lines.append(record["prompt"] or "Not generated")
            question_section_lines.append("")
            question_section_lines.append("Question Output:")
            question_section_lines.append(record["question"] or "Not provided")

    question_section = "\n".join(question_section_lines)

    return f"""# Prompt Building Run Log

## Metadata
- Datetime: {timestamp}
- Model used: {st.session_state['model_used'] or 'Not provided'}
- Step 3 variants: {variants_label}

## Inputs
### Initial Context
{st.session_state['context'] or 'Not provided'}

### Identified Theme
{st.session_state['theme_output'] or 'Not provided'}

### Theme Output (from Step 1 model run)
{st.session_state['theme_output'] or 'Not provided'}

### Factual Statements Output (from Step 2 model run)
{st.session_state['factual_output'] or 'Not provided'}

### Step 3A Output (summary statements from model run)
{st.session_state['step3a_output'] or 'Not provided'}

### Step 3B Output (reasoning conclusions from model run)
{st.session_state['step3b_output'] or 'Not provided'}

### Selected Statement for Step 4
{st.session_state['selected_statement'] or 'Not provided'}

### Final Question Output (from Step 4 model run)
{st.session_state['final_question_output'] or 'Not provided'}

## Built Prompts
### Step 1 Prompt: Theme Identification
{st.session_state['step1_prompt'] or 'Not generated'}

### Step 2 Prompt: Factual Statement Extraction
{st.session_state['step2_prompt'] or 'Not generated'}

### Step 3A Prompt: Summary Transformation
{st.session_state['step3a_prompt'] or 'Not generated'}

### Step 3B Prompt: Reasoning Transformation
{st.session_state['step3b_prompt'] or 'Not generated'}

### Step 4 Prompt: Question Generation
{st.session_state['step4_prompt'] or 'Not generated'}

{question_section}

{questions_table}
"""


def reset_all() -> None:
    for key, default in SESSION_KEYS.items():
        st.session_state[key] = default


def app() -> None:
    st.set_page_config(page_title="RAG Prompt Builder", layout="wide")
    init_state()

    st.title("Statement Extraction Prompt Builder")
    st.caption("Build sequential prompts only. No LLM API calls are made by this app.")

    autosave_exists = AUTOSAVE_PATH.exists()
    if st.button(
        "Clear autosave data" + (" ✓" if autosave_exists else " (no autosave)"),
        type="secondary",
        disabled=not autosave_exists,
    ):
        st.session_state["_confirm_clear"] = True

    if st.session_state.get("_confirm_clear"):
        st.warning("This will permanently delete the autosave file and reset all fields. This cannot be undone.")
        col_yes, col_no = st.columns([1, 4])
        with col_yes:
            if st.button("Yes, clear everything", type="primary"):
                AUTOSAVE_PATH.unlink(missing_ok=True)
                reset_all()
                st.session_state["_confirm_clear"] = False
                st.rerun()
        with col_no:
            if st.button("Cancel"):
                st.session_state["_confirm_clear"] = False
                st.rerun()

    st.divider()

    st.subheader("Run Metadata")
    st.session_state["model_used"] = st.text_input(
        "Model used for external runs (for logging)",
        value=st.session_state["model_used"],
        placeholder="Example: gemini-2.5-flash",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Reset all fields", type="secondary"):
            reset_all()
            st.rerun()
    with col_b:
        if st.button("Prefill Step 4 statement from previous output", type="secondary"):
            fill_from_previous_step()
            st.rerun()

    st.divider()

    st.header("Step 0: Paste Initial Context")
    st.session_state["context"] = st.text_area(
        "Context",
        value=st.session_state["context"],
        height=180,
        placeholder="Paste the source passage/context here.",
    )

    st.divider()

    st.header("Step 1: Theme Identification")
    if st.button("Generate Step 1 Prompt"):
        st.session_state["step1_prompt"] = STEP_1_TEMPLATE.format(
            context=st.session_state["context"].strip()
        )

    st.text_area(
        "Step 1 Prompt (copy and run in your LLM)",
        value=st.session_state["step1_prompt"],
        height=150,
    )

    st.session_state["theme_output"] = st.text_area(
        "Paste Step 1 output (theme)",
        value=st.session_state["theme_output"],
        height=80,
    )

    st.divider()

    st.header("Step 2: Factual Statement Extraction")
    if st.button("Generate Step 2 Prompt"):
        st.session_state["step2_prompt"] = STEP_2_TEMPLATE.format(
            theme=st.session_state["theme_output"].strip(),
            context=st.session_state["context"].strip(),
        )

    st.text_area(
        "Step 2 Prompt (copy and run in your LLM)",
        value=st.session_state["step2_prompt"],
        height=280,
    )

    st.session_state["factual_output"] = st.text_area(
        "Paste Step 2 output (factual statements)",
        value=st.session_state["factual_output"],
        height=140,
    )

    st.divider()

    st.header("Step 3 (Optional): Complex Question Statements")

    if st.button("Generate Step 3A and 3B Prompts"):
        st.session_state["step3a_prompt"] = STEP_3A_TEMPLATE.format(
            theme=st.session_state["theme_output"].strip(),
            statements=st.session_state["factual_output"].strip(),
        )
        st.session_state["step3b_prompt"] = STEP_3B_TEMPLATE.format(
            theme=st.session_state["theme_output"].strip(),
            statements=st.session_state["factual_output"].strip(),
        )

    col_3a, col_3b = st.columns(2)

    with col_3a:
        st.subheader("Step 3A: Summary")
        st.text_area(
            "Step 3A Prompt (copy and run in your LLM)",
            value=st.session_state["step3a_prompt"],
            height=260,
        )
        st.session_state["step3a_output"] = st.text_area(
            "Paste Step 3A output (summary statements)",
            value=st.session_state["step3a_output"],
            height=140,
        )

    with col_3b:
        st.subheader("Step 3B: Reasoning")
        st.text_area(
            "Step 3B Prompt (copy and run in your LLM)",
            value=st.session_state["step3b_prompt"],
            height=260,
        )
        st.session_state["step3b_output"] = st.text_area(
            "Paste Step 3B output (reasoning conclusions)",
            value=st.session_state["step3b_output"],
            height=140,
        )

    st.divider()

    st.header("Step 4: Question Generation")
    st.session_state["selected_statement"] = st.text_area(
        "Statement to convert into one question",
        value=st.session_state["selected_statement"],
        height=120,
        placeholder="Paste one chosen factual/summary/reasoning statement.",
    )

    if st.button("Generate Step 4 Prompt"):
        st.session_state["step4_prompt"] = _build_step4_prompt(
            theme=st.session_state["theme_output"],
            statement=st.session_state["selected_statement"],
        )

    st.text_area(
        "Step 4 Prompt (copy and run in your LLM)",
        value=st.session_state["step4_prompt"],
        height=230,
    )

    st.session_state["final_question_output"] = st.text_area(
        "Paste Step 4 output (final question)",
        value=st.session_state["final_question_output"],
        height=100,
    )

    col_step4_a, col_step4_b = st.columns(2)
    with col_step4_a:
        if st.button("Add current Step 4 to records"):
            statement = st.session_state["selected_statement"].strip()
            if statement:
                st.session_state["step4_items"].append(
                    {
                        "source": "manual",
                        "statement": statement,
                        "prompt": _build_step4_prompt(st.session_state["theme_output"], statement),
                        "question": st.session_state["final_question_output"].strip(),
                    }
                )
                st.success("Added current Step 4 entry to records.")
            else:
                st.warning("Add a statement first before saving to records.")

    with col_step4_b:
        if st.button("Clear Step 4 records"):
            st.session_state["step4_items"] = []
            st.rerun()

    st.subheader("Batch Step 4 Generation")
    st.caption("Build prompts for all statements from Step 2, Step 3A, and Step 3B, then paste each generated question.")

    if st.button("Generate batch Step 4 records from Step 2/3 outputs"):
        seen: set[str] = set()
        generated_items: list[dict] = []

        source_map = [
            ("step2_factual", st.session_state["factual_output"]),
            ("step3a_summary", st.session_state["step3a_output"]),
            ("step3b_reasoning", st.session_state["step3b_output"]),
        ]

        for source, text in source_map:
            for statement in _extract_statements(text):
                key = statement.lower()
                if key in seen:
                    continue
                seen.add(key)
                generated_items.append(
                    {
                        "source": source,
                        "statement": statement,
                        "prompt": _build_step4_prompt(st.session_state["theme_output"], statement),
                        "question": "",
                    }
                )

        st.session_state["step4_items"] = generated_items
        st.success(f"Generated {len(generated_items)} Step 4 records.")

    if st.session_state["step4_items"]:
        st.markdown(f"**Step 4 Records: {len(st.session_state['step4_items'])}**")
        for idx, item in enumerate(st.session_state["step4_items"]):
            item = _ensure_item_schema(item)
            st.session_state["step4_items"][idx] = item

            with st.expander(f"Record {idx + 1}: {item['source']}", expanded=False):
                st.text_area(
                    f"Statement #{idx + 1}",
                    value=item["statement"],
                    height=90,
                    disabled=True,
                    key=f"step4_statement_{idx}",
                )
                st.text_area(
                    f"Step 4 Prompt #{idx + 1}",
                    value=item["prompt"],
                    height=220,
                    disabled=True,
                    key=f"step4_prompt_{idx}",
                )
                question_value = st.text_area(
                    f"Paste Question Output #{idx + 1}",
                    value=item["question"],
                    height=90,
                    key=f"step4_question_{idx}",
                )
                st.session_state["step4_items"][idx]["question"] = question_value

    st.divider()

    st.header("Markdown Run Log")
    if st.button("Build Markdown Report"):
        st.session_state["report_markdown"] = build_report_markdown()
        output_dir = Path(__file__).parent / "prompt_runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"prompt-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        output_path = output_dir / filename
        output_path.write_text(st.session_state["report_markdown"], encoding="utf-8")
        st.session_state["last_saved_report_path"] = str(output_path)
        st.success(f"Auto-saved report: {output_path}")

    st.text_area(
        "Generated Markdown",
        value=st.session_state["report_markdown"],
        height=360,
    )

    if st.session_state["report_markdown"]:
        if st.session_state["last_saved_report_path"]:
            st.caption(f"Last auto-saved report: {st.session_state['last_saved_report_path']}")

        st.download_button(
            label="Download markdown report",
            data=st.session_state["report_markdown"],
            file_name=f"prompt-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md",
            mime="text/markdown",
        )

        if st.button("Save markdown report in prompt_runs/"):
            output_dir = Path(__file__).parent / "prompt_runs"
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"prompt-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
            output_path = output_dir / filename
            output_path.write_text(st.session_state["report_markdown"], encoding="utf-8")
            st.success(f"Saved: {output_path}")

    _save_session()


def _is_running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
    logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)

    if not _is_running_in_streamlit():
        print("This app must be started with Streamlit.")
        print("Run:")
        print('  streamlit run "experimentation/testset-generation/know your rag based prompting/prompt_builder_app.py"')
        sys.exit(0)

    app()
