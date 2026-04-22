# Session: 2026-04-17 #001

## Goal
Implement a simple local prompt-building web app for sequential statement-extraction workflow, with no LLM API calls, and include markdown logging of prompts, inputs, outputs, datetime, and model used.

## Prompts Summary
- User asked to implement the app in experimentation/testset-generation/know your rag based prompting.
- User asked for a markdown file that captures all built prompts, inputs, outputs, datetime, and model used.

## Actions Taken
- Created experimentation/testset-generation/know your rag based prompting/prompt_builder_app.py.
- Implemented Step 1 to Step 4 sequential prompt generation using guide-aligned templates.
- Added paste fields for external LLM outputs at each step, with state chaining across steps.
- Added model metadata input and markdown report generation containing prompts/inputs/outputs plus timestamp.
- Added download and local save options for generated markdown reports to prompt_runs/.
- Created experimentation/testset-generation/know your rag based prompting/prompt_run_log_template.md as a reusable manual template.
- Ran python syntax validation with py_compile on the new Streamlit app.

## Outcome
Implemented and validated a working single-file Streamlit prompt builder in the requested folder, plus a markdown run-log template and in-app report export/save workflow.

## Agent
GitHub Copilot (GPT-5.3-Codex)
