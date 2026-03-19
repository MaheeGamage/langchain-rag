# Session: 2026-03-19 #001

## Goal
Fix a ModuleNotFoundError when running evaluation/ragas/evals.py directly with Poetry.

## Prompts Summary
- User ran `poetry run python evaluation/ragas/evals.py` and got `ModuleNotFoundError: No module named 'app'`.

## Actions Taken
- Read evaluation/ragas/evals.py and identified incorrect project-root path insertion into `sys.path`.
- Updated path logic to use `Path(__file__).resolve().parents[2]`, which points to the repository root from evaluation/ragas/evals.py.
- Kept existing import flow intact and only changed the path bootstrap lines.
- Validated by loading the script top-level without running `main` via:
  - `poetry run python -c "import runpy; runpy.run_path('evaluation/ragas/evals.py', run_name='not_main'); print('evals.py imports OK')"`
- Confirmed no diagnostics errors in evaluation/ragas/evals.py.

## Outcome
The import-path issue is resolved. `app` imports now succeed when running the script from the repository root using the same invocation style.

## Agent
GitHub Copilot
