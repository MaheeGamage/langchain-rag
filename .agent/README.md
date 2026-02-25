# Agent Activity Logs

This directory stores structured records of agent sessions, prompts, and actions taken in this repository. All agents working in this codebase **must** create and maintain logs here.

---

## Directory Structure

```
.agent/
├── README.md                                       # This file — read before logging
├── sessions/                                       # One file per agent session
│   └── YYYY-MM-DD_session-NNN_short-summary.md
└── prompts/                                        # Optional: raw or summarized prompt history
    └── YYYY-MM-DD_prompts.md
```

---

## How to Create a Session Log

### 1. Name the file

Use the format: `sessions/YYYY-MM-DD_session-NNN_short-summary.md`

- `YYYY-MM-DD` — the date the session started
- `NNN` — a zero-padded sequence number for that day (001, 002, ...)
- `short-summary` — a kebab-case slug (3-5 words) describing what happened

Example: `sessions/2026-02-20_session-001_scaffold-agent-logging.md`

### 2. Use the session template

```markdown
# Session: YYYY-MM-DD #NNN

## Goal
<!-- One or two sentences describing what was asked or intended -->

## Prompts Summary
<!-- Bullet list of the key prompts or instructions given during this session -->
- ...

## Actions Taken
<!-- Bullet list of concrete actions: files created/edited/deleted, commands run, etc. -->
- ...

## Outcome
<!-- What was the result? Did it succeed? Any follow-up needed? -->

## Agent
<!-- Name or identifier of the agent that ran this session -->
```

### 3. Commit the log with your changes

Include the session log in the same commit (or a follow-up commit) as the code changes it describes. This keeps the audit trail aligned with `git log`.

---

## Rules

| Rule | Reason |
|---|---|
| One file per session | Keeps history granular and easy to diff |
| Commit logs with code | Ties agent activity to specific repo states |
| Never delete old logs | Logs are an audit trail, not a scratchpad |
| Be specific in "Actions Taken" | Vague entries are not useful for future agents |

---

## Example Session Log

```markdown
# Session: 2026-02-20 #001

## Goal
Scaffold the `.agent/` logging structure and document the conventions
for all future agents working in this repository.

## Prompts Summary
- User asked for the best way to store agent communication and prompt summaries.
- User asked for an initial record explaining how to create these logs to other agents.

## Actions Taken
- Created `.agent/README.md` with logging conventions and templates.
- Created `.agent/sessions/` directory for session logs.
- Created `.agent/prompts/` directory for prompt history.

## Outcome
Directory structure and documentation in place. Future agents can follow
this README to log their activity consistently.

## Agent
GitHub Copilot (Claude Sonnet 4.6)
```

---

## Prompt Logs (Optional)

If you want to preserve the raw or summarized prompt history for a session, create a file in `prompts/`:

```
prompts/YYYY-MM-DD_prompts.md
```

Format:

```markdown
# Prompts: YYYY-MM-DD

## Session 001
1. "best way to store summary of agent communication..."
2. "Can you create initial record explaining how to create these logs..."
```

---

## Querying Logs

Find all sessions that touched a specific file:

```bash
grep -rl "src/auth.ts" .agent/sessions/
```

List all sessions chronologically:

```bash
ls .agent/sessions/ | sort
```
