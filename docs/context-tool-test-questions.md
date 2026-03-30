# Context Tool Test Questions

Use these prompts in chat/agent testing to validate `get_pot_context`, `get_change_history`, `get_file_owner`, and `get_decisions` behavior.

## Basic retrieval

1. What is the latest merged PR in this project?
2. Summarize PR #78 in 5 bullet points.
3. Who authored PR #78 and when was it merged?
4. List all commits included in PR #78.
5. What problem was PR #78 trying to solve?

## Comments and discussions

1. Show all review comments in PR #78.
2. What feedback was given on PR #78 and by whom?
3. Which files had review discussion in PR #78?
4. Were there unresolved concerns mentioned in PR #78 comments?
5. Summarize the review thread decisions for PR #78.

## Decisions and rationale

1. What decisions were taken in PR #48?
2. Why was this architecture choice made in PR #48?
3. What alternatives were rejected in PR #48 discussions?
4. Show decisions linked to `app/modules/parsing/graph_construction/parsing_controller.py`.
5. Show decisions for function `parse_directory`.

## Change history / ownership

1. Show change history for `app/main.py`.
2. Which PRs modified `app/modules/context_graph/tasks.py`?
3. Who is the likely owner of `app/modules/parsing/graph_construction/parsing_controller.py`?
4. Which developers touched `app/main.py` most often?
5. What are the latest PRs affecting function `parse_directory`?

## Robustness / negative tests

1. Show comments for PR #99999.
2. What decisions were made in PR #99999?
3. Show ownership for `non/existent/file.py`.
4. What is the latest PR in an empty project?
5. Give project context for keyword: `__definitely_not_present__`.

## Mixed intent tests

1. Get latest PR and all comments in that PR.
2. For PR #78, give summary + comments + decisions.
3. For file `app/main.py`, give owners + recent changes + linked decisions.
4. Compare PR #78 and PR #79 by intent and impact.
5. Which PR introduced this function and what comments discussed it?

