Show the current state of the three-agent council. Read-only — do not dispatch any agents.

Steps:
1. Read `.claude/council/BOARD.md` and all `.claude/council/inbox/*.md`.
2. Print a concise dashboard:
   - **Accepted decisions** and **open questions** from BOARD.md.
   - **Per agent** (security / drl / engineer): the latest 1–3 inbox entries, summarized —
     newest findings, open proposals (with P-level), and any unanswered `@`-addressed questions.
   - **Queued for the engineer**: proposals from the researchers not yet marked done in
     engineer.md.
   - **Cross-agent threads**: questions raised but not yet answered, and who owes whom.
3. End with a one-line suggestion of the most useful next council action (e.g. "greenlight P1,
   run engineer" or "security still owes drl an answer on X").

Do not modify any files. Do not run agents. This is a status view only.
