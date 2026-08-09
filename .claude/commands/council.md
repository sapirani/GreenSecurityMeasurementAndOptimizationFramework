Run the three-agent council on a topic. You (main session) are the **orchestrator** — you
route between agents, keep the board, and synthesize for the user. The three agents are
`splunk-security-researcher`, `drl-researcher`, and `ml-software-engineer`, defined in
`.claude/agents/`, coordinating through `.claude/council/`.

Arguments: `$ARGUMENTS` = the topic/task. If empty, ask the user what to review.

Steps:
1. Read `.claude/council/BOARD.md` and `.claude/council/inbox/*.md` for current state.
2. Decide which agents the topic needs (usually the two researchers first; the engineer once
   there are concrete proposals). Dispatch them with the Agent tool, each `subagent_type` set
   to the matching agent, and a prompt that states the topic + "follow the council protocol."
   Run researchers in parallel (independent); run the engineer after, once proposals exist.
3. When agents finish, read their inbox entries. Relay any `@`-addressed cross-agent question
   to its target — either by messaging that agent (SendMessage if still alive) or noting it in
   BOARD.md's "Open questions" for the next round.
4. Update `.claude/council/BOARD.md`: append a Round-log line, move resolved items, and record
   any decision the user approves under "Accepted decisions."
5. Give the user a synthesized summary: key findings per agent, points of agreement/tension
   between the researchers, and the concrete proposals now queued for the engineer. Do not
   dump raw inbox files — synthesize. Ask the user which proposals to greenlight before the
   engineer implements anything non-trivial.

Rules:
- Only the engineer edits source. If the user wants changes applied, route them through the
  engineer agent, not yourself, unless the user tells you to do it directly.
- Never launch training/SLURM/live-Splunk actions on the agents' behalf unless the user asked.
- Keep the researchers read-only; if one proposes a fix, it goes to the engineer, not into code.
