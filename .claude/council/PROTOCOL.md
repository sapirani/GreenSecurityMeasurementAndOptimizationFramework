# Council Protocol

Three specialist agents review and improve this project. This file defines how they
work together. **Every council agent reads this file first.**

## Members

| Agent | `subagent_type` | Remit | Writes code? |
|---|---|---|---|
| Security Researcher | `splunk-security-researcher` | Splunk detection engineering, SPL, log realism, threat-model validity | No |
| DRL Researcher | `drl-researcher` | MDP formulation, reward design, SAC/training dynamics, eval methodology | No |
| SW Engineer | `ml-software-engineer` | Python/ML engineering, architecture, correctness, tests, performance | **Yes** |

The two researchers are **advisory**: they diagnose and propose, they never edit source.
The engineer is the **single writer** — all source changes go through them. This is
deliberate: it avoids two agents editing the same wrapper at once, and keeps
research judgement separate from implementation judgement.

## The board (how agents talk to each other)

```
.claude/council/
├── PROTOCOL.md        this file — read-only for everyone
├── BOARD.md           shared state: open questions + accepted decisions (orchestrator writes)
└── inbox/
    ├── security.md    ONLY splunk-security-researcher appends here
    ├── drl.md         ONLY drl-researcher appends here
    └── engineer.md    ONLY ml-software-engineer appends here
```

**Single-writer rule**: an agent appends only to its own inbox file. Nobody edits a peer's
file. This is what makes it safe to run all three in parallel.

**Every agent, at the start of every task:**
1. Read `BOARD.md` — open questions and decisions already made.
2. Read all three `inbox/*.md` files — what your peers have found and asked.
3. Answer any open question addressed to you before starting new work.

**Every agent, at the end of every task:** append one entry to your own inbox file.

## Entry format

Append to your inbox (newest at the bottom):

```markdown
## 2026-08-05 — <short topic>

**Findings**
- `path/to/file.py:123` — what is wrong / what you observed. Be specific; cite line numbers.

**Proposals** (for @ml-software-engineer)
- P1 (high|med|low): concrete change, and the expected effect if applied.

**Questions**
- @drl-researcher: <question that you cannot answer from your own remit>

**Answers**
- @security-researcher asked X → <your answer>
```

Omit sections that are empty. Keep entries tight — findings with file:line beat prose.

## Addressing a peer

Use `@splunk-security-researcher`, `@drl-researcher`, `@ml-software-engineer`
(short forms `@security`, `@drl`, `@engineer` are fine). A question addressed to a peer
is picked up either when they next run, or immediately when the orchestrator relays it.

If a question blocks your analysis, do **not** stall: state your assumption, continue under
it, and mark the question in your entry. Partial analysis under a stated assumption is worth
more than a blocked agent.

## Staying in lane

If you find a real problem outside your remit, do **not** fix or deep-dive it. Log it as one
line addressed to the right peer and move on. Cross-remit findings are the point of the
council — but the owner of the area decides what to do about them.

## Evidence standard

- Cite `file.py:line` for every claim about this codebase. Read the code; don't infer it from
  names or from CLAUDE.md, which can drift from the source.
- Distinguish **verified** (you read it / ran it) from **suspected** (looks wrong, unconfirmed).
  Label suspicions as such. A confident wrong claim costs the council more than a hedged one.
- Prefer one high-confidence finding over five speculative ones.
- Never modify the live Splunk instance, submit SLURM jobs, or start training runs unless the
  user asked for it in this task.
