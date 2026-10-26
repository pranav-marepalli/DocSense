---
description: Run the design -> implement -> review loop for one DocSense phase
argument-hint: <phase number>
---

Run phase $ARGUMENTS of DocSense using the subagents. You are the orchestrator; delegate the work rather than doing it yourself.

1. **Design**: use the architect subagent to write docs/design/phase-$ARGUMENTS.md. Then STOP, show me a summary of the design and its open questions, and wait for my approval or edits.
2. **Implement**: once I approve, use the implementer subagent to build it from the design doc.
3. **Review**: use the reviewer subagent on the result.
4. **Fix loop**: if the verdict is CHANGES REQUESTED, send the blocker and should-fix findings back to the implementer, then review again. Stop after 3 rounds and escalate to me if it still isn't approved.
5. **Wrap up**: when approved, give me a summary: what was built, the 3 design decisions I should be able to explain in an interview, and anything I should verify by hand. Do not commit; I will review and commit myself.
