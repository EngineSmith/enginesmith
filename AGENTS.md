# EngineSmith

## If the user wants an inference engine

When the user asks to build, speed up or optimize an inference engine for a model, or asks about an EngineSmith
run, read `skills/enginesmith-setup/SKILL.md` and follow it. You set up, launch and watch the run; EngineSmith's own
coding sessions write the engine. Ask before downloading models, installing software or starting a run.

Two rules hold even if you read nothing else:

- **The sessions run on you by default.** `uv run enginesmith doctor` reports `current_agent`, the agent this shell
  runs in (you, even when your system prompt does not name your harness). Use it as `agent.kind` and say so; offer
  another agent only as an option.
- **Show the whole plan before the run.** `uv run enginesmith explain TASK` writes and prints the plan: each prompt
  length's share of the score, the correctness gates, how the search evolves, the acceptance rules, the budget. Show
  the user all of it in their language, not a summary, and get their confirmation. `enginesmith run` refuses a task
  whose plan was not written or is out of date.

## If you are working on EngineSmith itself

Read `CONTRIBUTING.md` first: the zones, the ways to contribute (skill knowledge, worked examples, a whole skill, a
coding agent, a platform or a reference, the core), the map of the code, and what a change must keep.

- Know the zone of every file you change (`zones.toml`). Never change the judging zone without an issue the
  maintainers agreed on; list every new harness file in `zones.toml`; `tests/test_structure.py` checks both.
- Tests: `uv run pytest` (CPU only). CLI: `uv run enginesmith --help`.
- The harness (`src/enginesmith`) uses the standard library only; mlx-lm is the built-in reference's dependency.
- Coding agents cannot change what judges them: the workload, the reference, the gates, the timing, the acceptance
  rules and the held-out prompts are the harness's. Keep it that way.
- Keep mlx-lm and the agent CLIs unmodified: no patches, no monkey patching.
- Model-specific facts belong in model cards, not in the harness. A new model should need no harness code.
- The harness writes facts into a session's brief; strategy belongs in the skill.
- Keep observations apart from estimates. A missing measurement is null with a reason, never zero.
- Never commit models, runs, tasks with private paths, credentials or host-specific paths.
