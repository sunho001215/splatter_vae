# arm4 final analysis (2026-11-11)

Runs the FROZEN arm4 analysis (reports/results-artifact/src/arm4_rl.py, arm4_render_metrics.py, unchanged) on the
complete campaign and produces the artifact sections.

- `run_frozen.py` - executes the frozen scripts byte-for-byte; maps the two post-registration-edited provenance files
  (arch_compare ranking.py / rl_common.py) to their reconstructed registered bytes in `frozen_inputs/` (sha256 equal
  to the registration). Run against a scratch mirror (runs/, experiments/, s4d/ symlinked) so the registered
  reference-only outputs in reports/results-artifact/arm4/ stay untouched.
- `supplementary.py` - descriptive stability and cost measures (not registered endpoints).
- `figures.py` - figures from the JSON outputs (report venv with matplotlib).
- `pages.py` - HTML sections filled only from rl.json / render_metrics.json / supplementary.json.
- `integrate.py` - inserts the sections into the LIVE published hub / RL / rendering pages (previous content kept).

Final outputs and Codex reviews: /home/ws/ws/validation-20261031/arm4final/ (final-20261111/, codex/c1f-out,
codex/c2-out; Codex = *.log.last.md, Claude draft = recommendation-draft.claude.md).
