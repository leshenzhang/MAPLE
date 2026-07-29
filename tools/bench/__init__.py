"""maple-bench: reusable benchmark runner + parity gate for the MAPLE GPU-batch stack.

NOT part of the maple library. Lives in tools/bench/ on purpose (harness, not product).

Modules
-------
core.py         GradCounter / GPUSampler / ts1x loader / run-record schema (maple-bench-v1)
parity_gate.py  canon-vs-canon parity gate + same-saddle gate + negative-control self-test
run_matrix.py   CLI: {dispatcher} x {B} x {backend} matrix, replicates built in -> JSON per run
aggregate.py    run JSONs -> summary CSV + BASELINE json + delta-vs-baseline-spread verdicts

Iron rules encoded here (OPT_CAMPAIGN_2026-07-28.md section 1):
  * wall is ALWAYS paired with gradient-equivalents; s/grad-equiv is the primary metric
  * parity is canon-vs-canon (fresh-instance self-noise floor), never an f64 threshold
  * >=2 replicates before any ratio; spread is reported with every mean
  * gates return PASS / FAIL / SKIP -- a non-applicable check is SKIP, never PASS
"""
