# C4: Controller, baselines, budgets, Replay and frozen final evaluation

APPS smoke and suite always run TRAIN/DEV and keep the official test sealed. Core
Controller refuses direct APPS official-test access. The final command validates
every supplied policy's completed smoke, source implementation, Judge, dataset,
model and budget before it opens any test task; it then records a one-shot final
claim against every source. Final evaluation uses frozen Scientist/Challenger
snapshots only and cannot adapt or retry.

Direct, Self-Repair, fixed Meta RSI and evolved Meta RSI use the same episode call,
token and Judge ceilings, splits, seed, parent opportunity and Judge identity.
The archive reports actual use, so generated tokens and Judge launches need not
match. Model completions are archived with requests, roles, version identity and
usage. Replay verifies source/runtime context, consumes only these completions,
re-executes the safe control workflow and actual Judge, and checks exact evidence.
It issues no model request.

Controller evaluation archives every completed episode immediately. For failed
calls it retains ambiguous HTTP reservations, actual reported provider overages,
completed Judge work, failures and per-episode usage. Judge reservations are made
before touching tasks. Final evaluation is separately budgeted against the frozen
run's original limits.

Added official APPS data lock from the real Berkeley archive SHA256
`6ef8e98ecca10b0159df0da4b524ecc1ca782a3b9473c57fc547ebccbbc2d0ca`. Its task
manifest covers 5,000 official train and 5,000 official test task directories.
Train questions are hash-grouped into 4,042 TRAIN / 958 DEV tasks. The list locks
question/reference/test content hashes without publishing those data. 551 tasks
with missing, empty/mismatched or oversized official inputs are present and marked
CAPABILITY_UNSUPPORTED. They do not become false WA/AC. APPS may occur in model
pretraining and is not a private uncontaminated benchmark.

The independent comparison imports the original MIT APPS evaluator source at
`362aedc3c71cd7d9bd2fc96a6c80e11dbc38c7a5`, verifies its SHA256, extracts the
official comparison logic and stdin wrapper, and runs code only in a digest-pinned
container. Live official-evaluator parity uses trusted golden stdin/call/class/WA/
RE programs and a documented pyext RuntimeModule compatibility shim. It is an
optional test and was skipped because no reviewed OCI image or engine is
available. No candidate Python was executed on the host.

Validation: the final APPS unit subset and complete regression results are in the
root report. Smoke, metapolicy accept/rollback path, frozen final path, final
single-use guard, token and Judge accounting, offline Replay, dataset lock,
unsupported official tasks, and protocol conformance all use explicitly named
protocol fixtures where stated. They demonstrate engineering paths only. Actual
provider/model token charges, inference, Judge output parity, OCI security, Oracle
certification on a real task, official-test execution, and performance are
unverified. No APPS score or scientific gain is claimed.
