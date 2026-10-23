# C2: coding research runtime

`CodingResearchRuntime` implements the existing runtime protocol independently of
SynthesisRuntime. Real HTTP/HF clients generate full Python submissions. The
existing research-control-v1 worker runs bounded control code; only DockerJudge
runs Python. Versioned hypothesis, experiment, public result and model invocation
artifacts bind source, task, parent, provenance, actual token counts and control
traces. An explicit public EvidenceBundle projection excludes private scores and
judge traces from candidate generation; dev/test evidence rejects agent_view.

Four new behavior/failure tests exercise changed code versions, public-feedback
repair, token accounting, offline generation-tape replay with request matching,
malformed code, closed sessions, dev isolation and enforced model-call caps.
These use labeled protocol fixtures, not real model inference or Python judging.
All 101 tests passed in the C2 regression run (see final validation report for
later runs). Real HTTP/HF + OCI execution remains unverified in this environment.
