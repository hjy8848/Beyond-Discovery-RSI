# C1: APPS adapter and execution boundary

Implemented checksum-pinned APPS archive import, grouped train/dev partition,
sealed official test, stdin and function-call support, independent comparison
extracted from MIT-licensed official evaluator revision
`362aedc3c71cd7d9bd2fc96a6c80e11dbc38c7a5`, and digest-pinned OCI judge.

The judge sends inputs and submitted source into a nonroot, read-only,
networkless, seccomp-enabled container. Expected outputs remain in the host
comparator. Candidate Python never executes in ResearchSandbox or the host.
Container limits bound memory, CPU, processes, descriptors, input/output and wall
time; timeout cleanup removes the container, not just the Docker client.

Dataset import requires an operator-supplied archive SHA256 and records exact
membership. Public feedback uses explicitly reviewed public test sidecars only;
no hidden test is relabeled public. Reference solutions remain private.

Validation: seven adapter tests cover official-source comparator golden cases,
sealing, tampering, public projection, unavailable containers, immutable image
requirements, hardening arguments, safe wire types and failure/resource verdicts.
Protocol fixtures do not execute Python and are explicitly labeled as fixtures.
Live official-evaluator/container semantic parity and adversarial container escape
checks have NOT run: no Docker/Podman engine or reviewed judge image is available.
No real APPS archive or model execution is claimed by fixture tests.
