# Local engineering evidence

`local-evidence-20261023.tar.gz` contains complete synthetic local demo and control
archives (including dev/test audit data) and executed verification logs. These
are reviewer data, never candidate-generation prompts or training data.

`manifest.json` records checksums and the implementation hash. Extract to a new
folder, then use this exact source version:

```sh
mkdir -p /tmp/rsi-review-evidence
tar -xzf evidence/local-evidence-20261023.tar.gz -C /tmp/rsi-review-evidence
PYTHONPATH=src python3 -m beyond_discovery.run replay --domain stateful-scm --out /tmp/rsi-review-evidence/demo/stateful-scm
PYTHONPATH=src python3 -m beyond_discovery.run replay --domain string-transducer --out /tmp/rsi-review-evidence/demo/string-transducer
```

No live LLM, neural training or external benchmark run is present. Model fixtures
are explicitly tests of protocols and failures. No H1-H4 conclusion is supported.
